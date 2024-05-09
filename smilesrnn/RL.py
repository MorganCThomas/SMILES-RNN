import copy
import os
import warnings

import torch
from promptsmiles import FragmentLinker, ScaffoldDecorator
from tqdm.auto import tqdm

from smilesrnn import utils
from smilesrnn.gated_transformer import Model as StableTransformerModel
from smilesrnn.rnn import Model as RNNModel
from smilesrnn.transformer import Model as TransformerModel

warnings.filterwarnings("ignore", category=UserWarning)


class ReinforcementLearning:
    def __init__(
        self,
        device,
        model,
        agent,
        scoring_function,
        save_dir,
        optimizer,
        learning_rate,
        is_molscore=True,
        psmiles=None,
        psmiles_shuffle=True,
        psmiles_multi=False,
        psmiles_optimize=False,
        psmiles_lr_decay=1,  # 1=Do not decay
        psmiles_lr_epochs=10,
        freeze=None,
    ):
        # Device
        self.device = device
        if model == "RNN":
            self.model = RNNModel
        elif model == "Transformer":
            self.model = TransformerModel
        else:  # GTr
            self.model = StableTransformerModel
        # Load agent
        self.agent = self.model.load_from_file(
            file_path=agent, sampling_mode=False, device=device
        )
        self.agent.max_sequence_length = 512
        # Initialize promptsmiles transform
        self.psmiles = psmiles
        self.psmiles_shuffle = psmiles_shuffle
        self.psmiles_multi = psmiles_multi
        self.psmiles_optimize = psmiles_optimize
        if psmiles:
            if isinstance(psmiles, list):
                self.psmiles_transform = FragmentLinker(
                    fragments=self.psmiles,
                    batch_size=64,  # Placecholder, overridden by sampling
                    sample_fn=self.agent._pSMILES_sample,
                    evaluate_fn=self.agent._pSMILES_evaluate,
                    batch_prompts=True,
                    optimize_prompts=self.psmiles_optimize,
                    shuffle=self.psmiles_shuffle,
                    scan=False,
                    return_all=True,
                )
            elif isinstance(psmiles, str):
                self.psmiles_transform = ScaffoldDecorator(
                    scaffold=self.psmiles,
                    batch_size=64,  # Placecholder, overridden by sampling
                    sample_fn=self.agent._pSMILES_sample,
                    evaluate_fn=self.agent._pSMILES_evaluate,
                    batch_prompts=True,
                    optimize_prompts=self.psmiles_optimize,
                    shuffle=self.psmiles_shuffle,
                    return_all=True,
                )
            else:
                raise ValueError(
                    "promptsmiles must be a list of fragment smiles or a scaffold smiles"
                )
        else:
            self.psmiles_transform = None
        # Scoring function
        self.scoring_function = scoring_function
        self.molscore = is_molscore
        self.save_dir = save_dir
        # Optimizer
        self.optimizer = self._initialize_optimizer(
            lr=learning_rate, optimizer=optimizer
        )
        self.scheduler = self._initialize_scheduler(
            start_factor=psmiles_lr_decay, total_iters=psmiles_lr_epochs
        )
        if freeze is not None:
            self._freeze_network(freeze)
        self.record = None

    def _initialize_optimizer(self, lr=5e-4, optimizer=torch.optim.Adam):
        return optimizer(self.agent.network.parameters(), lr=lr)

    def _initialize_scheduler(self, start_factor=0.1, total_iters=10):
        # NOTE If we're using prompt smiles, you can soften learning rate for 10 updates to facilitate smoother learning
        if self.psmiles_transform:
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=start_factor,
                total_iters=total_iters,
                verbose=False,
            )

    def train(self, n_steps, save_freq):
        for step in tqdm(range(n_steps), total=n_steps):
            self._train_step(step=step)
            # Save the agent weights every few iterations
            if step % save_freq == 0 and step != 0:
                self.agent.save(os.path.join(self.save_dir, f"Agent_{step}.ckpt"))
        # If the entire training finishes, clean up
        self.agent.save(os.path.join(self.save_dir, f"Agent_{n_steps}.ckpt"))
        return self.record

    def _freeze_network(self, freeze):
        n_freeze = freeze * 4 + 1
        for i, param in enumerate(self.agent.network.parameters()):
            if i < n_freeze:  # Freeze parameter
                param.requires_grad = False

    def _update(self, loss, verbose=False):
        loss = loss.mean()
        if verbose:
            print(f"    Loss: {loss.data:.03f}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def _train_step(self, step):
        raise NotImplementedError

    def _sample_batch(self, batch_size):
        if self.psmiles_transform:
            smiles = self.psmiles_transform.sample(batch_size=batch_size)
            return None, smiles, None, None, None, None
        else:
            # NOTE Old implementation supplies psmiles and psmiles_shuffle to sample_sequences_and_smiles
            seqs, smiles, agent_likelihood, probs, log_probs, critic_values = (
                self.agent.sample_sequences_and_smiles(batch_size)
            )
            return seqs, smiles, agent_likelihood, probs, log_probs, critic_values

    def _score(self, smiles, step):
        try:
            if self.psmiles:
                smiles = copy.deepcopy(smiles[-1])
                # NOTE SMILES is a list of lists per iteration, so we take the final complete SMILES
                if isinstance(self.psmiles, list):
                    linkers = [
                        utils.extract_linker(smi, self.psmiles) for smi in smiles
                    ]
                    scores = self.scoring_function(
                        smiles, step, additional_formats={"linker": linkers}
                    )
                else:
                    scores = self.scoring_function(smiles, step)
            else:
                scores = self.scoring_function(smiles, step)
            scores = utils.to_tensor(scores).to(self.device)
        except (Exception, BaseException, SystemExit, KeyboardInterrupt) as e:
            if self.molscore:
                # If anything fails, save smiles, agent, scoring_function etc.
                utils.save_smiles(
                    smiles,
                    os.path.join(
                        self.save_dir, f"failed_{self.scoring_function.step}.smi"
                    ),
                )
                self.agent.save(os.path.join(self.save_dir, f"Agent_{step}.ckpt"))
                self.scoring_function._write_temp_state(step=self.scoring_function.step)
                self.scoring_function.kill_monitor()
                raise e
            else:
                utils.save_smiles(
                    smiles, os.path.join(self.save_dir, f"failed_{step + 1}.smi")
                )
                self.agent.save(os.path.join(self.save_dir, f"Agent_{step}.ckpt"))
                raise e
        return scores

    def _filter_smiles(self, smiles):
        """Remove un-encodable SMILES, needed when using promptsmiles."""
        failed = []
        for i, smi in enumerate(smiles[-1]):
            try:
                _ = self.agent.vocabulary.encode(self.agent.tokenizer.tokenize(smi))
            except KeyError:
                failed.append(i)

        for it in smiles:
            for i in reversed(failed):
                it.pop(i)
        return smiles


class Reinforce(ReinforcementLearning):
    _short_name = "RF"

    def __init__(
        self,
        device,
        model,
        agent,
        scoring_function,
        save_dir,
        optimizer,
        learning_rate,
        is_molscore=True,
        psmiles=None,
        psmiles_shuffle=True,
        psmiles_multi=False,
        psmiles_optimize=False,
        psmiles_lr_decay=1,
        psmiles_lr_epochs=10,
        freeze=None,
        batch_size=64,
        **kwargs,
    ):
        super().__init__(
            device=device,
            model=model,
            agent=agent,
            scoring_function=scoring_function,
            save_dir=save_dir,
            optimizer=optimizer,
            learning_rate=learning_rate,
            is_molscore=is_molscore,
            psmiles=psmiles,
            psmiles_multi=psmiles_multi,
            psmiles_shuffle=psmiles_shuffle,
            psmiles_optimize=psmiles_optimize,
            psmiles_lr_decay=psmiles_lr_decay,
            psmiles_lr_epochs=psmiles_lr_epochs,
            freeze=None,
        )

        # Parameters
        self.batch_size = batch_size
        # Record
        self.record = {"loss": [], "agent_nll": []}

    def _train_step(self, step):
        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = (
            self._sample_batch(self.batch_size)
        )
        # Filter un-tokenized smiles
        if self.psmiles:
            smiles = self._filter_smiles(smiles)
        # Score
        scores = self._score(smiles, step)
        # Compute loss
        if self.psmiles:
            # NOTE seq, probs, log_probs, critic_values is None
            # NOTE we recompute gradients here as sampling is without graph
            if self.psmiles_multi:
                for i in range(len(smiles)):
                    agent_likelihood = -self.agent.likelihood_smiles(smiles[i])
                    loss = self._compute_loss(agent_likelihood, scores)
                    self._update(loss, verbose=False)
            else:
                # NOTE for scaffold decoration i = -1, for fragment linking i = 0
                if isinstance(self.psmiles, list):
                    i = 0
                else:
                    i = -1
                agent_likelihood = -self.agent.likelihood_smiles(smiles[i])
                loss = self._compute_loss(agent_likelihood, scores)
                self._update(loss, verbose=False)
        else:
            agent_likelihood = -agent_likelihood
            loss = self._compute_loss(agent_likelihood, scores)
            self._update(loss, verbose=False)

    def _compute_loss(self, agent_likelihood, scores):
        loss = agent_likelihood * scores
        # Update
        self.record["loss"] += list(loss.detach().cpu().numpy())
        self.record["agent_nll"] += list(agent_likelihood.detach().cpu().numpy())
        return loss


class ReinforceRegularized(ReinforcementLearning):
    _short_name = "RF-reg"

    def __init__(
        self,
        device,
        model,
        agent,
        scoring_function,
        save_dir,
        optimizer,
        learning_rate,
        is_molscore=True,
        freeze=None,
        prior=None,
        batch_size=64,
        entropy_coefficient=0,
        kl_coefficient=10,
        **kwargs,
    ):
        super().__init__(
            device,
            model,
            agent,
            scoring_function,
            save_dir,
            optimizer,
            learning_rate,
            is_molscore,
            freeze=None,
        )
        if self.psmiles:
            raise NotImplementedError(
                "ReinforceRegularized does not support promptsmiles yet"
            )
        # Load prior
        if prior is None:
            self.prior = self.model.load_from_file(
                file_path=agent, sampling_mode=True, device=device
            )
        else:
            self.prior = self.model.load_from_file(
                file_path=prior, sampling_mode=True, device=device
            )
        # Parameters
        self.batch_size = batch_size
        self.entropy_co = entropy_coefficient
        self.kl_co = kl_coefficient
        # Record
        self.record = {"loss": [], "agent_nll": []}

    def _train_step(self, step):
        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = (
            self._sample_batch(self.batch_size)
        )
        # Score
        scores = self._score(smiles, step)
        # Compute loss
        loss = (
            (agent_likelihood * scores)
            + (self.entropy_co * self.agent.entropy(seqs))
            + (self.kl_co * self.agent.kl(seqs, self.prior))
        )
        # Update
        self.record["loss"] += list(loss.detach().cpu().numpy())
        self.record["agent_nll"] += list(agent_likelihood.detach().cpu().numpy())
        self._update(loss, verbose=False)


class Reinvent(ReinforcementLearning):
    _short_name = "RV"

    def __init__(
        self,
        device,
        model,
        agent,
        scoring_function,
        save_dir,
        optimizer,
        learning_rate,
        is_molscore=True,
        psmiles=None,
        psmiles_multi=False,
        psmiles_shuffle=True,
        psmiles_optimize=False,
        psmiles_lr_decay=1,
        psmiles_lr_epochs=10,
        freeze=None,
        prior=None,
        batch_size=64,
        sigma=60,
        **kwargs,
    ):
        super().__init__(
            device=device,
            model=model,
            agent=agent,
            scoring_function=scoring_function,
            save_dir=save_dir,
            optimizer=optimizer,
            learning_rate=learning_rate,
            is_molscore=is_molscore,
            psmiles=psmiles,
            psmiles_multi=psmiles_multi,
            psmiles_shuffle=psmiles_shuffle,
            psmiles_optimize=psmiles_optimize,
            psmiles_lr_decay=psmiles_lr_decay,
            psmiles_lr_epochs=psmiles_lr_epochs,
            freeze=None,
        )

        # Load prior
        if prior is None:
            self.prior = self.model.load_from_file(
                file_path=agent, sampling_mode=True, device=device
            )
        else:
            self.prior = self.model.load_from_file(
                file_path=prior, sampling_mode=True, device=device
            )
        # Parameters
        self.batch_size = batch_size
        self.sigma = sigma
        # Record
        self.record = {"loss": [], "prior_nll": [], "agent_nll": []}

    def _train_step(self, step):
        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = (
            self._sample_batch(self.batch_size)
        )
        # Filter un-tokenized smiles
        if self.psmiles:
            smiles = self._filter_smiles(smiles)
        # Score
        scores = self._score(smiles, step)
        # Compute loss
        if self.psmiles:
            # NOTE seq, probs, log_probs, critic_values is None
            # NOTE we recompute gradients here as sampling is without graph
            if self.psmiles_multi:
                for i in range(len(smiles)):
                    agent_likelihood = -self.agent.likelihood_smiles(smiles[i])
                    prior_likelihood = -self.prior.likelihood_smiles(smiles[i])
                    loss = self._compute_loss(
                        prior_likelihood, agent_likelihood, scores
                    )
                    self._update(loss, verbose=False)
            else:
                # NOTE for scaffold decoration i = -1, for fragment linking i = 0
                if isinstance(self.psmiles, list):
                    i = 0
                else:
                    i = -1
                agent_likelihood = -self.agent.likelihood_smiles(smiles[i])
                prior_likelihood = -self.prior.likelihood_smiles(smiles[i])
                loss = self._compute_loss(prior_likelihood, agent_likelihood, scores)
                self._update(loss, verbose=False)
        else:
            agent_likelihood = -agent_likelihood
            prior_likelihood = -self.prior.likelihood(seqs)
            loss = self._compute_loss(prior_likelihood, agent_likelihood, scores)
            self._update(loss, verbose=False)

    def _compute_loss(self, prior_likelihood, agent_likelihood, scores):
        augmented_likelihood = prior_likelihood + self.sigma * scores
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        # Update
        self.record["loss"] += list(loss.detach().cpu().numpy())
        self.record["prior_nll"] += list(-prior_likelihood.detach().cpu().numpy())
        self.record["agent_nll"] += list(-agent_likelihood.detach().cpu().numpy())
        return loss


class BestAgentReminder(ReinforcementLearning):
    _short_name = "BAR"

    def __init__(
        self,
        device,
        model,
        agent,
        scoring_function,
        save_dir,
        optimizer,
        learning_rate,
        is_molscore=True,
        psmiles=None,
        psmiles_multi=False,
        psmiles_shuffle=True,
        freeze=None,
        prior=None,
        batch_size=64,
        sigma=60,
        alpha=0.5,
        update_freq=5,
        **kwargs,
    ):
        super().__init__(
            device,
            model,
            agent,
            scoring_function,
            save_dir,
            optimizer,
            learning_rate,
            is_molscore,
            psmiles,
            psmiles_multi,
            psmiles_shuffle,
            freeze=None,
        )
        if self.psmiles:
            raise NotImplementedError(
                "BestAgentReminder does not support promptsmiles yet"
            )
        # Load prior
        if prior is None:
            self.prior = self.model.load_from_file(
                file_path=agent, sampling_mode=True, device=device
            )
        else:
            self.prior = self.model.load_from_file(
                file_path=prior, sampling_mode=True, device=device
            )
        # Initialize best agent
        self.best_agent = copy.deepcopy(self.agent)
        self.best_avg_score = 0
        # Parameters
        self.batch_size = batch_size
        self.sigma = sigma
        self.alpha = 0.5
        self.update_freq = update_freq
        # Record
        self.record = {"loss": [], "prior_nll": [], "agent_nll": []}

    def _sample_best_batch(self, batch_size):
        self.best_agent.network.eval()
        with torch.no_grad():
            (
                best_seqs,
                best_smiles,
                best_agent_likelihood,
                best_probs,
                best_log_probs,
                _,
            ) = self.best_agent.sample_sequences_and_smiles(batch_size // 2)

        return (
            best_seqs,
            best_smiles,
            best_agent_likelihood,
            best_probs,
            best_log_probs,
            _,
        )

    def _score(self, smiles, step, additional_formats={}):
        try:
            scores = self.scoring_function(
                smiles, additional_formats=additional_formats
            )
            best_scores = utils.to_tensor(scores[self.batch_size // 2 :]).to(
                self.device
            )
            scores = utils.to_tensor(scores[: self.batch_size // 2]).to(self.device)
        except (Exception, BaseException, SystemExit, KeyboardInterrupt) as e:
            if self.molscore:
                # If anything fails, save smiles, agent, scoring_function etc.
                utils.save_smiles(
                    smiles,
                    os.path.join(
                        self.save_dir, f"failed_{self.scoring_function.step}.smi"
                    ),
                )
                self.agent.save(os.path.join(self.save_dir, f"Agent_{step}.ckpt"))
                self.scoring_function.write_scores()
                self.scoring_function._write_temp_state(step=self.scoring_function.step)
                self.scoring_function.kill_monitor()
                raise e
            else:
                utils.save_smiles(
                    smiles, os.path.join(self.save_dir, f"failed_{step + 1}.smi")
                )
                self.agent.save(os.path.join(self.save_dir, f"Agent_{step}.ckpt"))
                raise e
        return scores, best_scores

    def _train_step(self, step):
        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = (
            self._sample_batch(self.batch_size)
        )
        best_seqs, best_smiles, best_agent_likelihood, best_probs, best_log_probs, _ = (
            self._sample_best_batch(self.batch_size)
        )
        # Score
        scores, best_scores = self._score(smiles + best_smiles, step)
        # Compute normal loss
        agent_likelihood = -agent_likelihood
        prior_likelihood = -self.prior.likelihood(seqs)
        augmented_likelihood = prior_likelihood + self.sigma * scores
        agent_loss = (1 - self.alpha) * torch.pow(
            (augmented_likelihood - agent_likelihood), 2
        ).mean()

        # Compute best agent loss
        best_agent_likelihood = -best_agent_likelihood
        current_agent_likelihood = -self.agent.likelihood(best_seqs)
        best_augmented_likelihood = best_agent_likelihood + self.sigma * best_scores
        best_agent_loss = (
            self.alpha
            * torch.pow(
                (best_augmented_likelihood - current_agent_likelihood), 2
            ).mean()
        )

        loss = agent_loss + best_agent_loss

        # Update
        # loss_record += list(torch.cat([agent_loss, best_agent_loss]).detach().cpu().numpy())
        # prior_record += list(torch.cat([-prior_likelihood, -current_agent_likelihood]).detach().cpu().numpy())
        # agent_record += list(torch.cat([-agent_likelihood, -best_agent_likelihood]).detach().cpu().numpy())
        self._update(loss, verbose=False)

        if step % self.update_freq == 0:
            if scores.mean().detach().cpu().numpy() > self.best_avg_score:
                self.best_avg_score = scores.mean().detach().cpu().numpy()
                self.best_agent = copy.deepcopy(self.agent)


class HillClimb(ReinforcementLearning):
    _short_name = "HC"

    def __init__(
        self,
        device,
        model,
        agent,
        scoring_function,
        save_dir,
        optimizer,
        learning_rate,
        is_molscore=True,
        psmiles=None,
        psmiles_multi=False,
        psmiles_shuffle=True,
        psmiles_optimize=False,
        psmiles_lr_decay=1,
        psmiles_lr_epochs=10,
        freeze=None,
        batch_size=64,
        topk=0.5,
        epochs_per_step=2,
        epochs_batch_size=256,
        **kwargs,
    ):
        super().__init__(
            device=device,
            model=model,
            agent=agent,
            scoring_function=scoring_function,
            save_dir=save_dir,
            optimizer=optimizer,
            learning_rate=learning_rate,
            is_molscore=is_molscore,
            psmiles=psmiles,
            psmiles_multi=psmiles_multi,
            psmiles_shuffle=psmiles_shuffle,
            psmiles_optimize=psmiles_optimize,
            psmiles_lr_decay=psmiles_lr_decay,
            psmiles_lr_epochs=psmiles_lr_epochs,
            freeze=None,
        )

        # Parameters
        self.batch_size = batch_size
        self.topk = topk
        self.epochs_per_step = epochs_per_step
        self.epochs_batch_size = epochs_batch_size
        # Record
        self.record = {"loss": [], "agent_nll": []}

    def _train_step(self, step):
        # Sample
        self.agent.network.eval()
        with torch.no_grad():
            seqs, smiles, agent_likelihood, probs, log_probs, _ = self._sample_batch(
                self.batch_size
            )
        # Filter un-tokenized smiles
        if self.psmiles:
            smiles = self._filter_smiles(smiles)
        # Score
        scores = self._score(smiles, step)
        # Record loss
        self.record["loss"] += list(agent_likelihood.detach().cpu().numpy())
        self.record["agent_nll"] += list(agent_likelihood.detach().cpu().numpy())
        # Rank
        sscore, sscore_idxs = scores.sort(descending=True)
        agenthc_seqs = seqs[sscore_idxs.data[: int(self.batch_size * self.topk)]]
        # Now switch back to training mode
        self.agent.network.train()
        # Repeat n epochs
        for e in range(0, self.epochs_per_step):
            # Shuffle the data each epoch
            shuffle_idx = torch.randperm(agenthc_seqs.shape[0])
            agenthc_seqs = agenthc_seqs[shuffle_idx]
            # Update in appropriate batch sizes
            for es in range(0, agenthc_seqs.shape[0], self.epochs_batch_size):
                if self.psmiles:
                    # NOTE seq, probs, log_probs, critic_values is None
                    # NOTE we recompute gradients here as sampling is without graph
                    if self.psmiles_multi:
                        for i in range(len(smiles)):
                            agenthc_smiles = [smiles[i][idx] for idx in shuffle_idx]
                            log_p = -self.agent.likelihood_smiles(agenthc_smiles)
                            self._update(log_p, verbose=False)
                    else:
                        # NOTE for scaffold decoration i = -1, for fragment linking i = 0
                        if isinstance(self.psmiles, list):
                            i = 0
                        else:
                            i = -1
                        agenthc_smiles = [smiles[i][idx] for idx in shuffle_idx]
                        log_p = -self.agent.likelihood_smiles(agenthc_smiles)
                        self._update(log_p, verbose=False)
                else:
                    log_p = self.agent.likelihood(
                        agenthc_seqs[es : es + self.epochs_batch_size]
                    )
                    self._update(log_p, verbose=False)


class HillClimbRegularized(ReinforcementLearning):
    _short_name = "HC-reg"

    def __init__(
        self,
        device,
        model,
        agent,
        scoring_function,
        save_dir,
        optimizer,
        learning_rate,
        is_molscore=True,
        psmiles=None,
        psmiles_multi=False,
        psmiles_shuffle=True,
        freeze=None,
        prior=None,
        batch_size=64,
        topk=0.5,
        epochs_per_step=2,
        epochs_batch_size=256,
        entropy_coefficient=0,
        kl_coefficient=10,
        **kwargs,
    ):
        super().__init__(
            device,
            model,
            agent,
            scoring_function,
            save_dir,
            optimizer,
            learning_rate,
            is_molscore,
            psmiles,
            psmiles_multi,
            psmiles_shuffle,
            freeze=None,
        )
        if self.psmiles:
            raise NotImplementedError(
                "HillClimbRegularized does not support promptsmiles yet"
            )
        # Load prior
        if prior is None:
            self.prior = self.model.load_from_file(
                file_path=agent, sampling_mode=True, device=device
            )
        else:
            self.prior = self.model.load_from_file(
                file_path=prior, sampling_mode=True, device=device
            )
        # Parameters
        self.batch_size = batch_size
        self.topk = topk
        self.epochs_per_step = epochs_per_step
        self.epochs_batch_size = epochs_batch_size
        self.entropy_co = entropy_coefficient
        self.kl_co = kl_coefficient
        # Record
        self.record = {"loss": [], "agent_nll": []}

    def _train_step(self, step):
        # Sample
        self.agent.network.eval()
        with torch.no_grad():
            seqs, smiles, agent_likelihood, probs, log_probs, _ = self._sample_batch(
                self.batch_size
            )
        # Score
        scores = self._score(smiles, step)
        # Record loss
        dummy_loss = (
            agent_likelihood
            + (self.entropy_co * self.agent.entropy(seqs))
            + (self.kl_co * self.agent.kl(seqs, self.prior))
        )
        self.record["loss"] += list(dummy_loss.detach().cpu().numpy())
        self.record["agent_nll"] += list(agent_likelihood.detach().cpu().numpy())
        # Rank
        sscore, sscore_idxs = scores.sort(descending=True)
        agenthc_seqs = seqs[sscore_idxs.data[: int(self.batch_size * self.topk)]]
        # Now switch back to training mode
        self.agent.network.train()
        # Repeat n epochs
        for e in range(0, self.epochs_per_step):
            # Shuffle the data each epoch
            shuffle_idx = torch.randperm(agenthc_seqs.shape[0])
            agenthc_seqs = agenthc_seqs[shuffle_idx]
            # Update in appropriate batch sizes
            for es in range(0, agenthc_seqs.shape[0], self.epochs_batch_size):
                log_p = self.agent.likelihood(
                    agenthc_seqs[es : es + self.epochs_batch_size]
                )
                # Regularize
                loss = (
                    log_p
                    + (
                        self.entropy_co
                        * self.agent.entropy(
                            agenthc_seqs[es : es + self.epochs_batch_size]
                        )
                    )
                    + (
                        self.kl_co
                        * self.agent.kl(
                            agenthc_seqs[es : es + self.epochs_batch_size], self.prior
                        )
                    )
                )
                self._update(loss, verbose=False)


class AugmentedHillClimb(ReinforcementLearning):
    _short_name = "AHC"

    def __init__(
        self,
        device,
        model,
        agent,
        scoring_function,
        save_dir,
        optimizer,
        learning_rate,
        is_molscore=True,
        psmiles=None,
        psmiles_multi=False,
        psmiles_shuffle=True,
        psmiles_optimize=False,
        psmiles_lr_decay=1,
        psmiles_lr_epochs=10,
        freeze=None,
        prior=None,
        batch_size=64,
        sigma=60,
        topk=0.5,
        **kwargs,
    ):
        super().__init__(
            device=device,
            model=model,
            agent=agent,
            scoring_function=scoring_function,
            save_dir=save_dir,
            optimizer=optimizer,
            learning_rate=learning_rate,
            is_molscore=is_molscore,
            psmiles=psmiles,
            psmiles_multi=psmiles_multi,
            psmiles_shuffle=psmiles_shuffle,
            psmiles_optimize=psmiles_optimize,
            psmiles_lr_decay=psmiles_lr_decay,
            psmiles_lr_epochs=psmiles_lr_epochs,
            freeze=None,
        )

        # Load prior
        if prior is None:
            self.prior = self.model.load_from_file(
                file_path=agent, sampling_mode=True, device=device
            )
        else:
            self.prior = self.model.load_from_file(
                file_path=prior, sampling_mode=True, device=device
            )
        # Parameters
        self.batch_size = batch_size
        self.sigma = sigma
        self.topk = topk
        self.psmiles = psmiles
        self.psmiles_shuffle = psmiles_shuffle
        self.psmiles_multi = psmiles_multi
        # Record
        self.record = {"loss": [], "prior_nll": [], "agent_nll": []}

    def _train_step(self, step):
        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = (
            self._sample_batch(
                self.batch_size,
                psmiles=self.psmiles,
                psmiles_shuffle=self.psmiles_shuffle,
            )
        )
        # Filter un-tokenized smiles
        if self.psmiles:
            smiles = self._filter_smiles(smiles)
        # Score
        scores = self._score(smiles, step)
        # Compute loss
        if self.psmiles:
            # NOTE seq, probs, log_probs, critic_values is None
            # NOTE we recompute gradients here as sampling is without graph
            if self.psmiles_multi:
                for i in range(len(smiles)):
                    agent_likelihood = -self.agent.likelihood_smiles(smiles[i])
                    prior_likelihood = -self.prior.likelihood_smiles(smiles[i])
                    loss = self._compute_loss(
                        prior_likelihood, agent_likelihood, scores
                    )
                    self._update(loss, verbose=False)
            else:
                # NOTE for scaffold decoration i = -1, for fragment linking i = 0
                if isinstance(self.psmiles, list):
                    i = 0
                else:
                    i = -1
                agent_likelihood = -self.agent.likelihood_smiles(smiles[i])
                prior_likelihood = -self.prior.likelihood_smiles(smiles[i])
                loss = self._compute_loss(prior_likelihood, agent_likelihood, scores)
                self._update(loss, verbose=False)
        else:
            agent_likelihood = -agent_likelihood
            prior_likelihood = -self.prior.likelihood(seqs)
            loss = self._compute_loss(prior_likelihood, agent_likelihood, scores)
            self._update(loss, verbose=False)

    def _compute_loss(self, prior_likelihood, agent_likelihood, scores):
        augmented_likelihood = prior_likelihood + self.sigma * scores
        sscore, sscore_idxs = scores.sort(descending=True)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        # Update
        self.record["loss"] += list(loss.detach().cpu().numpy())
        self.record["prior_nll"] += list(-prior_likelihood.detach().cpu().numpy())
        self.record["agent_nll"] += list(-agent_likelihood.detach().cpu().numpy())
        # AHC
        loss = loss[sscore_idxs.data[: int(self.batch_size * self.topk)]]
        return loss
