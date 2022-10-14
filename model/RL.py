import os
import copy
from tqdm.auto import tqdm
from rdkit import rdBase
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from model.rnn import Model as RNNModel
from model.transformer import Model as TransformerModel
from model.GTr import Model as StableTransformerModel
from model import utils


class ReinforcementLearning:
    def __init__(self,
                 device,
                 model,
                 agent,
                 scoring_function,
                 save_dir,
                 optimizer,
                 learning_rate,
                 is_molscore=True,
                 freeze=None):
        # Device
        self.device = device
        if model == 'RNN':
            self.model = RNNModel
        elif model == 'Transformer':
            self.model = TransformerModel
        else: # GTr
            self.model = StableTransformerModel
        # Load agent
        self.agent = self.model.load_from_file(file_path=agent, sampling_mode=False, device=device)
        # Scoring function
        self.scoring_function = scoring_function
        self.molscore = is_molscore
        self.save_dir = save_dir
        # Optimizer
        self.optimizer = optimizer(self.agent.network.parameters(), lr=learning_rate)
        if freeze is not None:
            self._freeze_network(freeze)
        self.record = None
        # Secret smiles prefix
        self._smiles_prefix = None

    def train(self, n_steps, save_freq):
        for step in tqdm(range(n_steps), total=n_steps):
            self._train_step(step=step)
            # Save the agent weights every few iterations
            if step % save_freq == 0 and step != 0:
                self.agent.save(os.path.join(self.save_dir, f'Agent_{step}.ckpt'))
        # If the entire training finishes, clean up
        self.agent.save(os.path.join(self.save_dir, f'Agent_{n_steps}.ckpt'))
        return self.record

    def _freeze_network(self, freeze):
        n_freeze = freeze * 4 + 1
        for i, param in enumerate(self.agent.network.parameters()):
            if i < n_freeze:  # Freeze parameter
                param.requires_grad = False

    def _update(self, loss, verbose=False):
        loss = loss.mean()
        if verbose:
            print(f'    Loss: {loss.data:.03f}')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _train_step(self, step):
        raise NotImplementedError

    def _sample_batch(self, batch_size):
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = self.agent.sample_sequences_and_smiles(
            batch_size)
        if self._smiles_prefix is not None:
            smiles = [self._smiles_prefix + smi for smi in smiles]
        return seqs, smiles, agent_likelihood, probs, log_probs, critic_values

    def _score(self, smiles, step):
        try:
            scores = self.scoring_function(smiles)
            scores = utils.to_tensor(scores).to(self.device)
        except (Exception, BaseException, SystemExit, KeyboardInterrupt) as e:
            if self.molscore:
                # If anything fails, save smiles, agent, scoring_function etc.
                utils.save_smiles(smiles,
                                  os.path.join(self.save_dir,
                                               f'failed_{self.scoring_function.step}.smi'))
                self.agent.save(os.path.join(self.save_dir, f'Agent_{step}.ckpt'))
                self.scoring_function._write_temp_state(step=self.scoring_function.step)
                self.scoring_function.kill_monitor()
                raise e
            else:
                utils.save_smiles(smiles,
                                  os.path.join(self.save_dir,
                                               f'failed_{step + 1}.smi'))
                self.agent.save(os.path.join(self.save_dir, f'Agent_{step}.ckpt'))
                raise e
        return scores


class Reinforce(ReinforcementLearning):
    _short_name = 'RF'
    def __init__(self, device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore=True, freeze=None,
                 batch_size=64, **kwargs):
        super().__init__(device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore, freeze=None)

        # Parameters
        self.batch_size = batch_size
        # Record
        self.record = {'loss': [],
                       'agent_nll': []}

    def _train_step(self, step):
        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = self._sample_batch(self.batch_size)
        # Score
        scores = self._score(smiles, step)
        # Compute loss
        loss = agent_likelihood * scores
        # Update
        self.record['loss'] += list(loss.detach().cpu().numpy())
        self.record['agent_nll'] += list(agent_likelihood.detach().cpu().numpy())
        self._update(loss, verbose=False)


class ReinforceRegularized(ReinforcementLearning):
    _short_name = 'RF-reg'
    def __init__(self, device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore=True, freeze=None,
                 prior=None, batch_size=64, entropy_coefficient=0, kl_coefficient=10, **kwargs):
        super().__init__(device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore, freeze=None)

        # Load prior
        if prior is None:
            self.prior = self.model.load_from_file(file_path=agent, sampling_mode=True, device=device)
        else:
            self.prior = self.model.load_from_file(file_path=prior, sampling_mode=True, device=device)
        # Parameters
        self.batch_size = batch_size
        self.entropy_co = entropy_coefficient
        self.kl_co = kl_coefficient
        # Record
        self.record = {'loss': [],
                       'agent_nll': []}

    def _train_step(self, step):
        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = self._sample_batch(self.batch_size)
        # Score
        scores = self._score(smiles, step)
        # Compute loss
        loss = (agent_likelihood * scores) + (self.entropy_co * self.agent.entropy(seqs)) + \
               (self.kl_co * self.agent.kl(seqs, self.prior))
        # Update
        self.record['loss'] += list(loss.detach().cpu().numpy())
        self.record['agent_nll'] += list(agent_likelihood.detach().cpu().numpy())
        self._update(loss, verbose=False)


class Reinvent(ReinforcementLearning):
    _short_name = 'RV'
    def __init__(self, device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore=True, freeze=None,
                 prior=None, batch_size=64, sigma=60, **kwargs):
        super().__init__(device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore, freeze=None)

        # Load prior
        if prior is None:
            self.prior = self.model.load_from_file(file_path=agent, sampling_mode=True, device=device)
        else:
            self.prior = self.model.load_from_file(file_path=prior, sampling_mode=True, device=device)
        # Parameters
        self.batch_size = batch_size
        self.sigma = sigma
        # Record
        self.record = {'loss': [],
                       'prior_nll': [],
                       'agent_nll': []}

    def _train_step(self, step):
        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = self._sample_batch(self.batch_size)
        # Score
        scores = self._score(smiles, step)
        # Compute loss
        agent_likelihood = - agent_likelihood
        prior_likelihood = - self.prior.likelihood(seqs)
        augmented_likelihood = prior_likelihood + self.sigma * scores
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        # Update
        self.record['loss'] += list(loss.detach().cpu().numpy())
        self.record['prior_nll'] += list(-prior_likelihood.detach().cpu().numpy())
        self.record['agent_nll'] += list(-agent_likelihood.detach().cpu().numpy())
        self._update(loss, verbose=False)


class BestAgentReminder(ReinforcementLearning):
    _short_name = 'BAR'
    def __init__(self, device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore=True, freeze=None,
                 prior=None, batch_size=64, sigma=60, alpha=0.5, update_freq=5, **kwargs):
        super().__init__(device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore, freeze=None)

        # Load prior
        if prior is None:
            self.prior = self.model.load_from_file(file_path=agent, sampling_mode=True, device=device)
        else:
            self.prior = self.model.load_from_file(file_path=prior, sampling_mode=True, device=device)
        # Initialize best agent
        self.best_agent = copy.deepcopy(self.agent)
        self.best_avg_score = 0
        # Parameters
        self.batch_size = batch_size
        self.sigma = sigma
        self.alpha = 0.5
        self.update_freq = update_freq
        # Record
        self.record = {'loss': [],
                       'prior_nll': [],
                       'agent_nll': []}

    def _sample_best_batch(self, batch_size):
        self.best_agent.network.eval()
        with torch.no_grad():
            best_seqs, best_smiles, best_agent_likelihood, best_probs, best_log_probs, _ = self.best_agent.sample_sequences_and_smiles(
                batch_size // 2)

        return best_seqs, best_smiles, best_agent_likelihood, best_probs, best_log_probs, _

    def _score(self, smiles, step):
        try:
            scores = self.scoring_function(smiles)
            best_scores = utils.to_tensor(scores[self.batch_size // 2:]).to(self.device)
            scores = utils.to_tensor(scores[:self.batch_size // 2]).to(self.device)
        except (Exception, BaseException, SystemExit, KeyboardInterrupt) as e:
            if self.molscore:
                # If anything fails, save smiles, agent, scoring_function etc.
                utils.save_smiles(smiles,
                                  os.path.join(self.save_dir,
                                               f'failed_{self.scoring_function.step}.smi'))
                self.agent.save(os.path.join(self.save_dir, f'Agent_{step}.ckpt'))
                self.scoring_function.write_scores()
                self.scoring_function._write_temp_state(step=self.scoring_function.step)
                self.scoring_function.kill_monitor()
                raise e
            else:
                utils.save_smiles(smiles,
                                  os.path.join(self.save_dir,
                                               f'failed_{step + 1}.smi'))
                self.agent.save(os.path.join(self.save_dir, f'Agent_{step}.ckpt'))
                raise e
        return scores, best_scores

    def _train_step(self, step):
        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = self._sample_batch(self.batch_size)
        best_seqs, best_smiles, best_agent_likelihood, best_probs, best_log_probs, _ = self._sample_best_batch(self.batch_size)
        # Score
        scores, best_scores = self._score(smiles+best_smiles, step)
        # Compute normal loss
        agent_likelihood = - agent_likelihood
        prior_likelihood = - self.prior.likelihood(seqs)
        augmented_likelihood = prior_likelihood + self.sigma * scores
        agent_loss = (1 - self.alpha) * torch.pow((augmented_likelihood - agent_likelihood), 2).mean()

        # Compute best agent loss
        best_agent_likelihood = - best_agent_likelihood
        current_agent_likelihood = - self.agent.likelihood(best_seqs)
        best_augmented_likelihood = best_agent_likelihood + self.sigma * best_scores
        best_agent_loss = self.alpha * torch.pow((best_augmented_likelihood - current_agent_likelihood), 2).mean()

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
    _short_name = 'HC'
    def __init__(self, device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore=True, freeze=None,
                 batch_size=64, topk=0.5, epochs_per_step=2, epochs_batch_size=256, **kwargs):
        super().__init__(device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore, freeze=None)

        # Parameters
        self.batch_size = batch_size
        self.topk = topk
        self.epochs_per_step = epochs_per_step
        self.epochs_batch_size = epochs_batch_size
        # Record
        self.record = {'loss': [],
                       'agent_nll': []}

    def _train_step(self, step):
        # Sample
        self.agent.network.eval()
        with torch.no_grad():
            seqs, smiles, agent_likelihood, probs, log_probs, _ = self._sample_batch(self.batch_size)
        # Score
        scores = self._score(smiles, step)
        # Record loss
        self.record['loss'] += list(agent_likelihood.detach().cpu().numpy())
        self.record['agent_nll'] += list(agent_likelihood.detach().cpu().numpy())
        # Rank
        sscore, sscore_idxs = scores.sort(descending=True)
        agenthc_seqs = seqs[sscore_idxs.data[:int(self.batch_size * self.topk)]]
        # Now switch back to training mode
        self.agent.network.train()
        # Repeat n epochs
        for e in range(0, self.epochs_per_step):
            # Shuffle the data each epoch
            shuffle_idx = torch.randperm(agenthc_seqs.shape[0])
            agenthc_seqs = agenthc_seqs[shuffle_idx]
            # Update in appropriate batch sizes
            for es in range(0, agenthc_seqs.shape[0], self.epochs_batch_size):
                log_p = self.agent.likelihood(agenthc_seqs[es:es + self.epochs_batch_size])
                self._update(log_p, verbose=False)


class HillClimbRegularized(ReinforcementLearning):
    _short_name = 'HC-reg'
    def __init__(self, device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore=True, freeze=None,
                 prior=None, batch_size=64, topk=0.5, epochs_per_step=2, epochs_batch_size=256,
                 entropy_coefficient=0, kl_coefficient=10, **kwargs):
        super().__init__(device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore, freeze=None)

        # Load prior
        if prior is None:
            self.prior = self.model.load_from_file(file_path=agent, sampling_mode=True, device=device)
        else:
            self.prior = self.model.load_from_file(file_path=prior, sampling_mode=True, device=device)
        # Parameters
        self.batch_size = batch_size
        self.topk = topk
        self.epochs_per_step = epochs_per_step
        self.epochs_batch_size = epochs_batch_size
        self.entropy_co = entropy_coefficient
        self.kl_co = kl_coefficient
        # Record
        self.record = {'loss': [],
                       'agent_nll': []}

    def _train_step(self, step):
        # Sample
        self.agent.network.eval()
        with torch.no_grad():
            seqs, smiles, agent_likelihood, probs, log_probs, _ = self._sample_batch(self.batch_size)
        # Score
        scores = self._score(smiles, step)
        # Record loss
        dummy_loss = agent_likelihood + (self.entropy_co * self.agent.entropy(seqs)) +\
                     (self.kl_co * self.agent.kl(seqs, self.prior))
        self.record['loss'] += list(dummy_loss.detach().cpu().numpy())
        self.record['agent_nll'] += list(agent_likelihood.detach().cpu().numpy())
        # Rank
        sscore, sscore_idxs = scores.sort(descending=True)
        agenthc_seqs = seqs[sscore_idxs.data[:int(self.batch_size * self.topk)]]
        # Now switch back to training mode
        self.agent.network.train()
        # Repeat n epochs
        for e in range(0, self.epochs_per_step):
            # Shuffle the data each epoch
            shuffle_idx = torch.randperm(agenthc_seqs.shape[0])
            agenthc_seqs = agenthc_seqs[shuffle_idx]
            # Update in appropriate batch sizes
            for es in range(0, agenthc_seqs.shape[0], self.epochs_batch_size):
                log_p = self.agent.likelihood(agenthc_seqs[es:es + self.epochs_batch_size])
                # Regularize
                loss = log_p + (self.entropy_co * self.agent.entropy(agenthc_seqs[es:es + self.epochs_batch_size])) +\
                       (self.kl_co * self.agent.kl(agenthc_seqs[es:es + self.epochs_batch_size], self.prior))
                self._update(loss, verbose=False)


class AugmentedHillClimb(ReinforcementLearning):
    _short_name = 'AHC'
    def __init__(self, device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore=True, freeze=None,
                 prior=None, batch_size=64, sigma=60, topk=0.5, **kwargs):
        super().__init__(device, model, agent, scoring_function, save_dir, optimizer, learning_rate, is_molscore, freeze=None)

        # Load prior
        if prior is None:
            self.prior = self.model.load_from_file(file_path=agent, sampling_mode=True, device=device)
        else:
            self.prior = self.model.load_from_file(file_path=prior, sampling_mode=True, device=device)
        # Parameters
        self.batch_size = batch_size
        self.sigma = sigma
        self.topk = topk
        # Record
        self.record = {'loss': [],
                       'prior_nll': [],
                       'agent_nll': []}

    def _train_step(self, step):
        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = self._sample_batch(self.batch_size)
        # Score
        scores = self._score(smiles, step)
        # Compute loss
        agent_likelihood = - agent_likelihood
        prior_likelihood = - self.prior.likelihood(seqs)
        augmented_likelihood = prior_likelihood + self.sigma * scores
        sscore, sscore_idxs = scores.sort(descending=True)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        # Update
        self.record['loss'] += list(loss.detach().cpu().numpy())
        self.record['prior_nll'] += list(-prior_likelihood.detach().cpu().numpy())
        self.record['agent_nll'] += list(-agent_likelihood.detach().cpu().numpy())
        loss = loss[sscore_idxs.data[:int(self.batch_size * self.topk)]]
        self._update(loss, verbose=False)



