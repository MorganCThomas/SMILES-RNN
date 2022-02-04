import os
import argparse
import logging
import json
import copy
from tqdm.auto import tqdm
from rdkit import rdBase
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from model.model import Model
from model import utils

from molscore.manager import MolScore

rdBase.DisableLog("rdApp.error")

logger = logging.getLogger('reinforcement_learning')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def train_step(optimizer, loss, verbose=False):
    loss = loss.mean()
    if verbose:
        print(f'    Loss: {loss.data:.03f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main(args):
    # Set device
    device = utils.set_default_device_cuda(args.device)
    logger.info(f'Device set to {device.type}')

    # Scoring_function
    ms = MolScore(model_name='SMILES-RNN', task_config=args.molscore_config)
    ms.log_parameters({k: vars(args)[k] for k in
                       ['prior', 'agent', 'batch_size', 'rl_mode', 'sigma', 'kl_coefficient', 'entropy_coefficient']
                       if k in vars(args).keys()})

    # Also save these parameters for good measure
    with open(os.path.join(ms.save_dir, 'SMILES-RNN.params'), 'wt') as f:
        json.dump(vars(args), f)

    # Load prior
    logger.info(f'Loading models')
    if args.rl_mode not in ['HC', 'PG', 'A2C']:  # Don't need the prior in some cases
        prior = Model.load_from_file(file_path=args.prior, sampling_mode=True, device=device)

    # Load agent
    agent = Model.load_from_file(file_path=args.agent, sampling_mode=False, device=device)
    if args.rl_mode in ['A2C', 'A2C-reg', 'PPO', 'PPO-reg']:
        agent.RNN2Critic()
    if args.rl_mode == 'BAR':
        best_agent = copy.deepcopy(agent)
        best_avg_score = 0

    # Freeze layers (embedding + 4 parameters per RNN layer)
    if args.freeze is not None:
        n_freeze = args.freeze * 4 + 1
        for i, param in enumerate(agent.network.parameters()):
            if i < n_freeze:  # Freeze parameter
                param.requires_grad = False

    # Setup optimizer and learning rate
    optimizer = torch.optim.Adam(agent.network.parameters(), lr=args.learning_rate)

    # Start training
    loss_record = []
    prior_record = []
    agent_record = []

    for step in tqdm(range(args.n_steps), total=args.n_steps):

        # Sample
        if args.rl_mode in ['HC', 'HC-reg']:
            agent.network.eval()  # If HC, we compute loss from smaller batches later
            with torch.no_grad():
                seqs, smiles, agent_likelihood, probs, log_probs, critic_values = agent.sample_sequences_and_smiles(
                    args.batch_size)

        elif args.rl_mode == 'BAR':
            seqs, smiles, agent_likelihood, probs, log_probs, critic_values = agent.sample_sequences_and_smiles(
                args.batch_size//2)

            best_agent.network.eval()
            with torch.no_grad():
                best_seqs, best_smiles, best_agent_likelihood, best_probs, best_log_probs, _ = best_agent.sample_sequences_and_smiles(
                    args.batch_size//2)

        else:
            seqs, smiles, agent_likelihood, probs, log_probs, critic_values = agent.sample_sequences_and_smiles(
                args.batch_size)

        # Score
        try:
            if args.rl_mode == 'BAR':
                scores = ms(smiles+best_smiles)
                best_scores = utils.to_tensor(scores[args.batch_size//2:]).to(device)
                scores = utils.to_tensor(scores[:args.batch_size//2]).to(device)

            else:
                scores = ms(smiles)
                scores = utils.to_tensor(scores).to(device)
        except:
            utils.save_smiles(smiles, os.path.join(ms.save_dir, f'failed_{ms.step}.smi'))
            agent.save(os.path.join(ms.save_dir, f'Agent_{step}.ckpt'))
            ms.write_scores()
            ms._write_temp_state(step=ms.step)
            ms.kill_monitor()
            raise

        # Compute loss
        if (args.rl_mode == 'reinvent') or (args.rl_mode == 'reinvent2'):
            agent_likelihood = - agent_likelihood
            prior_likelihood = - prior.likelihood(seqs)
            augmented_likelihood = prior_likelihood + args.sigma * scores
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            # Update
            loss_record += list(loss.detach().cpu().numpy())
            prior_record += list(-prior_likelihood.detach().cpu().numpy())
            agent_record += list(-agent_likelihood.detach().cpu().numpy())
            train_step(optimizer, loss, verbose=args.verbose)

        if args.rl_mode == 'BAR':
            # Agent sample
            agent_likelihood = - agent_likelihood
            prior_likelihood = - prior.likelihood(seqs)
            augmented_likelihood = prior_likelihood + args.sigma * scores
            agent_loss = (1 - args.alpha) * torch.pow((augmented_likelihood - agent_likelihood), 2).mean()

            # Best agent sample
            best_agent_likelihood = - best_agent_likelihood
            current_agent_likelihood = - agent.likelihood(best_seqs)
            best_augmented_likelihood = best_agent_likelihood + args.sigma * best_scores
            best_agent_loss = args.alpha * torch.pow((best_augmented_likelihood - current_agent_likelihood), 2).mean()

            loss = agent_loss + best_agent_loss

            # Update
            #loss_record += list(torch.cat([agent_loss, best_agent_loss]).detach().cpu().numpy())
            #prior_record += list(torch.cat([-prior_likelihood, -current_agent_likelihood]).detach().cpu().numpy())
            #agent_record += list(torch.cat([-agent_likelihood, -best_agent_likelihood]).detach().cpu().numpy())
            train_step(optimizer, loss, verbose=args.verbose)

            if step % args.update_freq == 0:
                if scores.mean().detach().cpu().numpy() > best_avg_score:
                    best_avg_score = scores.mean().detach().cpu().numpy()
                    best_agent = copy.deepcopy(agent)


        if args.rl_mode == 'augHC':
            agent_likelihood = - agent_likelihood
            prior_likelihood = - prior.likelihood(seqs)
            augmented_likelihood = prior_likelihood + args.sigma * scores
            sscore, sscore_idxs = scores.sort(descending=True)
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            # Update
            loss_record += list(loss.detach().cpu().numpy())
            prior_record += list(-prior_likelihood.detach().cpu().numpy())
            agent_record += list(-agent_likelihood.detach().cpu().numpy())
            loss = loss[sscore_idxs.data[:int(args.batch_size * args.topk)]]
            train_step(optimizer, loss, verbose=args.verbose)

        if args.rl_mode == 'augSH':
            agent_likelihood = - agent_likelihood
            prior_likelihood = - prior.likelihood(seqs)
            augmented_likelihood = prior_likelihood + args.sigma * (scores + 1)
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            # Update
            loss_record += list(loss.detach().cpu().numpy())
            prior_record += list(-prior_likelihood.detach().cpu().numpy())
            agent_record += list(-agent_likelihood.detach().cpu().numpy())
            train_step(optimizer, loss, verbose=args.verbose)

        if args.rl_mode == 'HC':
            loss_record += list(agent_likelihood.detach().cpu().numpy())
            agent_record += list(agent_likelihood.detach().cpu().numpy())
            sscore, sscore_idxs = scores.sort(descending=True)
            agenthc_seqs = seqs[sscore_idxs.data[:int(args.batch_size * args.topk)]]
            # Now switch back to training mode
            agent.network.train()
            # Repeat n epochs
            for e in range(0, args.epochs_per_step):
                # Shuffle the data each epoch
                shuffle_idx = torch.randperm(agenthc_seqs.shape[0])
                agenthc_seqs = agenthc_seqs[shuffle_idx]
                # Update in appropriate batch sizes
                for es in range(0, agenthc_seqs.shape[0], args.epochs_batch_size):
                    log_p = agent.likelihood(agenthc_seqs[es:es+args.epochs_batch_size])
                    train_step(optimizer, log_p, verbose=args.verbose)

        if args.rl_mode == 'HC-reg':
            dummy_loss = agent_likelihood + (args.kl_coefficient*agent.kl(seqs, prior))
            loss_record += list(dummy_loss.detach().cpu().numpy())
            prior_record += list(prior.likelihood(seqs).detach().cpu().numpy())
            agent_record += list(agent_likelihood.detach().cpu().numpy())
            sscore, sscore_idxs = scores.sort(descending=True)
            agenthc_seqs = seqs[sscore_idxs.data[:int(args.batch_size * args.topk)]]
            # Now switch back to training mode
            agent.network.train()
            # Repeat n epochs
            for e in range(0, args.epochs_per_step):
                # Shuffle the data each epoch
                shuffle_idx = torch.randperm(agenthc_seqs.shape[0])
                agenthc_seqs = agenthc_seqs[shuffle_idx]
                # Update in appropriate batch sizes
                for es in range(0, agenthc_seqs.shape[0], args.epochs_batch_size):
                    log_p = agent.likelihood(agenthc_seqs[es:es + args.epochs_batch_size])
                    # Regularize by penalizing kl divergence
                    loss = log_p + (args.kl_coefficient*agent.kl(agenthc_seqs[es:es + args.epochs_batch_size], prior))
                    train_step(optimizer, loss, verbose=args.verbose)

        if args.rl_mode == 'PG':
            loss = agent_likelihood * scores
            # Update
            loss_record += list(loss.detach().cpu().numpy())
            agent_record += list(agent_likelihood.detach().cpu().numpy())
            train_step(optimizer, loss, verbose=args.verbose)

        if args.rl_mode == 'PG-reg':
            loss = (agent_likelihood*scores) + (args.entropy_coefficient*agent.entropy(seqs)) + \
                   (args.kl_coefficient*agent.kl(seqs, prior))
            # Update
            loss_record += list(loss.detach().cpu().numpy())
            prior_record += list(prior.likelihood(seqs).detach().cpu().numpy())
            agent_record += list(agent_likelihood.detach().cpu().numpy())
            train_step(optimizer, loss, verbose=args.verbose)

        if args.rl_mode == 'A2C':
            raise NotImplementedError
            advantage = scores - critic_values.sum(dim=1)  # Let's sum over all sequence steps
            loss = (agent_likelihood*advantage) + torch.pow(advantage, 2)
            # Update
            loss_record += list(loss.detach().cpu().numpy())
            agent_record += list(agent_likelihood.detach().cpu().numpy())
            train_step(optimizer, loss, verbose=args.verbose)

        if args.rl_mode == 'A2C-reg':
            raise NotImplementedError
            advantage = scores - critic_values.sum(dim=1)  # Let's sum over all sequence steps
            loss = (agent_likelihood * advantage) + torch.pow(advantage, 2) +\
                   (args.entropy_coefficient*agent.entropy(seqs)) +\
                   (args.kl_coefficient*agent.kl(seqs, prior))
            # Update
            loss_record += list(loss.detach().cpu().numpy())
            train_step(optimizer, loss, verbose=args.verbose)

        if args.rl_mode == 'PPO':
            raise NotImplementedError
            # TODO as ppo_epochs, episode_size, episode_epochs, batch_size
            old_probs, old_log_probs, _ = prior.probabilities(seqs)
            rewards = torch.zeros(critic_values.shape)
            for i in range(args.batch_size):
                rewards[i, :] = utils.to_tensor(scores)[i]
            advantage = rewards - critic_values
            ratio = (probs / old_probs)
            policy = torch.zeros(probs.shape)
            for i, (rat, adv) in enumerate(zip(ratio, advantage)):
                policy[i, :] = -torch.min((rat * adv), torch.clip(rat, 1-0.2, 1+0.2) * adv)
            loss = policy.mean(dim=1) + (args.value_coefficient * agent.value_loss(seqs, advantage))
            # Update
            loss_record += list(loss.detach().cpu().numpy())
            train_step(optimizer, loss, verbose=args.verbose)

        if args.rl_mode == 'PPO-reg':
            raise NotImplementedError
            old_probs, old_log_probs, _ = prior.probabilities(seqs)
            rewards = torch.zeros(critic_values.shape)
            for i in range(args.batch_size):
                rewards[i, :] = utils.to_tensor(scores)[i]
            advantage = rewards - critic_values
            ratio = (probs / old_probs)
            policy = torch.zeros(probs.shape)
            for i, (rat, adv) in enumerate(zip(ratio, advantage)):
                policy[i, :] = -torch.min((rat * adv), torch.clip(rat, 1 - 0.2, 1 + 0.2) * adv)
            loss = policy.mean(dim=1) + (args.value_coefficient * agent.value_loss(seqs, advantage)) -\
                   (args.entropy_coefficient * agent.entropy(seqs)) + (args.kl_coefficient * agent.kl(seqs, prior))
            # Update
            loss_record += list(loss.detach().cpu().numpy())
            train_step(optimizer, loss, verbose=args.verbose)

        # Save the agent weights every few iterations
        if step % args.save_freq == 0 and step != 0:
            agent.save(os.path.join(ms.save_dir, f'Agent_{step}.ckpt'))

    # If the entire training finishes, clean up
    agent.save(os.path.join(ms.save_dir, f'Agent_{args.n_steps}.ckpt'))
    ms.log_parameters({'loss': loss_record,
                       'prior_likelihood': prior_record,
                       'agent_likelihood': agent_record})
    ms.write_scores()
    ms.kill_monitor()
    return


def get_args():
    parser = argparse.ArgumentParser(description='Optimize an RNN towards a reward via reinforment learning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = parser.add_argument_group('Required arguments')
    required.add_argument('-p', '--prior', type=str, help='Path to prior checkpoint (.ckpt)', required=True)
    required.add_argument('-m', '--molscore_config', type=str, help='Path to molscore config (.json)', required=True)

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('-a', '--agent', type=str, help='Path to agent checkpoint (.ckpt)')
    optional.add_argument('-d', '--device', default='gpu', help=' ')
    optional.add_argument('-f', '--freeze', help='Number of RNN layers to freeze', type=int)
    optional.add_argument('--save_freq', type=int, default=100, help='How often to save models')
    optional.add_argument('--verbose', action='store_true', help='Whether to print loss')

    subparsers = parser.add_subparsers(title='Optimization algorithms', dest='rl_mode',
                                       help='Which reinforcement learning algorithm to use')

    reinvent_parser = subparsers.add_parser('reinvent', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    reinvent_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    reinvent_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    reinvent_parser.add_argument('-s', '--sigma', type=int, default=60, help='Scaling coefficient of score')
    reinvent_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    reinvent2_parser = subparsers.add_parser('reinvent2', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    reinvent2_parser.add_argument('--n_steps', type=int, default=250, help=' ')
    reinvent2_parser.add_argument('--batch_size', type=int, default=128, help=' ')
    reinvent2_parser.add_argument('-s', '--sigma', type=int, default=120, help='Scaling coefficient of score')
    reinvent2_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    bar_parser = subparsers.add_parser('BAR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bar_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    bar_parser.add_argument('--batch_size', type=int, default=64, help='Batch size per agent (will be effectively doubled)')
    bar_parser.add_argument('-s', '--sigma', type=int, default=60, help='Scaling coefficient of score')
    bar_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')
    bar_parser.add_argument('-a', '--alpha', type=float, default=0.5, help='Scaling parameter of agent/best_agent',
                            metavar="[0-1]")
    bar_parser.add_argument('-uf', '--update_freq', type=int, default=5, help='Frequency of training steps to update the agent')

    augHC_parser = subparsers.add_parser('augHC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    augHC_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    augHC_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    augHC_parser.add_argument('-s', '--sigma', type=int, default=60, help='Scaling coefficient of score')
    augHC_parser.add_argument('-k', '--topk', type=float, default=0.5, help='Fraction of top molecules to keep',
                              metavar="[0-1]")
    augHC_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    augSH_parser = subparsers.add_parser('augSH', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    augSH_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    augSH_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    augSH_parser.add_argument('-s', '--sigma', type=int, default=30, help='Scaling coefficient of score')
    augSH_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    HC_parser = subparsers.add_parser('HC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    HC_parser.add_argument('--n_steps', type=int, default=30, help=' ')
    HC_parser.add_argument('--batch_size', type=int, default=1024, help=' ')
    HC_parser.add_argument('--epochs_per_step', type=int, default=2, help=' ')
    HC_parser.add_argument('--epochs_batch_size', type=int, default=256, help=' ')
    HC_parser.add_argument('-k', '--topk', type=float, default=0.5, help='Fraction of top molecules to keep',
                           metavar="[0-1]")
    HC_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    HCr_parser = subparsers.add_parser('HC-reg', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    HCr_parser.add_argument('--n_steps', type=int, default=30, help=' ')
    HCr_parser.add_argument('--batch_size', type=int, default=1024, help=' ')
    HCr_parser.add_argument('--epochs_per_step', type=int, default=2, help=' ')
    HCr_parser.add_argument('--epochs_batch_size', type=int, default=256, help=' ')
    HCr_parser.add_argument('-k', '--topk', type=float, default=0.5, help='Fraction of top molecules to keep',
                            metavar="[0-1]")
    HCr_parser.add_argument('-klc', '--kl_coefficient', type=float, default=10,
                            help='Coefficient of KL loss contribution')
    HCr_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    PG_parser = subparsers.add_parser('PG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    PG_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    PG_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    PG_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Adam learning rate')

    PGr_parser = subparsers.add_parser('PG-reg', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    PGr_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    PGr_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    PGr_parser.add_argument('-ec', '--entropy_coefficient', type=float, default=0,
                            help='Coefficient of entropy loss contribution')
    PGr_parser.add_argument('-klc', '--kl_coefficient', type=float, default=10,
                            help='Coefficient of KL loss contribution')
    PGr_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Adam learning rate')

    A2C_parser = subparsers.add_parser('A2C', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    A2C_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    A2C_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    A2C_parser.add_argument('-vlc', '--value_coefficient', type=float, default=0.5,
                            help='Coefficient of value loss contribution')
    A2C_parser.add_argument('-lr', '--learning_rate', type=float, default=4e-4, help='Adam learning rate')

    A2Cr_parser = subparsers.add_parser('A2C-reg', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    A2Cr_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    A2Cr_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    A2Cr_parser.add_argument('-vlc', '--value_coefficient', type=float, default=0.5,
                             help='Coefficient of value loss contribution')
    A2Cr_parser.add_argument('-ec', '--entropy_coefficient', type=float, default=0.013,
                             help='Coefficient of entropy loss contribution')
    A2Cr_parser.add_argument('-klc', '--kl_coefficient', type=float, default=10,
                             help='Coefficient of KL loss contribution')
    A2Cr_parser.add_argument('-lr', '--learning_rate', type=float, default=4e-4, help='Adam learning rate')

    PPO_parser = subparsers.add_parser('PPO', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    PPO_parser.add_argument('--n_steps', type=int, default=200, help=' ')
    PPO_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    PPO_parser.add_argument('-vlc', '--value_coefficient', type=float, default=0.5,
                            help='Coefficient of value loss contribution')
    PPO_parser.add_argument('-lr', '--learning_rate', type=float, default=4e-4, help='Adam learning rate')

    PPOr_parser = subparsers.add_parser('PPO-reg', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    PPOr_parser.add_argument('--n_steps', type=int, default=200, help=' ')
    PPOr_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    PPOr_parser.add_argument('-vlc', '--value_coefficient', type=float, default=0.5,
                             help='Coefficient of value loss contribution')
    PPOr_parser.add_argument('-ec', '--entropy_coefficient', type=float, default=0.013,
                             help='Coefficient of entropy loss contribution')
    PPOr_parser.add_argument('-klc', '--kl_coefficient', type=float, default=0,
                             help='Coefficient of KL loss contribution')
    PPOr_parser.add_argument('-lr', '--learning_rate', type=float, default=4e-4, help='Adam learning rate')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # Set agent as prior if not specified
    if args.agent is None:
        setattr(args, 'agent', args.prior)
    main(args)
