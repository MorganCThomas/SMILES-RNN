import os
import argparse
import logging
import json
from tqdm.auto import tqdm
from rdkit import rdBase

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

    # Load model
    logger.info(f'Loading models')
    prior = Model.load_from_file(file_path=args.prior, sampling_mode=True, device=device)
    agent = Model.load_from_file(file_path=args.agent, sampling_mode=False, device=device)
    if args.rl_mode in ['A2C', 'A2C-reg', 'PPO', 'PPO-reg']:
        agent.RNN2Critic()

    # Freeze layers (embedding + 4 parameters per RNN layer)
    if args.freeze is not None:
        n_freeze = args.freeze * 4 + 1
        for i, param in enumerate(agent.network.parameters()):
            if i < n_freeze:  # Freeze parameter
                param.requires_grad = False

    # Setup optimizer and learning rate
    optimizer = torch.optim.Adam(agent.network.parameters(), lr=args.learning_rate)

    # Start training
    for step in tqdm(range(args.n_steps), total=args.n_steps):

        # Sample
        seqs, smiles, agent_likelihood, probs, log_probs, critic_values = agent.sample_sequences_and_smiles(args.batch_size)
        agent_likelihood = - agent_likelihood
        prior_likelihood = - prior.likelihood(seqs)

        # Score
        try:
            scores = ms(smiles)
        except:
            utils.save_smiles(smiles, os.path.join(ms.save_dir, f'failed_{ms.step}.smi'))
            agent.save(os.path.join(ms.save_dir, f'Agent_{step}.ckpt'))
            ms.write_scores()
            ms.kill_dash_monitor()
            raise

        # Compute loss
        if args.rl_mode == 'PG':
            policy = torch.zeros(log_probs.shape)
            for i, (prob, reward) in enumerate(zip(log_probs, utils.to_tensor(scores))):
                policy[i, :] = prob * reward
            loss = policy.mean(dim=1)

        if args.rl_mode == 'PG-reg':
            policy = torch.zeros(log_probs.shape)
            for i, (prob, reward) in enumerate(zip(log_probs, utils.to_tensor(scores))):
                policy[i, :] = prob * reward
            loss = policy.mean(dim=1) + (args.entropy_coefficient * agent.entropy(seqs)) +\
                   (args.kl_coefficient * agent.kl(seqs, prior))

        if args.rl_mode == 'A2C':
            rewards = torch.zeros(critic_values.shape)
            for i in range(args.batch_size):
                rewards[i, :] = utils.to_tensor(scores)[i]
            advantage = rewards - critic_values
            policy = torch.zeros(log_probs.shape)
            for i, (prob, adv) in enumerate(zip(log_probs, advantage)):
                policy[i, :] = (prob * adv) + torch.pow(adv, 2)
            loss = policy.mean(dim=1) + (args.value_coefficient * agent.value_loss(seqs, advantage))

        if args.rl_mode == 'A2C-reg':
            rewards = torch.zeros(critic_values.shape)
            for i in range(args.batch_size):
                rewards[i, :] = utils.to_tensor(scores)[i]
            advantage = rewards - critic_values
            policy = torch.zeros(log_probs.shape)
            for i, (prob, adv) in enumerate(zip(log_probs, advantage)):
                policy[i, :] = (prob * adv) + torch.pow(adv, 2)
            loss = policy.mean(dim=1) + (args.value_coefficient * agent.value_loss(seqs, advantage)) -\
                   (args.entropy_coefficient * agent.entropy(seqs)) + (args.kl_coefficient * agent.kl(seqs, prior))

        if args.rl_mode == 'PPO':
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

        if args.rl_mode == 'PPO-reg':
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

        if args.rl_mode == 'reinvent':
            augmented_likelihood = prior_likelihood + args.sigma * utils.to_tensor(scores)
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        if args.rl_mode == 'augHC':
            augmented_likelihood = prior_likelihood + args.sigma * utils.to_tensor(scores)
            sscore, sscore_idxs = utils.to_tensor(scores).sort(descending=True)
            aughc_likelihood = augmented_likelihood[sscore_idxs.data[:int(args.batch_size // 2)]]
            agenthc_likelihood = agent_likelihood[sscore_idxs.data[:int(args.batch_size // 2)]]
            loss = torch.pow((aughc_likelihood - agenthc_likelihood), 2)

        if args.rl_mode == 'HC':
            sscore, sscore_idxs = utils.to_tensor(scores).sort(descending=True)
            agenthc_likelihood = agent_likelihood[sscore_idxs.data[:int(args.batch_size // 2)]]
            loss = -agenthc_likelihood

        # Update
        loss = loss.mean()
        print(f'    Loss: {loss.data:.03f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the agent weights every few iterations
        if step % args.save_freq == 0 and step != 0:
            agent.save(os.path.join(ms.save_dir, f'Agent_{step}.ckpt'))

    # If the entire training finishes, clean up
    agent.save(os.path.join(ms.save_dir, f'Agent_{args.n_steps}.ckpt'))
    ms.write_scores()
    ms.kill_dash_monitor()
    return


def get_args():
    parser = argparse.ArgumentParser(description='Optimize an RNN towards a reward via reinforment learning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = parser.add_argument_group('Required arguments')
    required.add_argument('-p', '--prior', type=str, help='Path to prior checkpoint (.ckpt)', required=True)
    required.add_argument('-a', '--agent', type=str, help='Path to agent checkpoint (.ckpt)', required=True)
    required.add_argument('-m', '--molscore_config', type=str, help='Path to molscore config (.json)', required=True)

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('--batch_size', type=int, default=64, help=' ')
    optional.add_argument('--n_steps', type=int, default=200, help=' ')
    optional.add_argument('-d', '--device', default='gpu', help=' ')
    optional.add_argument('-f', '--freeze', help='Number of RNN layers to freeze', type=int)
    optional.add_argument('--save_freq', type=int, default=100, help='How often to save models')

    subparsers = parser.add_subparsers(title='Optimization algorithms', dest='rl_mode',
                                       help='Which reinforcement learning algorithm to use')

    reinvent_parser = subparsers.add_parser('reinvent')
    reinvent_parser.add_argument('-s', '--sigma', type=int, default=60, help='Scaling coefficient of score')
    reinvent_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    augHC_parser = subparsers.add_parser('augHC')
    augHC_parser.add_argument('-s', '--sigma', type=int, default=30, help='Scaling coefficient of score')
    augHC_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    HC_parser = subparsers.add_parser('HC')
    # TODO add top k
    HC_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    PG_parser = subparsers.add_parser('PG')
    PG_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Adam learning rate')

    PGr_parser = subparsers.add_parser('PG-reg')
    PGr_parser.add_argument('-ec', '--entropy_coefficient', type=float, default=0,
                            help='Coefficient of entropy loss contribution')
    PGr_parser.add_argument('-klc', '--kl_coefficient', type=float, default=10,
                            help='Coefficient of KL loss contribution')
    PGr_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Adam learning rate')

    A2C_parser = subparsers.add_parser('A2C')
    A2C_parser.add_argument('-vlc', '--value_coefficient', type=float, default=0.5,
                            help='Coefficient of value loss contribution')
    A2C_parser.add_argument('-lr', '--learning_rate', type=float, default=4e-4, help='Adam learning rate')

    A2Cr_parser = subparsers.add_parser('A2C-reg')
    A2Cr_parser.add_argument('-vlc', '--value_coefficient', type=float, default=0.5,
                             help='Coefficient of value loss contribution')
    A2Cr_parser.add_argument('-ec', '--entropy_coefficient', type=float, default=0.013,
                             help='Coefficient of entropy loss contribution')
    A2Cr_parser.add_argument('-klc', '--kl_coefficient', type=float, default=0,
                             help='Coefficient of KL loss contribution')
    A2Cr_parser.add_argument('-lr', '--learning_rate', type=float, default=4e-4, help='Adam learning rate')

    PPO_parser = subparsers.add_parser('PPO')
    PPO_parser.add_argument('-vlc', '--value_coefficient', type=float, default=0.5,
                            help='Coefficient of value loss contribution')
    PPO_parser.add_argument('-lr', '--learning_rate', type=float, default=4e-4, help='Adam learning rate')

    PPOr_parser = subparsers.add_parser('PPO-reg')
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
    main(args)
