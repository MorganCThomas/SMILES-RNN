import os
import argparse
import logging
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
    ms = MolScore(model_name='reinvent', task_config=args.molscore_config)
    ms.log_parameters({'optimization': args.rl_mode, 'batch_size': args.batch_size, 'sigma': args.sigma,
                       'prior': os.path.basename(args.prior), 'init_agent': os.path.basename(args.agent)})

    # Load model
    logger.info(f'Loading models')
    prior = Model.load_from_file(file_path=args.prior, sampling_mode=True, device=device)
    agent = Model.load_from_file(file_path=args.agent, sampling_mode=False, device=device)

    # Freeze layers (embedding + 4 parameters per RNN layer)
    if args.freeze is not None:
        n_freeze = args.freeze * 4 + 1
        for i, param in enumerate(agent.network.parameters()):
            if i < n_freeze:  # Freeze parameter
                param.requires_grad = False

    # Setup optimizer
    optimizer = torch.optim.Adam(agent.network.parameters(), lr=0.0005)

    # Start training
    for step in tqdm(range(args.n_steps), total=args.n_steps):

        seqs, smiles, agent_likelihood = agent.sample_sequences_and_smiles(args.batch_size)
        agent_likelihood = -agent_likelihood
        prior_likelihood = -prior.likelihood(seqs)
        try:
            scores = ms(smiles)
        except:
            utils.save_smiles(smiles, os.path.join(ms.save_dir, f'failed_{ms.step}.smi'))
            agent.save(os.path.join(ms.save_dir, f'Agent_{step}.ckpt'))
            ms.write_scores()
            ms.kill_dash_monitor()
            raise

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
            loss = - agenthc_likelihood.mean()

        # Update
        loss = loss.mean()
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
    optional.add_argument('-s', '--sigma', type=int, default=60, help='Scaling coefficient of score')
    optional.add_argument('-rl', '--rl_mode', type=str, default='reinvent',
                          choices=['reinvent', 'augHC', 'HC'],
                          help='Which reinforcement learning algorithm to use')
    optional.add_argument('--save_freq', type=int, default=100, help='How often to save models')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
