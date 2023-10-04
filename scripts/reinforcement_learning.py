import os
import argparse
import logging
import json
from rdkit import rdBase
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from model import RL_strategies, utils

from molscore.manager import MolScore

rdBase.DisableLog("rdApp.error")

logger = logging.getLogger('reinforcement_learning')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def main(args):
    # Setup scoring function
    ms = MolScore(model_name='SMILES-RNN', task_config=args.molscore_config)
    ms.log_parameters({k: vars(args)[k] for k in
                       ['model', 'prior', 'agent', 'batch_size', 'rl_strategy', 'sigma', 'kl_coefficient', 'entropy_coefficient']
                       if k in vars(args).keys()})

    # Save these parameters for good measure
    with open(os.path.join(ms.save_dir, 'SMILES-RNN.params'), 'wt') as f:
        json.dump(vars(args), f)

    # Setup device
    args.device = utils.set_default_device_cuda(args.device)
    logger.info(f'Device set to {args.device.type}')

    # Setup RL
    assert any([args.rl_strategy == s._short_name for s in RL_strategies]), f"{args.rl_strategy} not found"
    for strategy in RL_strategies:
        if args.rl_strategy == strategy._short_name:
            RL = strategy
    RL = RL(scoring_function=ms, save_dir=ms.save_dir, optimizer=torch.optim.Adam,
            **vars(args))
    # Cheap fixed SMILES prefix
    if args.smiles_prefix is not None:
        RL._smiles_prefix = args.smiles_prefix

    # Start training
    record = RL.train(n_steps=args.n_steps, save_freq=args.save_freq)

    # Wrap up MolScore
    if record is not None: ms.log_parameters(record)
    ms.write_scores()
    ms.kill_monitor()


def get_args():
    parser = argparse.ArgumentParser(description='Optimize a model towards a reward via reinforment learning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = parser.add_argument_group('Required arguments')
    required.add_argument('-p', '--prior', type=str, help='Path to prior checkpoint (.ckpt)', required=True)
    required.add_argument('-m', '--molscore_config', type=str, help='Path to molscore config (.json)', required=True)
    required.add_argument('--model', type=str, choices=['RNN', 'Transformer', 'GTr'], help='Choice of architecture', required=True)

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('-a', '--agent', type=str, help='Path to agent checkpoint (.ckpt)')
    optional.add_argument('-d', '--device', default='gpu', help=' ')
    optional.add_argument('-f', '--freeze', help='Number of RNN layers to freeze', type=int)
    optional.add_argument('--save_freq', type=int, default=100, help='How often to save models')
    optional.add_argument('--verbose', action='store_true', help='Whether to print loss')
    optional.add_argument('--smiles_prefix', type=str, default=None, help='Smiles prefix added after generation (i.e. for scoring')

    subparsers = parser.add_subparsers(title='RL strategy', dest='rl_strategy',
                                       help='Which reinforcement learning algorithm to use')

    reinvent_parser = subparsers.add_parser('RV', description="REINVENT",
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    reinvent_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    reinvent_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    reinvent_parser.add_argument('-s', '--sigma', type=int, default=60, help='Scaling coefficient of score')
    reinvent_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    reinvent2_parser = subparsers.add_parser('RV2', description="REINVENT (v2.0 defaults)",
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

    augHC_parser = subparsers.add_parser('AHC', description="Augmented Hill-Climb",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    augHC_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    augHC_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    augHC_parser.add_argument('-s', '--sigma', type=int, default=60, help='Scaling coefficient of score')
    augHC_parser.add_argument('-k', '--topk', type=float, default=0.5, help='Fraction of top molecules to keep',
                              metavar="[0-1]")
    augHC_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    HC_parser = subparsers.add_parser('HC', description="Hill-Climb",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    HC_parser.add_argument('--n_steps', type=int, default=30, help=' ')
    HC_parser.add_argument('--batch_size', type=int, default=1024, help=' ')
    HC_parser.add_argument('--epochs_per_step', type=int, default=2, help=' ')
    HC_parser.add_argument('--epochs_batch_size', type=int, default=256, help=' ')
    HC_parser.add_argument('-k', '--topk', type=float, default=0.5, help='Fraction of top molecules to keep',
                           metavar="[0-1]")
    HC_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    HCr_parser = subparsers.add_parser('HC-reg', description="Hill-Climb regularized",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    HCr_parser.add_argument('--n_steps', type=int, default=30, help=' ')
    HCr_parser.add_argument('--batch_size', type=int, default=1024, help=' ')
    HCr_parser.add_argument('--epochs_per_step', type=int, default=2, help=' ')
    HCr_parser.add_argument('--epochs_batch_size', type=int, default=256, help=' ')
    HCr_parser.add_argument('-k', '--topk', type=float, default=0.5, help='Fraction of top molecules to keep',
                            metavar="[0-1]")
    HCr_parser.add_argument('-klc', '--kl_coefficient', type=float, default=10,
                            help='Coefficient of KL loss contribution')
    HCr_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Adam learning rate')

    PG_parser = subparsers.add_parser('RF', description="REINFORCE",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    PG_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    PG_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    PG_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Adam learning rate')

    PGr_parser = subparsers.add_parser('RF-reg', description="REINFORCE regularized",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    PGr_parser.add_argument('--n_steps', type=int, default=500, help=' ')
    PGr_parser.add_argument('--batch_size', type=int, default=64, help=' ')
    PGr_parser.add_argument('-ec', '--entropy_coefficient', type=float, default=0,
                            help='Coefficient of entropy loss contribution')
    PGr_parser.add_argument('-klc', '--kl_coefficient', type=float, default=10,
                            help='Coefficient of KL loss contribution')
    PGr_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Adam learning rate')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # Correct some input arguments
    if args.rl_strategy == 'RV2':
        args.rl_strategy = 'RV'
    # Set agent as prior if not specified
    if args.agent is None:
        setattr(args, 'agent', args.prior)
    main(args)
