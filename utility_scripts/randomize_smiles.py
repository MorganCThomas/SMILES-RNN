#!/usr/bin/env python

import random
from rdkit import Chem
from multiprocessing import Pool
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm.auto import tqdm
from functools import partial
from itertools import chain
from model.utils import read_smiles, save_smiles, randomize_smiles


def get_args():
    parser = ArgumentParser(description='Goal-directed generation benchmark for SMILES RNN',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input smiles (.smi)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output smiles (.smi)'
    )

    optional = parser.add_argument_group('Optional')
    optional.add_argument(
        '--random_type',
        type=str,
        default='restricted',
        help='The type (unrestricted, restricted) of randomization performed.'
    )
    optional.add_argument(
        '--n_rand',
        type=int,
        default=10,
        help='Number of randomized smiles per canonical'
    )
    optional.add_argument(
        '--random_seed',
        type=int,
        default=123,
        help='Random seed'
    )
    optional.add_argument(
        '--no_shuffle',
        action='store_true',
        help='Do not reshuffle data'
    )
    optional.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Number of cores to use'
    )
    args = parser.parse_args()
    return args


def main(args):
    # Read smiles
    smiles = read_smiles(args.input)

    # Randomize smiles
    with Pool(args.n_jobs) as pool:
        prandomize_smiles = partial(randomize_smiles, n_rand=args.n_rand, random_type=args.random_type)
        random_smiles = [rand_smiles
                         for rand_smiles in tqdm(pool.imap(prandomize_smiles, smiles), total=len(smiles))
                         if rand_smiles is not None]
        random_smiles = list(chain.from_iterable(random_smiles))

    # Info
    print(f"{len(random_smiles)} generated from {len(smiles)} "
          f"({len(random_smiles)/len(smiles)} unique randomized smiles on average)")

    # Shuffle
    if not args.no_shuffle:
        random.shuffle(random_smiles)

    # Save output
    save_smiles(random_smiles, args.output)


if __name__ == '__main__':
    args = get_args()
    random.seed(args.random_seed)
    main(args)