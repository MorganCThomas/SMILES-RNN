#!/usr/bin/env python

import random
from multiprocessing import Pool
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm.auto import tqdm
from model.utils import read_smiles, save_smiles

import deepsmiles


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
        '--no_rings',
        action='store_false',
        help='Re-formulate SMILES rings e.g., c1ccccc1 -> cccccc6'
    )
    optional.add_argument(
        '--no_branches',
        action='store_false',
        help='Re-formulate SMILES branching e.g., C(OC)(SC)F -> COC))SC))F'
    )
    optional.add_argument(
        '--no_shuffle',
        action='store_true',
        help='Do not reshuffle data'
    )
    optional.add_argument(
        '--random_seed',
        type=int,
        default=123,
        help='Random seed'
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
    converter = deepsmiles.Converter(rings=args.no_rings, branches=args.no_branches)
    with Pool(args.n_jobs) as pool:
        deep_smiles = [deep_smi for
                       deep_smi in tqdm(pool.imap(converter.encode, smiles), total=len(smiles))
                       if deep_smi is not None]

    # Info
    print(f"{len(set(deep_smiles))} unique deepSMILES generated from {len(set(smiles))} SMILES")

    # Shuffle
    if not args.no_shuffle:
        random.shuffle(deep_smiles)

    # Save output
    save_smiles(deep_smiles, args.output)


if __name__ == '__main__':
    args = get_args()
    random.seed(args.random_seed)
    main(args)