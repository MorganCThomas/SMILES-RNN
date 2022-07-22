#!/usr/bin/env python
import os
from os import path
import argparse
import logging

from rdkit import rdBase
from rdkit.Chem import AllChem as Chem

from model.model import *
from model import utils

rdBase.DisableLog("rdApp.error")

logger = logging.getLogger('sample')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def main(args):
    # Make absolute output directory
    output_directory = os.path.dirname(path.abspath(args.output))
    if not path.exists(output_directory):
        os.makedirs(output_directory)

    # Set device
    device = utils.set_default_device_cuda(args.device)
    logger.info(f'Device set to {device.type}')

    # Load model
    model = Model.load_from_file(file_path=args.model, sampling_mode=True, device=device)

    # Sample TODO different sample modes e.g. beam search, temperature
    if args.native:
        smiles, _ = model.sample_native(num=args.number, temperature=args.temperature)
    else:
        smiles, _ = model.sample_smiles(num=args.number, temperature=args.temperature)

    # If looking for unique only smiles, keep sampling until a unique number is reached
    if args.unique:
        logger.info('Canonicalizing smiles')
        canonical_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles
                            if Chem.MolFromSmiles(smi)]

        logger.info(f'Topping up {len(set(canonical_smiles))} smiles')
        while (len(set(canonical_smiles)) < args.number):
            if args.native:
                new_smiles, _ = model.sample_native(num=(args.number - len(set(canonical_smiles))),
                                                    temperature=args.temperature)
            else:
                new_smiles, _ = model.sample_smiles(num=(args.number - len(set(canonical_smiles))),
                                                    temperature=args.temperature)
            new_canonical_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in new_smiles
                                    if Chem.MolFromSmiles(smi)]
            canonical_smiles += new_canonical_smiles

        smiles = list(set(canonical_smiles))

    # Save
    logger.info(f'Saving {len(set(smiles))} smiles')
    utils.save_smiles(smiles, args.output)
    return


def get_args():
    parser = argparse.ArgumentParser(description='Sample smiles from model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', type=str, help='Path to checkpoint (.ckpt)', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to save file (e.g. Data/Prior_10k.smi)', required=True)
    parser.add_argument('-d', '--device', default='gpu', help=' ')
    parser.add_argument('-n', '--number', type=int, default=10000, help=' ')
    parser.add_argument('-t', '--temperature', type=float, default=1.0,
                        help='Temperature to sample (1: multinomial, <1: Less random, >1: More random)')
    parser.add_argument('--unique', action='store_true', help='Keep sampling until n unique canonical molecules have been sampled')
    parser.add_argument('--native', action='store_true', help='If trained using an alternative grammar e.g., SELFIES. don\'t convet back to SMILES')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)