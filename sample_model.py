#!/usr/bin/env python
import os
from os import path
import argparse
import logging
from rdkit import rdBase

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
    smiles, _ = model.sample_smiles(num=args.number)

    # Save
    with open(args.output, 'wt') as f:
        _ = [f.write(smi+'\n') for smi in smiles]
    return


def get_args():
    parser = argparse.ArgumentParser(description='Sample smiles from model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', type=str, help='Path to checkpoint (.ckpt)', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to save file e.g. Data/Prior_10k.smi)', required=True)
    parser.add_argument('-d', '--device', default='gpu', help=' ')
    parser.add_argument('-n', '--number', type=int, default=10000, help=' ')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)