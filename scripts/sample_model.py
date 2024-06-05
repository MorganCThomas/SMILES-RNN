#!/usr/bin/env python
import argparse
import logging
import os
from functools import partial
from os import path

from promptsmiles import FragmentLinker, ScaffoldDecorator
from rdkit import rdBase
from rdkit.Chem import AllChem as Chem

from smilesrnn import utils
from smilesrnn.gated_transformer import Model as StableTransformerModel
from smilesrnn.rnn import Model as RNNModel
from smilesrnn.transformer import Model as TransformerModel

rdBase.DisableLog("rdApp.error")

logger = logging.getLogger("sample")
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
    device = utils.get_device(args.device)
    logger.info(f"Device set to {device.type}")

    # Load model
    if args.model == "RNN":
        model = RNNModel.load_from_file(
            file_path=args.path, sampling_mode=True, device=device
        )
    elif args.model == "Transformer":
        model = TransformerModel.load_from_file(
            file_path=args.path, sampling_mode=True, device=device
        )
    elif args.model == "GTr":
        model = StableTransformerModel.load_from_file(
            file_path=args.path, sampling_mode=True, device=device
        )
    else:
        print("Model must be either [RNN, Transformer, GTr]")
        raise KeyError

    # Sample TODO different sample modes e.g. beam search
    if args.native:
        if args.psmiles:
            raise NotImplementedError(
                "PromptSMILES is not implemented with non-SMILES grammars"
            )
        smiles, _ = model.sample_native(num=args.number, temperature=args.temperature)
    else:
        if args.psmiles:
            pSMILES_sample = partial(
                model._pSMILES_sample, temperature=args.temperature
            )
            batch_size = 128
            if isinstance(args.psmiles, list):
                psmiles_transform = FragmentLinker(
                    fragments=args.psmiles,
                    batch_size=batch_size,
                    sample_fn=pSMILES_sample,
                    evaluate_fn=model._pSMILES_evaluate,
                    batch_prompts=True,
                    optimize_prompts=True,
                    shuffle=True,
                    scan=False,
                    return_all=False,
                )
            elif isinstance(args.psmiles, str):
                psmiles_transform = ScaffoldDecorator(
                    scaffold=args.psmiles,
                    batch_size=batch_size,
                    sample_fn=pSMILES_sample,
                    evaluate_fn=model._pSMILES_evaluate,
                    batch_prompts=True,
                    optimize_prompts=True,
                    shuffle=True,
                    return_all=False,
                )
            smiles = []
            for batch_size in [batch_size for _ in range(args.number // batch_size)] + [
                args.number % batch_size
            ]:
                smiles += psmiles_transform.sample()
        else:
            smiles, _ = model.sample_smiles(
                num=args.number, temperature=args.temperature
            )

    # If looking for unique only smiles, keep sampling until a unique number is reached
    if args.unique:
        if args.psmiles:
            raise NotImplementedError(
                "Unique sampling is not yet implemented with PromptSMILES"
            )
        logger.info("Canonicalizing smiles")
        canonical_smiles = [
            Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            for smi in smiles
            if Chem.MolFromSmiles(smi)
        ]

        logger.info(f"Topping up {len(set(canonical_smiles))} smiles")
        while len(set(canonical_smiles)) < args.number:
            if args.native:
                new_smiles, _ = model.sample_native(
                    num=(args.number - len(set(canonical_smiles))),
                    temperature=args.temperature,
                    psmiles=args.psmiles,
                )
            else:
                new_smiles, _ = model.sample_smiles(
                    num=(args.number - len(set(canonical_smiles))),
                    temperature=args.temperature,
                    psmiles=args.psmiles,
                )
            new_canonical_smiles = [
                Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                for smi in new_smiles
                if Chem.MolFromSmiles(smi)
            ]
            canonical_smiles += new_canonical_smiles

        smiles = list(set(canonical_smiles))

    # Save
    logger.info(f"Saving {len(set(smiles))} smiles")
    utils.save_smiles(smiles, args.output)
    return


def get_args():
    parser = argparse.ArgumentParser(
        description="Sample smiles from model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p", "--path", type=str, help="Path to checkpoint (.ckpt)", required=True
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Choice of architecture",
        choices=["RNN", "Transformer", "GTr"],
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save file (e.g. Data/Prior_10k.smi)",
        required=True,
    )
    parser.add_argument("-d", "--device", default="gpu", help="Device to use")
    parser.add_argument(
        "-n", "--number", type=int, default=10000, help="Number of smiles to be sampled"
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature to sample (1: multinomial, <1: Less random, >1: More random)",
    )
    parser.add_argument(
        "--psmiles",
        type=str,
        default=None,
        help="Either scaffold smiles labelled with decoration points (*) or fragments for linking with connection points (*) and seperated by a period .",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Keep sampling until n unique canonical molecules have been sampled",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="If trained using an alternative grammar e.g., SELFIES. don't convet back to SMILES",
    )
    args = parser.parse_args()
    # Process prompt smiles
    if args.psmiles and ("." in args.psmiles):
        args.psmiles = args.psmiles.split(".")
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
