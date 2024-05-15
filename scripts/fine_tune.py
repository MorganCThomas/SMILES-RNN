#!/usr/bin/env python
import argparse
import logging
import os
from itertools import chain
from os import path

import torch
from rdkit import rdBase
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from smilesrnn import utils
from smilesrnn.dataset import Dataset, calculate_nlls_from_model
from smilesrnn.gated_transformer import Model as StableTransformerModel
from smilesrnn.rnn import Model
from smilesrnn.transformer import Model as TransformerModel
from smilesrnn.utils import randomize_smiles
from smilesrnn.vocabulary import fit_smiles_to_vocabulary

rdBase.DisableLog("rdApp.error")

logger = logging.getLogger("fine_tune")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


# ---- Main ----
def main(args):
    # Make absolute output directory
    args.output_directory = path.abspath(args.output_directory)
    if not path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Set device
    device = utils.get_device(args.device)
    logger.info(f"Device set to {device.type}")

    # Setup Tensorboard
    writer = SummaryWriter(
        log_dir=path.join(args.output_directory, f"tb_{args.suffix}")
    )

    # Load smiles
    logger.info("Loading smiles")
    tune_smiles = utils.read_smiles(args.tune_smiles)
    # Augment by randomization
    if args.randomize:
        logger.info(f"Randomizing {len(tune_smiles)} training smiles")
        tune_smiles = [
            randomize_smiles(smi)
            for smi in tqdm(tune_smiles)
            if randomize_smiles(smi) is not None
        ]
        tune_smiles = list(chain.from_iterable(tune_smiles))
        logger.info(f"Returned {len(tune_smiles)} randomized training smiles")
    if args.valid_smiles is not None:
        valid_smiles = utils.read_smiles(args.valid_smiles)
    if args.test_smiles is not None:
        test_smiles = utils.read_smiles(args.test_smiles)

    # Load model
    if args.model == "RNN":
        prior = Model.load_from_file(
            file_path=args.prior, sampling_mode=False, device=device
        )
    elif args.model == "Transformer":
        prior = TransformerModel(
            file_path=args.prior, sampling_mode=False, device=device
        )
    elif args.model == "GTr":
        prior = StableTransformerModel(
            file_path=args.prior, sampling_mode=False, device=device
        )
    else:
        print("Model must be either [RNN, Transformer, GTr]")
        raise KeyError

    # Freeze layers (embedding + 4 parameters per RNN layer)
    if args.freeze is not None:
        n_freeze = args.freeze * 4 + 1
        for i, param in enumerate(prior.network.parameters()):
            if i < n_freeze:  # Freeze parameter
                param.requires_grad = False

    # Set tokenizer
    tokenizer = prior.tokenizer

    # Update smiles to fit vocabulary
    tune_smiles = fit_smiles_to_vocabulary(prior.vocabulary, tune_smiles, tokenizer)

    # Create dataset
    dataset = Dataset(
        smiles_list=tune_smiles, vocabulary=prior.vocabulary, tokenizer=tokenizer
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Dataset.collate_fn
    )

    # Setup optimizer TODO update to adaptive learning
    optimizer = torch.optim.Adam(prior.network.parameters(), lr=0.001)

    # Train model
    logger.info("Beginning training")
    for e in range(1, args.n_epochs + 1):
        logger.info(f"Epoch {e}")
        for step, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            # Sample from DataLoader
            input_vectors = batch.long()

            # Calculate loss
            log_p = prior.likelihood(input_vectors)
            loss = log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            # Throwing an error here with cuda 11 something to do with devices, runs fine on CPU
            optimizer.step()

        # Decrease learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 1 - 0.03  # Decrease by

        # Validate every epoch TODO early stopping based on NLL
        prior.network.eval()
        with torch.no_grad():
            # Sample new molecules
            sampled_smiles, sampled_likelihood = prior.sample_smiles()
            validity, mols = utils.fraction_valid_smiles(sampled_smiles)
            writer.add_scalar("Validity", validity, e)
            if len(mols) > 0:
                utils.add_mols(
                    writer,
                    "Molecules sampled",
                    mols[:10],
                    mols_per_row=5,
                    global_step=e,
                )

            # Check likelihood on other datasets
            tune_dataloader, _ = calculate_nlls_from_model(prior, tune_smiles)
            tune_likelihood = next(tune_dataloader)
            writer.add_scalars(
                "Train_NLL",
                {
                    "sampled": sampled_likelihood.mean(),
                    "tuning": tune_likelihood.mean(),
                },
                e,
            )
            if args.valid_smiles is not None:
                valid_dataloader, _ = calculate_nlls_from_model(prior, valid_smiles)
                valid_likelihood = next(valid_dataloader)
                writer.add_scalars(
                    "Valid_NLL",
                    {
                        "sampled": sampled_likelihood.mean(),
                        "tuning": tune_likelihood.mean(),
                        "valid": valid_likelihood.mean(),
                    },
                    e,
                )

            if args.test_smiles is not None:
                test_dataloader, _ = calculate_nlls_from_model(prior, test_smiles)
                test_likelihood = next(test_dataloader)
                writer.add_scalars(
                    "Test_NLL",
                    {
                        "sampled": sampled_likelihood.mean(),
                        "tuning": tune_likelihood.mean(),
                        "test": test_likelihood.mean(),
                    },
                    e,
                )
        prior.network.train()

        # Save every epoch
        prior.save(
            file=path.join(args.output_directory, f"Prior_{args.suffix}_Epoch-{e}.ckpt")
        )

    return


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-trained prior model based on a smaller dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "-p", "--prior", type=str, help="Path to prior file", required=True
    )
    required.add_argument(
        "-i",
        "--tune_smiles",
        type=str,
        help="Path to fine-tuning smiles file",
        required=True,
    )
    required.add_argument(
        "-o",
        "--output_directory",
        type=str,
        help="Output directory to save model",
        required=True,
    )
    required.add_argument(
        "-s", "--suffix", type=str, help="Suffix to name files", required=True
    )
    required.add_argument(
        "--model",
        type=str,
        choices=["RNN", "Transformer", "GTr"],
        help="Choice of architecture",
        required=True,
    )

    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--randomize",
        action="store_true",
        help="Training smiles will be randomized using default arguments (10 restricted)",
    )
    optional.add_argument("--valid_smiles", help="Validation smiles")
    optional.add_argument("--test_smiles", help="Test smiles")
    optional.add_argument("--n_epochs", type=int, default=10, help=" ")
    optional.add_argument("--batch_size", type=int, default=128, help=" ")
    optional.add_argument(
        "-d", "--device", default="gpu", help="cpu/gpu or device number"
    )
    optional.add_argument(
        "-f", "--freeze", help="Number of RNN layers to freeze", type=int
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
