import os
import sys
import time
import json
from os import path
import argparse
import logging
from tqdm.auto import tqdm
from rdkit import rdBase
from itertools import chain

from torch.utils.tensorboard import SummaryWriter

from model.vocabulary import *
from model.rnn import *
from model.transformer import Model as TransformerModel
from model.GTr import Model as StableTransformerModel
from model.dataset import *
from model import utils
from model.utils import randomize_smiles

rdBase.DisableLog("rdApp.error")

logger = logging.getLogger('train_prior')
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

    # Save all args out
    with open(os.path.join(args.output_directory, 'params.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)

    # Set device
    device = utils.set_default_device_cuda(args.device)
    logger.info(f'Device set to {device.type}')

    # Setup Tensorboard
    writer = SummaryWriter(log_dir=path.join(args.output_directory, f'tb_{args.suffix}'))

    # Load smiles
    logger.info('Loading smiles')
    train_smiles = utils.read_smiles(args.train_smiles)
    # Augment by randomization
    if args.randomize:
        logger.info(f'Randomizing {len(train_smiles)} training smiles')
        train_smiles = [randomize_smiles(smi) for smi in tqdm(train_smiles) if randomize_smiles(smi) is not None]
        train_smiles = list(chain.from_iterable(train_smiles))
        logger.info(f'Returned {len(train_smiles)} randomized training smiles')
    # Load other smiles
    all_smiles = train_smiles
    if args.valid_smiles is not None:
        valid_smiles = utils.read_smiles(args.valid_smiles)
        all_smiles += valid_smiles
    if args.test_smiles is not None:
        test_smiles = utils.read_smiles(args.test_smiles)
        all_smiles += test_smiles

    # Set tokenizer
    if args.grammar == 'SMILES':
        tokenizer = SMILESTokenizer()
    if args.grammar == 'deepSMILES':
        tokenizer = DeepSMILESTokenizer()
    if args.grammar == 'deepSMILES_r':
        tokenizer = DeepSMILESTokenizer(branches=False)
    if args.grammar == 'deepSMILES_cr':
        tokenizer = DeepSMILESTokenizer(branches=False, compress=True)
    if args.grammar == 'deepSMILES_b':
        tokenizer = DeepSMILESTokenizer(rings=False)
    if args.grammar == 'deepSMILES_cb':
        tokenizer = DeepSMILESTokenizer(rings=False, compress=True)
    if args.grammar == 'deepSMILES_c':
        tokenizer = DeepSMILESTokenizer(compress=True)
    if args.grammar == 'SELFIES':
        tokenizer = SELFIESTokenizer()
    if args.grammar == 'SmiZip':
        if not args.smizip_ngrams or not os.path.isfile(args.smizip_ngrams):
            sys.exit('ERROR: For the SmiZip grammar, you need to use the --smizip-ngrams option to specify the SmiZip JSON file listing the n-grams')
        with open(args.smizip_ngrams) as inp:
            ngrams = json.load(inp)['ngrams']
        if '$' not in ngrams or '^' not in ngrams:
            sys.exit("ERROR: For the SmiZip grammar, the list of n-grams must include the start and end tokens '^' and '$'. You can add these to an existing SmiZip JSON file using SmiZip's 'add_char_to_ngrams' command")
        tokenizer = SmiZipTokenizer(ngrams)

    # Create vocabulary
    logger.info('Creating vocabulary')
    smiles_vocab = create_vocabulary(smiles_list=all_smiles, tokenizer=tokenizer)

    # Create dataset
    dataset = Dataset(smiles_list=train_smiles, vocabulary=smiles_vocab,
                      tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, collate_fn=Dataset.collate_fn)

    if args.model == 'RNN':
        # Set network params
        network_params = {
            "layer_size": args.layer_size,
            "num_layers": args.num_layers,
            "cell_type": args.cell_type,
            "embedding_layer_size": args.embedding_layer_size,
            "dropout": args.dropout,
            "layer_normalization": args.layer_normalization
        }
        # Create model
        logger.info('Loading model')
        prior = Model(vocabulary=smiles_vocab, tokenizer=tokenizer,
                    network_params=network_params, max_sequence_length=256, device=device)
    elif args.model == 'Transformer':
        # Set network params
        network_params = {
            "n_heads": args.n_heads,
            "n_dims": args.n_dims,
            "ff_dims": args.ff_dims,
            "n_layers": args.n_layers,
            "dropout": args.dropout
        }
        # Create model
        logger.info('Loading model')
        prior = TransformerModel(vocabulary=smiles_vocab, tokenizer=tokenizer,
                            network_params=network_params, max_sequence_length=256, device=device)
    elif args.model == 'GTr':
        # Set network params
        network_params = {
            "n_heads": args.n_heads,
            "n_dims": args.n_dims,
            "ff_dims": args.ff_dims,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "gating": True
        }
        # Create model
        logger.info('Loading model')
        prior = StableTransformerModel(vocabulary=smiles_vocab, tokenizer=tokenizer,
                                    network_params=network_params, max_sequence_length=256, device=device)
    else:
        print("Model must be of type [RNN, Transformer, GTr]")
        raise KeyError

    # Setup optimizer TODO update to adaptive learning
    optimizer = torch.optim.Adam(prior.network.parameters(), lr=args.learning_rate)

    # Train model
    logger.info('Beginning training')
    global_step = 0
    start_time = time.time()
    for e in range(1, args.n_epochs+1):
        logger.info(f'Epoch {e}')
        for step, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            # Update total step
            global_step += 1

            # Sample from DataLoader
            input_vectors = batch.long()

            # Calculate loss
            log_p = prior.likelihood(input_vectors)
            loss = log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            # Throwing an error here, something to do with devices, runs fine on CPU... funny setup with default device?
            optimizer.step()

            # Decrease learning rate
            if (step % 500 == 0) & (step != 0):
                # Decrease learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= (1 - 0.03)  # Decrease by

            # Validate TODO early stopping based on validation NLL
            if step % args.validate_frequency == 0:
                # Validate
                prior.network.eval()
                with torch.no_grad():
                    # Sample new molecules
                    sampled_smiles, sampled_likelihood = prior.sample_smiles()
                    validity, mols = utils.fraction_valid_smiles(sampled_smiles)
                    writer.add_scalar('All/Validity', validity, global_step)
                    writer.add_scalar(f'Epoch {e}/Validity', validity, step)
                    if len(mols) > 0:
                        utils.add_mols(writer, f'Epoch {e}', mols[:10], mols_per_row=5, global_step=step)

                    # Check likelihood on other datasets
                    train_dataloader, _ = calculate_nlls_from_model(prior, train_smiles)
                    train_likelihood = next(train_dataloader)
                    writer.add_scalars(f'All/Train_NLL',
                                       {'sampled': sampled_likelihood.mean(),
                                        'train': train_likelihood.mean()},
                                       global_step)
                    writer.add_scalars(f'Epoch {e}/Train_NLL',
                                       {'sampled': sampled_likelihood.mean(),
                                        'train': train_likelihood.mean()},
                                       step)
                    if args.valid_smiles is not None:
                        valid_dataloader, _ = calculate_nlls_from_model(prior, valid_smiles)
                        valid_likelihood = next(valid_dataloader)
                        writer.add_scalars(f'All/Valid_NLL',
                                           {'sampled': sampled_likelihood.mean(),
                                            'train': train_likelihood.mean(),
                                            'valid': valid_likelihood.mean()},
                                           global_step)
                        writer.add_scalars(f'Epoch {e}/Valid_NLL',
                                           {'sampled': sampled_likelihood.mean(),
                                            'train': train_likelihood.mean(),
                                            'valid': valid_likelihood.mean()},
                                           step)

                    if args.test_smiles is not None:
                        test_dataloader, _ = calculate_nlls_from_model(prior, test_smiles)
                        test_likelihood = next(test_dataloader)
                        writer.add_scalars(f'All/Test_NLL',
                                           {'train': train_likelihood.mean(),
                                            'valid': valid_likelihood.mean(),
                                            'test': test_likelihood.mean()},
                                           global_step)
                        writer.add_scalars(f'Epoch {e}/Test_NLL',
                                           {'train': train_likelihood.mean(),
                                            'valid': valid_likelihood.mean(),
                                            'test': test_likelihood.mean()},
                                           step)
                prior.network.train()

        # Save every epoch
        prior.save(file=path.join(args.output_directory, f'Prior_{args.suffix}_Epoch-{e}.ckpt'))

    # Add training time
    end_time = time.time()
    training_time = (end_time - start_time) / 60  # in minutes
    # Add this to params
    with open(os.path.join(args.output_directory, 'params.json'), 'r') as f:
        params = json.load(f)
    params.update({'total_time_mins': training_time})
    params.update({'epoch_time_mins': training_time/args.n_epochs})
    with open(os.path.join(args.output_directory, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser(description='Train an initial prior model based on smiles data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--train_smiles', type=str, help='Path to smiles file', required=True)
    required.add_argument('-o', '--output_directory', type=str, help='Output directory to save model', required=True)
    required.add_argument('-s', '--suffix', type=str, help='Suffix to name files', required=True)

    #optional = parser.add_argument_group('Optional arguments')
    parser.add_argument('--grammar', choices=['SMILES', 'deepSMILES', 'deepSMILES_r', 'deepSMILES_cr',
                                                'deepSMILES_c', 'deepSMILES_cb', 'deepSMILES_b', 'SELFIES',
                                                'SmiZip'],
                          default='SMILES',
                          help='Choice of grammar to use, SMILES will be encoded and decoded via grammar')
    parser.add_argument('--randomize', action='store_true',
                          help='Training smiles will be randomized using default arguments (10 restricted)')

    parser.add_argument('--smizip-ngrams', help='SmiZip JSON file containing the list of n-grams')
    parser.add_argument('--valid_smiles', help='Validation smiles')
    parser.add_argument('--test_smiles', help='Test smiles')
    parser.add_argument('--validate_frequency', default=500, help=' ')
    parser.add_argument('--n_epochs', type=int, default=5, help=' ')
    parser.add_argument('--batch_size', type=int, default=128, help=' ')
    parser.add_argument('-d', '--device', default='gpu', help='cpu/gpu or device number')

    subparsers = parser.add_subparsers(help='Model architecture', dest='model')

    rnn = subparsers.add_parser('RNN', help='Use simple forward RNN with GRU or LSTM')
    rnn.add_argument('--layer_size', type=int, default=512, help=' ')
    rnn.add_argument('--num_layers', type=int, default=3, help=' ')
    rnn.add_argument('--cell_type', choices=['lstm', 'gru'], default='gru', help=' ')
    rnn.add_argument('--embedding_layer_size', type=int, default=256, help=' ')
    rnn.add_argument('--dropout', type=float, default=0.0, help=' ')
    rnn.add_argument('--learning_rate', type=float, default=1e-3, help=' ')
    rnn.add_argument('--layer_normalization', action='store_true')

    transformer = subparsers.add_parser('Transformer', help='TransformerEncoder model')
    transformer.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    transformer.add_argument('--n_dims', type=int, default=512, help=' ')
    transformer.add_argument('--ff_dims', type=int, default=1024, help='Size of final linear layer')
    transformer.add_argument('--n_layers', type=int, default=4, help='Number of stacked sub-encoders')
    transformer.add_argument('--dropout', type=float, default=0.1, help=' ')
    transformer.add_argument('--learning_rate', type=float, default=1e-3, help=' ')

    GTr = subparsers.add_parser('GTr', help='StableTransformerEncoder model')
    GTr.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    GTr.add_argument('--n_dims', type=int, default=512, help=' ')
    GTr.add_argument('--ff_dims', type=int, default=1024, help='Size of final linear layer')
    GTr.add_argument('--n_layers', type=int, default=4, help='Number of stacked sub-encoders')
    GTr.add_argument('--dropout', type=float, default=0.1, help=' ')
    GTr.add_argument('--learning_rate', type=float, default=1e-3, help=' ')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
