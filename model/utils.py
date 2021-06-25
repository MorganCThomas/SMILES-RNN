import io
import os
import gzip
import torch
import logging
import numpy as np
import torch.utils.tensorboard.summary as tbxs
from rdkit import Chem
import rdkit.Chem.Draw as Draw

logger = logging.getLogger('utils')


def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def set_default_device_cuda(device='gpu'):
    """Sets the default device (cpu or cuda) used for all tensors."""
    if not torch.cuda.is_available() or (device == 'cpu'):
        device = torch.device('cpu')
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)
        return device
    elif (device in ['gpu', 'cuda']) and torch.cuda.is_available():  # device_name == "cuda":
        device = torch.device('cuda')
        tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
        torch.set_default_tensor_type(tensor)
        return device
    elif torch.cuda.is_available(): # Assume an index
        device = torch.device(f'cuda:{int(device)}')
        tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
        torch.set_default_tensor_type(tensor)
        return device


def read_smiles(file_path):
    """Read a smiles file separated by \n"""
    if any(['gz' in ext for ext in os.path.basename(file_path).split('.')[1:]]):
        logger.debug('\'gz\' found in file path: using gzip')
        with gzip.open(file_path) as f:
            smiles = f.read().splitlines()
            smiles = [smi.decode('utf-8') for smi in smiles]
    else:
        with open(file_path, 'rt') as f:
            smiles = f.read().splitlines()
    return smiles


def save_smiles(smiles, file_path):
    """Save smiles to a file path seperated by \n"""
    if (not os.path.exists(os.path.dirname(file_path))) and (os.path.dirname(file_path) != ''):
        os.makedirs(os.path.dirname(file_path))
    if any(['gz' in ext for ext in os.path.basename(file_path).split('.')[1:]]):
        logger.debug('\'gz\' found in file path: using gzip')
        with gzip.open(file_path, 'wb') as f:
            _ = [f.write((smi+'\n').encode('utf-8')) for smi in smiles if smi is not None]
    else:
        with open(file_path, 'wt') as f:
            _ = [f.write(smi+'\n') for smi in smiles if smi is not None]
    return


def fraction_valid_smiles(smiles):
    i = 0
    mols = []
    for smile in smiles:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                i += 1
                mols.append(mol)
        except TypeError:  # None passed as smile
            pass
    fraction = 100 * i / len(smiles)
    return round(fraction, 2), mols


def add_mol(writer, tag, mol, global_step=None, walltime=None, size=(300, 300)):
    """
    Adds a molecule to the images section of Tensorboard.
    """
    image = Draw.MolToImage(mol, size=size)
    add_image(writer, tag, image, global_step, walltime)


def add_mols(writer, tag, mols, mols_per_row=1, legends=None, global_step=None, walltime=None, size_per_mol=(300, 300)):
    """
    Adds molecules in a grid.
    """
    image = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=size_per_mol, legends=legends)
    add_image(writer, tag, image, global_step, walltime)


def add_image(writer, tag, image, global_step=None, walltime=None):
    """
    Adds an image from a PIL image.
    """
    channel = len(image.getbands())
    width, height = image.size

    output = io.BytesIO()
    image.save(output, format='png')
    image_string = output.getvalue()
    output.close()

    summary_image = tbxs.Summary.Image(height=height, width=width, colorspace=channel,
                                       encoded_image_string=image_string)
    summary = tbxs.Summary(value=[tbxs.Summary.Value(tag=tag, image=summary_image)])
    writer.file_writer.add_summary(summary, global_step, walltime)
