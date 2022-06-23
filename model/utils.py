import io
import re
import os
import gzip
from pyrsistent import l
import torch
import logging
import random
import numpy as np
from model.vocabulary import SMILESTokenizer, DeepSMILESTokenizer
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
        raise NotImplementedError
        device = torch.device(f'cuda:{int(device)}')
        #tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
        tensor = torch.FloatTensor
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

def randomize_smiles(smi, n_rand=10, random_type="restricted", keep_last=False):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param smi: A SMILES string
    :param n_rand: Number of randomized smiles per molecule
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :param keep_last: Whether to try and enforce the last atom as always the last atom, not always possible so checked at a later point
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    assert random_type in ['restricted', 'unrestricted'], f"Type {random_type} is not valid"
    mol = Chem.MolFromSmiles(smi)

    if not mol:
        return None

    if random_type == "unrestricted":
        rand_smiles = []
        for i in range(n_rand):
        #while len(rand_smiles) < n_rand:
            if keep_last:
                random_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False, rootedAtAtom=len(smi))
            else:
                rand_smiles.append(Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False))
        return list(set(rand_smiles))

    if random_type == "restricted":
        rand_smiles = []
        for i in range(n_rand):
        #while len(rand_smiles) < n_rand:
            if keep_last:
                new_atom_order = list(range(mol.GetNumAtoms()))
                last_atom = new_atom_order.pop(-1)
                random.shuffle(new_atom_order)
                new_atom_order.append(last_atom)
                random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
                # Check last atom is still last atom... ? How?
                random_smiles = Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
                # Atleast if it's a different symbol ignore
                if smi[-1] != random_smiles[-1]:
                    continue
                rand_smiles.append(random_smiles)
            else:
                new_atom_order = list(range(mol.GetNumAtoms()))
                random.shuffle(new_atom_order)
                random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
                rand_smiles.append(Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False))
        return rand_smiles

def reverse_smiles(smiles):
    """
    Reverse a smiles string while maintaining syntax
    """
    # Reversed tokenized list so square brackets, Br, Cl and double ring numbers remain in order
    tokenizer = DeepSMILESTokenizer(rings=False, branch_tokens=True)
    tdsmiles = tokenizer.tokenize(smiles, with_begin_and_end=False)
    rtdsmiles = list(reversed(tdsmiles))
    # Don't need to Correct brackets
    #bracket_dict = {'(': ')', ')': '('}
    #rtsmiles = [bracket_dict[t] if t in bracket_dict.keys() else t for t in rtsmiles]
    # Push rings back
    ring_idxs = [i for i, t in enumerate(rtdsmiles) if re.search("^[0-9]{1}$|^[0-9]{2}$", t)]
    for i in ring_idxs:
        ring = rtdsmiles.pop(i)
        rtdsmiles.insert(i+1, ring)
    # Push parenthesis back
    #p_idxs = [i for i, t in enumerate(rtdsmiles) if re.search("(\)+)", t)]
    #for i in p_idxs:
    #    p = rtdsmiles.pop(i)
    #    rtdsmiles.insert(i+1, p)
    # Change order of ring numbering
    ring_count = 1
    ring_map = {} # Index to new ring number
    for i, t in enumerate(rtdsmiles):
        if re.search("^[0-9]{1}$|^[0-9]{2}$", t):
            if t in ring_map.keys():
                continue
            else:
                ring_map[t] = str(ring_count)
                ring_count += 1
    rtdsmiles = [ring_map[t] if t in ring_map.keys() else t for t in rtdsmiles]
    # Push ring branches back
    corrected_rtdsmiles = []
    for i, t in enumerate(reversed(rtdsmiles)):
        # Correct parenthesis
        if re.search("\)+", t):
            # Count non character atoms up to length of branch
            correction_count = len(t)
            for ci, ct in enumerate(corrected_rtdsmiles):
                if re.search("^[0-9]+$", ct):
                    correction_count += 1
                elif re.search("^\)+$", ct):
                        correction_count += (1 + len(ct))
                else:
                    if ci == correction_count:
                        break
            print(correction_count)           
            corrected_rtdsmiles.insert(correction_count, t)
        else:
            corrected_rtdsmiles.insert(0, t)
        print(''.join(corrected_rtdsmiles))

    rsmiles = tokenizer.untokenize(corrected_rtdsmiles, convert_to_smiles=True)
    return rsmiles

def reverse_smiles2(smiles, renumber_rings=False):
    """
    Reverse smiles while maintaining syntax
    """
    smiles = "CC(=O)c1cccc(-c2nn(C(C)C)c3ncnc(N)c23)c1"
    smiles = "CCNCC1OC(n2cc(C)c(=O)[nH]c2=O)CC1O"

    # REGEX
    square_brackets = re.compile(r"(\[[^\]]*\])")
    brcl = re.compile(r"(Br|Cl)")
    rings = re.compile(r"([a-zA-Z][0-9]+)")

    # Find parenthesis indexes
    open_count = 0
    close_count = 0
    open_close_idxs = []
    for i, t in enumerate(smiles):
        if (t == '(') and (open_count == 0):
            # Grab branch source
            find_source = True
            count_back = 1
            while find_source:
                if re.match("[a-zA-Z]", smiles[i-count_back]):
                    open_close_idxs.append(i-count_back)
                    find_source = False
                else:
                    count_back += 1
            open_count += 1
        elif t == '(':
            open_count += 1
        elif t == ')':
            close_count += 1
            if open_count == close_count:
                open_close_idxs.append(i)
                open_count = 0
                close_count = 0
        else:
            pass

    # Split by parenthesis indexes
    splitted = []
    for i in range(0, len(open_close_idxs), 2):
        # Add before bracket bit
        if i == 0:
            splitted.append(smiles[:open_close_idxs[i]])
        else:
            splitted.append(smiles[open_close_idxs[i-1]+1: open_close_idxs[i]])
        # Add bracket
        splitted.append(smiles[open_close_idxs[i]:open_close_idxs[i+1]+1])
    # Add bit after
    splitted.append(smiles[open_close_idxs[-1]+1:])

    # Split outside parenthesis
    def split_non_parenthesis(splitted, regex):
        new_splitted = []
        for i, t in enumerate(splitted):
            if re.search("\)$", t):
                new_splitted.extend([t])
            else:
                new_split = [s for s in regex.split(t) if s != '']
                new_splitted.extend(new_split)
        return new_splitted

    for regex in [square_brackets, brcl, rings]:
        splitted = split_non_parenthesis(splitted, regex)

    # Reverse the tokens
    rsplitted = list(reversed(splitted))
    reverse_smiles = ''.join(rsplitted)

    # Re-number the rings in order of appearance
    if renumber_rings:
        # Change order of ring numbering
        ring_count = 1
        ring_map = {} # Index to new ring number
        for i, t in enumerate(rsplitted):
            if re.search(".[0-9]{1}$|.[0-9]{2}$", t):
                if t in ring_map.keys():
                    continue
                else:
                    ring_map[t] = str(ring_count)
                    ring_count += 1
        rtdsmiles = [ring_map[t] if t in ring_map.keys() else t for t in rsplitted]

    # Parenthesis following atom before.
