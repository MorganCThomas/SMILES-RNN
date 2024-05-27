"""
Vocabulary helper class from https://github.com/MolecularAI/Reinvent
"""

import copy
import re

import numpy as np
from tqdm import tqdm

try:
    import deepsmiles
except ImportError:
    deepsmiles = None

try:
    import selfies
except ImportError:
    selfies = None

try:
    import smizip
except ImportError:
    smizip = None

try:
    import atomInSmiles as AIS
except ImportError:
    AIS = None

try:
    import safe
except ImportError:
    safe = None


# contains the data structure
class Vocabulary:
    """Stores the tokens and their conversion to vocabulary indexes."""

    def __init__(self, tokens=None, starting_id=0):
        self._tokens = {}
        self._current_id = starting_id

        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

    def add(self, token):
        """Adds a token."""
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        """Adds many tokens."""
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id):
        other_val = self._tokens[token_or_id]
        del self._tokens[other_val]
        del self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        return token_or_id in self._tokens

    def __eq__(self, other_vocabulary):
        return self._tokens == other_vocabulary._tokens  # pylint: disable=W0212

    def __len__(self):
        return len(self._tokens) // 2

    def encode(self, tokens):
        """Encodes a list of tokens as vocabulary indexes."""
        vocab_index = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            vocab_index[i] = self._tokens[token]
        return vocab_index

    def decode(self, vocab_index):
        """Decodes a vocabulary index matrix to a list of tokens."""
        tokens = []
        for idx in vocab_index:
            tokens.append(self[idx])
        return tokens

    def _add(self, token, idx):
        if idx not in self._tokens:
            self._tokens[token] = idx
            self._tokens[idx] = token
        else:
            raise ValueError("IDX already present in vocabulary")

    def tokens(self):
        """Returns the tokens from the vocabulary"""
        return [t for t in self._tokens if isinstance(t, str)]


class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "SMILES"
    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)"),
        "atom": re.compile(r"[a-zA-Z]"),
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def __init__(self):
        self.GRAMMAR = copy.deepcopy(self.GRAMMAR)
        self.REGEXPS = copy.deepcopy(self.REGEXPS)
        self.REGEXP_ORDER = copy.deepcopy(self.REGEXP_ORDER)

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""

        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens, **kwargs):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        return smi


class DeepSMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "deepSMILES"
    REGEXPS = {"brackets": re.compile(r"(\[[^\]]*\])"), "brcl": re.compile(r"(Br|Cl)")}
    REGEXP_ORDER = ["brackets", "brcl"]

    def __init__(self, rings=True, branches=True, compress=False):
        if deepsmiles is None:
            raise ModuleNotFoundError(
                "No module named 'deepsmiles'. Install with 'pip install deepsmiles'."
            )
        self.converter = deepsmiles.Converter(rings=rings, branches=branches)
        self.run_compression = compress

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string via conversion to deepSMILES"""
        data = self.converter.encode(data)
        if self.run_compression:
            data = self.compress(data)

        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens, convert_to_smiles=True):
        """Untokenizes a deepSMILES string followed by conversion to SMILES"""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        if convert_to_smiles:
            try:
                if self.run_compression:
                    smi = self.decompress(smi)
                smi = self.converter.decode(smi)
            except Exception:  # deepsmiles.DecodeError doesn't capture IndexError?
                smi = None
        return smi

    def compress(self, dsmi):
        """
        > compress("C)C")
        'C)1C'
        > compress("C)))C")
        'C)3C'
        > compress("C))))))))))C")
        'C)10C'
        """
        compressed = []
        N = len(dsmi)
        i = 0
        while i < N:
            x = dsmi[i]
            compressed.append(x)
            if x == ")":
                start = i
                while i + 1 < N and dsmi[i + 1] == ")":
                    i += 1
                compressed.append(str(i + 1 - start))
            i += 1
        return "".join(compressed)

    def decompress(self, cdsmi):
        """
        > decompress("C)1C")
        'C)C'
        > decompress("C)3C")
        'C)))C'
        > decompress("C)10C")
        'C))))))))))C'
        > decompress("C)C")
        Traceback (most recent call last):
            ...
        ValueError: A number should follow the parenthesis in C)C
        > decompress("C)")
        Traceback (most recent call last):
            ...
        ValueError: A number should follow the parenthesis in C)
        """
        decompressed = []
        N = len(cdsmi)
        i = 0
        while i < N:
            x = cdsmi[i]
            if x == ")":
                start = i
                while i + 1 < N and cdsmi[i + 1].isdigit():
                    i += 1
                if i == start:
                    raise ValueError(
                        f"A number should follow the parenthesis in {cdsmi}"
                    )
                number = int(cdsmi[start + 1 : i + 1])
                decompressed.append(")" * number)
            else:
                decompressed.append(x)
            i += 1
        return "".join(decompressed)


class SELFIESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "SELFIES"

    def __init__(self):
        if selfies is None:
            raise ModuleNotFoundError(
                "No module named 'selfies'. Install with 'pip install selfies'."
            )

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string via conversion to SELFIES"""
        data = selfies.encoder(data)
        tokens = list(selfies.split_selfies(data))
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens, convert_to_smiles=True):
        """Untokenizes a SELFIES string followed by conversion to SMILES"""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        if convert_to_smiles:
            try:
                smi = selfies.decoder(smi)
            except Exception:
                smi = None
        return smi


class AISTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "AIS"

    def __init__(self):
        if AIS is None:
            raise ModuleNotFoundError(
                "No module named 'atomInSmiles'. Install with 'pip install atomInSmiles'."
            )

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string via conversion to atomInSmiles"""
        data = AIS.encode(data)
        tokens = data.split(" ")
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens, convert_to_smiles=True):
        """Untokenizes an atomInSmiles string followed by conversion to SMILES"""
        smi = ""
        for token in tokens:
            if token == "$":
                smi = smi.rstrip()
                break
            if token != "^":
                smi += token + " "
        if convert_to_smiles:
            try:
                smi = AIS.decode(smi)
            except Exception:
                smi = None
        return smi


class SAFETokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "SAFE"

    def __init__(self):
        if safe is None:
            raise ModuleNotFoundError(
                "No module named 'safe'. Install with 'pip install safe-mol'."
            )

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string via conversion to atomInSmiles"""
        data = safe.encode(data)
        tokens = safe.split(data)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens, convert_to_smiles=True):
        """Untokenizes an atomInSmiles string followed by conversion to SMILES"""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        if convert_to_smiles:
            try:
                smi = safe.decode(smi)
            except Exception:
                smi = None
        return smi


class SmiZipTokenizer:
    """Deals with the tokenization and untokenization of SmiZipped SMILES."""

    GRAMMAR = "SmiZip"

    def __init__(self, ngrams):
        if smizip is None:
            raise ImportError(
                "No module named 'smizip'. Try 'python3 -m pip install smizip'."
            )
        self.zipper = smizip.SmiZip(ngrams)

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string via conversion to SmiZip tokens"""
        tokens = self.zipper.zip(data, format=1)  # format=1 returns the tokens
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens, convert_to_smiles=True):
        """Join the SmiZip tokens to create a SMILES string"""
        ntokens = []
        for token in tokens:
            if token == "$":
                break
            if token == "^":
                continue
            ntokens.append(token)
        return "".join(ntokens) if convert_to_smiles else ",".join(ntokens)


def create_vocabulary(smiles_list, tokenizer):
    """Creates a vocabulary for the SMILES syntax."""
    tokens = set()
    for smi in tqdm(smiles_list, desc="Creating vocabulary"):
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary = Vocabulary()
    vocabulary.update(
        ["$", "^"] + sorted(tokens)
    )  # end token is 0 (also counts as padding)
    return vocabulary


def update_vocabulary(vocabulary, smiles_list, tokenizer):
    """Updates a vocabulary for the SMILES syntax."""
    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary.update(sorted(tokens))  # end token is 0 (also counts as padding)
    return vocabulary


def fit_smiles_to_vocabulary(vocabulary, smiles_list, tokenizer):
    """Creates a vocabulary for the SMILES syntax."""
    fit_smiles = []
    unfit_smiles = []
    unfit_tokens = []
    for smi in smiles_list:
        tokens = tokenizer.tokenize(smi, with_begin_and_end=False)
        if all([token in vocabulary._tokens.keys() for token in tokens]):
            fit_smiles.append(smi)
        else:
            unfit_tokens += [
                token for token in tokens if token not in vocabulary._tokens.keys()
            ]
            unfit_smiles.append(smi)

    if len(unfit_smiles) > 0:
        print(
            f"WARNING: {len(unfit_smiles)} smiles do not fit existing vocabulary due to presence of the"
            f" following tokens\n{set(unfit_tokens)}"
        )

    return fit_smiles
