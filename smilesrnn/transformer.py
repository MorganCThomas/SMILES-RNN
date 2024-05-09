"""
Simple transformer encoder for SMILES generation
"""

import math
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import smilesrnn.vocabulary as voc


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Implements a N stacked multi-headed attention network for molecule generation
    """

    def __init__(
        self, voc_size, n_heads=8, n_dims=256, ff_dims=1024, n_layers=4, dropout=0.0
    ):
        super(TransformerEncoder, self).__init__()

        self._nheads = n_heads
        self._n_dims = n_dims
        self._ff_dims = ff_dims
        self._dropout = dropout
        self._num_layers = n_layers
        self._layer_normalization = False

        self._embedding = nn.Embedding(voc_size, self._n_dims)
        self._positional_encoder = PositionalEncoding(
            self._n_dims, dropout=self._dropout
        )  # Same dim as embedding
        encoder_layer = nn.TransformerEncoderLayer(
            self._n_dims, self._nheads, self._ff_dims, dropout=self._dropout
        )
        encoder_norm = nn.LayerNorm(self._n_dims)
        self._encoder = nn.TransformerEncoder(
            encoder_layer, self._num_layers, encoder_norm
        )
        self._linear = nn.Linear(self._n_dims, voc_size)

    def forward(self, seqs, no_mask=False):
        """
        Performs forward pass on the model (Transpose input and Output for consistent API with Model)
        :param sequences: Input tensor (batch_size, seq_size)
        :return: Output tensor (batch_size, seq_size, voc_size)
        """
        seqs = seqs.T
        if no_mask:
            mask = None
        else:
            mask = TransformerEncoder.generate_square_subsequent_mask(
                seqs.shape[0]
            )  # .to(self.device)
        embedded = self._embedding(seqs)
        positional_encoded = self._positional_encoder(embedded)
        encoded = self._encoder(positional_encoded, mask=mask)
        out = self._linear(encoded).transpose(
            1, 0
        )  # Transpose to batch first to be consistent with api
        return out

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            "n_heads": self._nheads,
            "n_dims": self._n_dims,
            "ff_dims": self._ff_dims,
            "n_layers": self._num_layers,
            "dropout": self._dropout,
        }


class Model:
    """
    Implements a Transformer Encoder Auto-regressive model
    """

    def __init__(
        self,
        vocabulary: voc.Vocabulary,
        tokenizer,
        network_params=None,
        max_sequence_length=256,
        device=torch.device("cuda"),
    ):
        """
        Implements an RNN.
        :param vocabulary: Vocabulary to use.
        :param tokenizer: Tokenizer to use.
        :param network_params: Dictionary with all parameters required to correctly initialize the Transformer class.
        :param max_sequence_length: The max size of SMILES sequence that can be generated.
        """
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.device = device

        if not isinstance(network_params, dict):
            network_params = {}

        self.network = TransformerEncoder(len(self.vocabulary), **network_params)
        self.network.to(self.device)

        self._nll_loss = nn.NLLLoss(reduction="none").to(self.device)

    @classmethod
    def load_from_file(
        cls, file_path: str, sampling_mode=False, device=torch.device("cuda")
    ):
        """
        Loads a model from a single file
        :param file_path: input file path
        :param sampling_mode: Sampling only, not training
        :param device: Torch device
        :return: new instance of the RNN or an exception if it was not possible to load it.
        """
        if torch.cuda.is_available():
            save_dict = torch.load(file_path, map_location=device)
        else:
            save_dict = torch.load(file_path, map_location=lambda storage, loc: storage)

        network_params = save_dict.get("network_params", {})
        model = Model(
            vocabulary=save_dict["vocabulary"],
            tokenizer=save_dict.get("tokenizer", voc.SMILESTokenizer()),
            network_params=network_params,
            max_sequence_length=save_dict["max_sequence_length"],
            device=device,
        )
        model.network.load_state_dict(save_dict["network"])
        model.network.to(device)
        if sampling_mode:
            # Also disable network gradients ...
            model.network.eval()
            for param in model.network.parameters():
                param.requires_grad = False
        return model

    def save(self, file: str):
        """
        Saves the model into a file
        :param file: it's actually a path
        """
        save_dict = {
            "vocabulary": self.vocabulary,
            "tokenizer": self.tokenizer,
            "max_sequence_length": self.max_sequence_length,
            "network": self.network.state_dict(),
            "network_params": self.network.get_params(),
        }
        torch.save(save_dict, file)

    def likelihood_smiles(self, smiles) -> torch.Tensor:
        tokens = [self.tokenizer.tokenize(smile) for smile in smiles]
        encoded = [self.vocabulary.encode(token) for token in tokens]
        sequences = [torch.tensor(encode, dtype=torch.long) for encode in encoded]

        def collate_fn(encoded_seqs):
            """Function to take a list of encoded sequences and turn them into a batch"""
            max_length = max([seq.size(0) for seq in encoded_seqs])
            collated_arr = torch.zeros(
                len(encoded_seqs), max_length, dtype=torch.long
            )  # padded with zeroes
            for i, seq in enumerate(encoded_seqs):
                collated_arr[i, : seq.size(0)] = seq
            return collated_arr

        padded_sequences = collate_fn(sequences)
        return self.likelihood(padded_sequences)

    def likelihood(self, sequences) -> torch.Tensor:
        """
        Retrieves the likelihood of a given sequence. Used in training.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size) Log likelihood for each example.
        """
        sequences = sequences.to(self.device)
        logits = self.network(
            sequences[:, :-1]
        )  # all steps done at once # Skip last character
        log_probs = logits.log_softmax(dim=2)
        return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    def sample_native(self, num=128, temperature=1.0) -> Tuple[List, np.array]:
        """
        Samples n strings from the model according to the native grammar.
        :param num: Number of SMILES to sample.
        :param batch_size: Number of sequences to sample at the same time.
        :return:
            :smiles: (n) A list with SMILES.
            :likelihoods: (n) A list of likelihoods.
        """
        smiles = []
        likelihoods = []
        batch_sizes = [64 for _ in range(num // 64)]
        batch_sizes += [num % 64] if num % 64 != 0 else []
        for size in batch_sizes:
            batch_seqs, batch_likelihoods, _, _, _ = self._batch_sample(
                num=size, temperature=temperature
            )
            smiles.extend(
                [
                    self.tokenizer.untokenize(
                        self.vocabulary.decode(seq), convert_to_smiles=False
                    )
                    for seq in batch_seqs.cpu().numpy()
                ]
            )
            likelihoods.extend(batch_likelihoods.data.cpu().numpy())
        return smiles, likelihoods

    def sample_smiles(self, num=128, temperature=1.0) -> Tuple[List, np.array]:
        """
        Samples n SMILES from the model.
        :param num: Number of SMILES to sample.
        :return:
            :smiles: (n) A list with SMILES.
            :likelihoods: (n) A list of likelihoods.
        """
        smiles = []
        likelihoods = []
        batch_sizes = [64 for _ in range(num // 64)]
        batch_sizes += [num % 64] if num % 64 != 0 else []
        for size in batch_sizes:
            batch_seqs, batch_likelihoods, _, _, _ = self._batch_sample(
                num=size, temperature=temperature
            )
            smiles.extend(
                [
                    self.tokenizer.untokenize(self.vocabulary.decode(seq))
                    for seq in batch_seqs.cpu().numpy()
                ]
            )
            likelihoods.extend(batch_likelihoods.data.cpu().numpy())
        return smiles, likelihoods

    def sample_sequences_and_smiles(
        self, num=128, temperature=1.0
    ) -> Tuple[
        torch.Tensor,
        List,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, None],
    ]:
        seqs, likelihoods, probs, log_probs, _ = self._batch_sample(
            num=num, temperature=temperature
        )
        smiles = [
            self.tokenizer.untokenize(self.vocabulary.decode(seq))
            for seq in seqs.cpu().numpy()
        ]
        return seqs, smiles, likelihoods, probs, log_probs, None

    def _batch_sample(
        self, num=128, batch_size=64, temperature=1.0
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, None],
    ]:
        with torch.no_grad():
            # To ensure all sizes match up, we'll pad with zero and remove non-zero columns after
            sequences = torch.zeros(
                (num, self.max_sequence_length), dtype=torch.long, requires_grad=False
            )
            action_probs = torch.zeros(
                (num, self.max_sequence_length), requires_grad=False
            )
            action_log_probs = torch.zeros(
                (num, self.max_sequence_length), requires_grad=False
            )
            # nlls = torch.zeros(num)

            # Sample in batches
            batch_sizes = [batch_size for _ in range(num // batch_size)]
            batch_sizes += [num % batch_size] if num % batch_size != 0 else []
            batch_idx = 0
            for size in batch_sizes:
                # TODO has to be done one by one, otherwise only inserting one bloody token!
                input_vectors = torch.zeros(
                    (size, self.max_sequence_length), dtype=torch.long
                )
                input_vectors[:, 0] = self.vocabulary["^"]
                input_vectors = input_vectors.to(self.device)
                sequences[batch_idx : batch_idx + size, 0] = self.vocabulary[
                    "^"
                ] * torch.ones(size, dtype=torch.long)
                for t in range(1, self.max_sequence_length - 1):
                    logits = self.network(input_vectors[:, :t], no_mask=False)[
                        :, -1, :
                    ]  # Final prediction only
                    logits = logits / temperature
                    probabilities = logits.softmax(dim=1)
                    log_probs = logits.log_softmax(dim=1)
                    next_token = torch.multinomial(probabilities, 1).squeeze()
                    input_vectors[:, t] = next_token

                    sequences[batch_idx : batch_idx + size, t] = next_token
                    action_probs.data[batch_idx : batch_idx + size, t] = torch.tensor(
                        [p[a] for p, a in zip(probabilities, next_token)]
                    )
                    action_log_probs.data[batch_idx : batch_idx + size, t] = (
                        torch.tensor([p[a] for p, a in zip(log_probs, next_token)])
                    )

                    # nlls[batch_idx:batch_idx + size] += self._nll_loss(log_probs, next_token)

                    if next_token.sum() == 0:  # If all sequences terminate, finish.
                        break

                batch_idx += size

            # Trim any completely non zero cols
            non_zero_cols = [
                col_idx
                for col_idx, col in enumerate(torch.split(sequences, 1, dim=1))
                if not torch.all(col == 0)
            ]
            sequences = sequences[:, non_zero_cols]
            action_probs = action_probs[:, non_zero_cols]
            action_log_probs = action_log_probs[:, non_zero_cols]

        # Compute nlls afterwards at once to save memory - watch batch size ...
        sequences = sequences.long()
        nlls = self.likelihood(sequences=sequences)

        return sequences.data, nlls, action_probs, action_log_probs, None

    def _pSMILES_evaluate(self, **kwargs):
        raise NotImplementedError(
            "PromptSMILES for Transformers has not yet been implemented"
        )

    def _pSMILES_sample(self, **kwargs):
        raise NotImplementedError(
            "PromptSMILES for Transformers has not yet been implemented"
        )


if __name__ == "__main__":
    import torch
    from model.dataset import Dataset
    from model.transformer import Model
    from model.utils import read_smiles, set_default_device_cuda
    from model.vocabulary import SMILESTokenizer, create_vocabulary

    device = set_default_device_cuda("gpu")
    train_smiles = read_smiles(
        "../../project/Priors/ChEMBL_potent/processed_data/ChEMBL28p_all.smi.gz"
    )
    tokenizer = SMILESTokenizer()
    smiles_vocab = create_vocabulary(smiles_list=train_smiles, tokenizer=tokenizer)
    dataset = Dataset(
        smiles_list=train_smiles, vocabulary=smiles_vocab, tokenizer=tokenizer
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=Dataset.collate_fn
    )
    prior = Model(vocabulary=smiles_vocab, tokenizer=tokenizer, device=device)
    optimizer = torch.optim.Adam(prior.network.parameters(), lr=1e-3)
    step, batch = 0, next(iter(dataloader))
    input_vectors = batch.long()
    log_p = prior.likelihood(input_vectors[:, :-1])
    loss = log_p.mean()
