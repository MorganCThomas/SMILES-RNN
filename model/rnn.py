"""
Adaption of RNN model from https://github.com/MolecularAI/Reinvent
"""

from functools import partial
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tf

import model.vocabulary as voc
from model import utils


class RNN(nn.Module):
    """
    Implements a N layer LSTM(M)|GRU(M) cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """

    def __init__(self, voc_size, layer_size=512, num_layers=3, cell_type='lstm', embedding_layer_size=256, dropout=0.,
                 layer_normalization=False):
        """
        Implements a N layer GRU|LSTM cell including an embedding layer and an output linear layer back to the size of the
        vocabulary
        :param voc_size: Size of the vocabulary.
        :param layer_size: Size of each of the RNN layers.
        :param num_layers: Number of RNN layers.
        :param embedding_layer_size: Size of the embedding layer.
        """
        super(RNN, self).__init__()

        self._layer_size = layer_size
        self._embedding_layer_size = embedding_layer_size
        self._num_layers = num_layers
        self._cell_type = cell_type.lower()
        self._dropout = dropout
        self._layer_normalization = layer_normalization

        self._embedding = nn.Embedding(voc_size, self._embedding_layer_size)
        if self._cell_type == 'gru':
            self._rnn = nn.GRU(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                               dropout=self._dropout, batch_first=True)
        elif self._cell_type == 'lstm':
            self._rnn = nn.LSTM(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                                dropout=self._dropout, batch_first=True)
        else:
            raise ValueError('Value of the parameter cell_type should be "gru" or "lstm"')
        self._linear = nn.Linear(self._layer_size, voc_size)

    def forward(self, input_vector, hidden_state=None):
        """
        Performs a forward pass on the model. Note: you pass the **whole** sequence.
        :param input_vector: Input tensor (batch_size, seq_size).
        :param hidden_state: Hidden state tensor.
        """
        batch_size, seq_size = input_vector.size()
        embedded_data = self._embedding(input_vector)  # (batch, seq, embedding)
        if hidden_state is None:
            size = (self._num_layers, batch_size, self._layer_size)
            if self._cell_type == "gru":
                hidden_state = torch.zeros(*size).to(embedded_data.device)
            else:
                hidden_state = [torch.zeros(*size).to(embedded_data.device), torch.zeros(*size).to(embedded_data.device)]
        output_vector, hidden_state_out = self._rnn(embedded_data, hidden_state)

        if self._layer_normalization:
            output_vector = tf.layer_norm(output_vector, output_vector.size()[1:])
        output_vector = output_vector.reshape(-1, self._layer_size)

        output_data = self._linear(output_vector).view(batch_size, seq_size, -1)
        return output_data, None, hidden_state_out

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            'dropout': self._dropout,
            'layer_size': self._layer_size,
            'num_layers': self._num_layers,
            'cell_type': self._cell_type,
            'embedding_layer_size': self._embedding_layer_size
        }


class RNNCritic(nn.Module):
    """
    Adds a critic layer to RNN
    """

    def __init__(self, rnn):
        """
        Implements a N layer GRU|LSTM cell including an embedding layer and an output linear layer back to the size of the
        vocabulary
        :param voc_size: Size of the vocabulary.
        :param layer_size: Size of each of the RNN layers.
        :param num_layers: Number of RNN layers.
        :param embedding_layer_size: Size of the embedding layer.
        """
        super().__init__()

        self.RNN = rnn
        self._critic = nn.Linear(self.RNN._layer_size, 1)

    def forward(self, input_vector, hidden_state=None):
        """
        Performs a forward pass on the model. Note: you pass the **whole** sequence.
        :param input_vector: Input tensor (batch_size, seq_size).
        :param hidden_state: Hidden state tensor.
        """
        batch_size, seq_size = input_vector.size()
        if hidden_state is None:
            size = (self.RNN._num_layers, batch_size, self.RNN._layer_size)
            if self.RNN._cell_type == "gru":
                hidden_state = torch.zeros(*size)
            else:
                hidden_state = [torch.zeros(*size), torch.zeros(*size)]
        embedded_data = self.RNN._embedding(input_vector)  # (batch,seq, embedding)
        output_vector, hidden_state_out = self.RNN._rnn(embedded_data, hidden_state)

        if self.RNN._layer_normalization:
            output_vector = tf.layer_norm(output_vector, output_vector.size()[1:])
        output_vector = output_vector.reshape(-1, self.RNN._layer_size)

        output_data = self.RNN._linear(output_vector).view(batch_size, seq_size, -1)
        critic_data = self._critic(output_vector).view(batch_size, -1)
        return output_data, critic_data, hidden_state_out

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            'dropout': self.RNN._dropout,
            'layer_size': self.RNN._layer_size,
            'num_layers': self.RNN._num_layers,
            'cell_type': self.RNN._cell_type,
            'embedding_layer_size': self.RNN._embedding_layer_size
        }


class Model:
    """
    Implements an RNN model using SMILES.
    """

    def __init__(self, vocabulary: voc.Vocabulary, tokenizer, network_params=None,
                 max_sequence_length=256, device=torch.device('cuda')):
        """
        Implements an RNN.
        :param vocabulary: Vocabulary to use.
        :param tokenizer: Tokenizer to use.
        :param network_params: Dictionary with all parameters required to correctly initialize the RNN class.
        :param max_sequence_length: The max size of SMILES sequence that can be generated.
        """
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.device = device

        if not isinstance(network_params, dict):
            network_params = {}

        self.network = RNN(len(self.vocabulary), **network_params)
        self.network.to(self.device)

        self._nll_loss = nn.NLLLoss(reduction="none").to(device)

    @classmethod
    def load_from_file(cls, file_path: str, sampling_mode=False, device=torch.device('cuda')):
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
            vocabulary=save_dict['vocabulary'],
            tokenizer=save_dict.get('tokenizer', voc.SMILESTokenizer()),
            network_params=network_params,
            max_sequence_length=save_dict['max_sequence_length'],
            device=device
        )
        try:
            if save_dict['network_type'] == 'RNNCritic':
                model.RNN2Critic()
        except KeyError:
            pass
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
            'vocabulary': self.vocabulary,
            'tokenizer': self.tokenizer,
            'max_sequence_length': self.max_sequence_length,
            'network': self.network.state_dict(),
            'network_type': self.network._get_name(),
            'network_params': self.network.get_params()
        }
        torch.save(save_dict, file)

    def likelihood_smiles(self, smiles) -> torch.Tensor:
        tokens = [self.tokenizer.tokenize(smile) for smile in smiles]
        encoded = [self.vocabulary.encode(token) for token in tokens]
        sequences = [torch.tensor(encode, dtype=torch.long) for encode in encoded]

        def collate_fn(encoded_seqs):
            """Function to take a list of encoded sequences and turn them into a batch"""
            max_length = max([seq.size(0) for seq in encoded_seqs])
            collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
            for i, seq in enumerate(encoded_seqs):
                collated_arr[i, :seq.size(0)] = seq
            return collated_arr

        padded_sequences = collate_fn(sequences)
        return self.likelihood(padded_sequences)

    def probability_smiles(self, smiles) -> torch.Tensor:
        tokens = [self.tokenizer.tokenize(smile) for smile in smiles]
        encoded = [self.vocabulary.encode(token) for token in tokens]
        sequences = [torch.tensor(encode, dtype=torch.long) for encode in encoded]

        def collate_fn(encoded_seqs):
            """Function to take a list of encoded sequences and turn them into a batch"""
            max_length = max([seq.size(0) for seq in encoded_seqs])
            collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
            for i, seq in enumerate(encoded_seqs):
                collated_arr[i, :seq.size(0)] = seq
            return collated_arr

        padded_sequences = collate_fn(sequences)
        return self.probabilities(padded_sequences)

    def likelihood(self, sequences) -> torch.Tensor:
        """
        Retrieves the likelihood of a given sequence. Used in training.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size) Log likelihood for each example.
        """
        sequences = sequences.to(self.device)
        logits, _, _ = self.network(sequences[:, :-1])  # all steps done at once
        log_probs = logits.log_softmax(dim=2)
        return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    def probabilities(self, sequences) -> torch.Tensor:
        """
        Retrieves the probabilities of a given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:
          (batch_size, sequence length) Probabilities for each example.
          (batch_size, sequence length) Log probabilities for each example.
        """
        sequences = sequences.to(self.device)
        logits, critic_values, _ = self.network(sequences[:, :])
        probs = logits.softmax(dim=2)
        log_probs = logits.log_softmax(dim=2)
        action_probs = torch.zeros(sequences[:, :].shape)
        action_log_probs = torch.zeros(sequences[:, :].shape)
        for i, (seq, prob, log_prob) in enumerate(zip(sequences[:, :], probs, log_probs)):
            for t, (a, p, lp) in enumerate(zip(seq, prob, log_prob)):
                action_probs[i, t] = p[a]
                action_log_probs[i, t] = lp[a]
        return sequences, logits, action_probs, action_log_probs, critic_values

    def entropy(self, sequences) -> torch.Tensor:
        """
        Retrieves the entropy of a given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size) Entropy for each example.
        """
        logits, _, _ = self.network(sequences[:, :-1])  # all steps done at once
        probs = logits.log_softmax(dim=2)
        log_probs = logits.softmax(dim=2)
        entropies = torch.zeros(probs.shape[0])
        # Non-padding characters i.e. seq == 0
        for i, (seq, prob, log_prob) in enumerate(zip(sequences[:, :-1], probs, log_probs)):
            seq_entropies = []
            for s, p, lp in zip(seq, prob, log_prob):
                if s != 0:
                    seq_entropies.append(-torch.sum(lp * p))
            entropies[i] = torch.tensor(seq_entropies).mean()
        return entropies

    def value_loss(self, sequences, advantage) -> torch.Tensor:
        """
        Given sequence and advantage, calculate the loss of non-padding characters.
        :param sequences: (batch_size, sequence_length) A batch of sequences
        :param advantage: (batch_size) Value loss for each example.
        :return:
        """
        value_loss = torch.zeros(advantage.shape[0])
        # Non-padding characters i.e. seq == 0
        for i, (seq, adv) in enumerate(zip(sequences[:, :-1], advantage)):
            seq_adv = []
            for s, a in zip(seq, adv):
                if s != 0:
                    seq_adv.append(a)
            value_loss[i] = torch.tensor(seq_adv).pow(2).mean()
        return value_loss

    def kl(self, sequences, prior) -> torch.Tensor:
        """
        Retrieves the kl divergence of a given sequence and prior.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :param prior: A prior model
        :return:  (batch_size) Entropy for each example.
        """
        logits, _, _ = self.network(sequences[:, :-1])  # all steps done at once
        prior_logits, _, _ = prior.network(sequences[:, :-1])  # all steps done at once
        probs = logits.softmax(dim=2)
        prior_probs = prior_logits.softmax(dim=2)
        kls = torch.zeros(probs.shape[0])
        # Non-padding characters i.e. seq == 0
        for i, (seq, prob, prior_prob) in enumerate(zip(sequences[:, :-1], probs, prior_probs)):
            seq_kls = []
            for s, p, pp in zip(seq, prob, prior_prob):
                if s != 0:
                    seq_kls.append(torch.sum(p * (p / pp).log()))
            kls[i] = torch.tensor(seq_kls).mean()
        return kls

    def sample_native(self, num=128, batch_size=128, temperature=1.0, partial=None) -> Tuple[List, np.array]:
        """
        Samples n strings from the model according to the native grammar.
        :param num: Number of SMILES to sample.
        :param batch_size: Number of sequences to sample at the same time.
        :return:
            :smiles: (n) A list with SMILES.
            :likelihoods: (n) A list of likelihoods.
        """
        if partial is not None:
            seqs, likelihoods, _, _, _ = self._batch_sample_partial(num=num, temperature=temperature, partial_smiles=partial)
        else:
            seqs, likelihoods, _, _, _ = self._batch_sample(num=num, temperature=temperature)
        smiles = [self.tokenizer.untokenize(self.vocabulary.decode(seq), convert_to_smiles=False)
                  for seq in seqs.cpu().numpy()]
        likelihoods = likelihoods.data.cpu().numpy()
        return smiles, likelihoods

    def sample_smiles(self, num=128, batch_size=128, temperature=1.0, partial=None) -> Tuple[List, np.array]:
        """
        Samples n SMILES from the model.
        :param num: Number of SMILES to sample.
        :param batch_size: Number of sequences to sample at the same time.
        :return:
            :smiles: (n) A list with SMILES.
            :likelihoods: (n) A list of likelihoods.
        """
        if partial is not None:
            seqs, likelihoods, _, _, _ = self._batch_sample_partial(num=num, batch_size=batch_size, temperature=temperature, partial_smiles=partial)
        else:
            seqs, likelihoods, _, _, _ = self._batch_sample(num=num, batch_size=batch_size, temperature=temperature)
        smiles = [self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()]
        likelihoods = likelihoods.data.cpu().numpy()
        return smiles, likelihoods

    def sample_sequences_and_smiles(self, batch_size=128, temperature=1.0, partial=None) -> \
            Tuple[torch.Tensor, List, torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        if partial is not None:
            seqs, likelihoods, probs, log_probs, values = self._batch_sample_partial(num=batch_size, temperature=temperature, partial_smiles=partial)
        else:
            seqs, likelihoods, probs, log_probs, values = self._batch_sample(num=batch_size, temperature=temperature)
        smiles = [self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()]
        return seqs, smiles, likelihoods, probs, log_probs, values

    def _optimize_partial_smiles(self, smi: str, at_idx: int):  # Model
        """
        Optimize partial SMILES for a particular attachment index with respect to the current model
        :param smi: SMILES with (*)
        :param at_idx: Selected attachment index
        :return: Optimal SMILES, respective NLL
        """
        # Possibly optimal smiles (RDKit root must be * idx and not attachment point), need to correct attachment indexes to RDKit indexes
        at_idx = utils.correct_attachment_idx(smi, at_idx)
        rand_smi = utils.randomize_smiles(smi, n_rand=10, random_type='restricted', rootAtom=at_idx, reverse=True)
        if rand_smi is None:
            return smi, None
        with torch.no_grad():
            nlls = self.likelihood_smiles([utils.strip_attachment_points(smi)[0] for smi in rand_smi]).cpu().numpy()
        opt_idx = np.argmin(nlls)
        opt_smi = rand_smi[opt_idx]
        opt_nll = nlls[opt_idx]
        return opt_smi, opt_nll
    
    def _preferred_smiles(self, smiles):
        alt_smiles = list(set([smiles] + utils.randomize_smiles(smiles, keep_last=True) + utils.randomize_smiles(smiles, random_type='unrestricted', keep_last=True)))
        with torch.no_grad():
            nlls = self.likelihood_smiles(alt_smiles).cpu().numpy()
        preferred_smiles, nll = list(sorted(zip(alt_smiles, nlls), key=lambda x: x[1]))[0]
        return preferred_smiles, nll

    def RNN2Critic(self):
        self.network = RNNCritic(self.network)

    def _sample(self, batch_size=64, temperature=1.0, pseq=None) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        
        sequences = torch.zeros((batch_size, self.max_sequence_length), dtype=torch.long)
        sequences[:, 0] = self.vocabulary["^"] 
        input_vector = sequences[:, 0]
        input_vector = input_vector.to(self.device)
        action_probs = torch.zeros((batch_size, self.max_sequence_length), dtype=torch.float, requires_grad=True) 
        action_log_probs = torch.zeros((batch_size, self.max_sequence_length), dtype=torch.float, requires_grad=True) 
        values = torch.zeros((batch_size, self.max_sequence_length), dtype=torch.float, requires_grad=True) \
            if self.network._get_name() == 'RNNCritic' else None
        hidden_state = None
        nlls = torch.zeros(batch_size).to(self.device)
        for t in range(self.max_sequence_length - 1):
            logits, value, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1) / temperature
            probabilities = logits.softmax(dim=1)
            log_probs = logits.log_softmax(dim=1)
            input_vector = torch.multinomial(probabilities, 1).view(-1)

            # Enforce sampling
            if pseq is not None:
                enforce = (pseq[:, t] != 0)
                input_vector = (~enforce * input_vector) + (enforce * pseq[:, t])

            # Store outputs
            sequences[:, t] = input_vector
            action_probs.data[:, t] = torch.tensor([p[a] for p, a in zip(probabilities, input_vector)])
            action_log_probs.data[:, t] = torch.tensor([p[a] for p, a in zip(log_probs, input_vector)])
            if self.network._get_name() == 'RNNCritic':
                values.data[:, t] = value.squeeze(-1) 
            nlls += self._nll_loss(log_probs, input_vector)
            if input_vector.sum() == 0:  # If all sequences terminate, finish early.
                break

        return sequences.data, nlls, action_probs, action_log_probs, values

    def _batch_sample(self, num=128, batch_size=64, temperature=1.0) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        # To ensure all sizes match up, we'll pad with zero and remove non-zero columns after
        sequences = torch.zeros((num, self.max_sequence_length), dtype=torch.long)
        nlls = torch.zeros(num, device=self.device)
        action_probs = torch.zeros((num, self.max_sequence_length), requires_grad=True)
        action_log_probs = torch.zeros((num, self.max_sequence_length), requires_grad=True)
        values = torch.zeros((num, self.max_sequence_length), requires_grad=True) \
            if self.network._get_name() == 'RNNCritic' else None

        # Sample in batches
        batch_sizes = [batch_size for _ in range(num // batch_size)]
        batch_sizes += [num % batch_size] if num % batch_size != 0 else []
        batch_idx = 0
        for size in batch_sizes:
            start_token = torch.zeros(size, dtype=torch.long)
            start_token[:] = self.vocabulary["^"]
            input_vector = start_token
            input_vector = input_vector.to(self.device)
            sequences[batch_idx:batch_idx + size, 0] = self.vocabulary["^"] * torch.ones(size, dtype=torch.long)
            hidden_state = None
            for t in range(1, self.max_sequence_length):
                logits, value, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
                logits = logits.squeeze(1) / temperature
                probabilities = logits.softmax(dim=1)
                log_probs = logits.log_softmax(dim=1)
                input_vector = torch.multinomial(probabilities, 1).view(-1)

                sequences[batch_idx:batch_idx+size, t] = input_vector
                action_probs.data[batch_idx:batch_idx+size, t] = torch.tensor([p[a] for p, a in
                                                                               zip(probabilities, input_vector)])
                action_log_probs.data[batch_idx:batch_idx+size, t] = torch.tensor([p[a] for p, a in
                                                                                   zip(log_probs, input_vector)])
                if self.network._get_name() == 'RNNCritic':
                    values.data[batch_idx:batch_idx+size, t] = value.squeeze(1)

                nlls[batch_idx:batch_idx+size] += self._nll_loss(log_probs, input_vector)

                if input_vector.sum() == 0:  # If all sequences terminate, finish.
                    break

            batch_idx += size

        # Trim any completely non zero cols
        non_zero_cols = [col_idx for col_idx, col in enumerate(torch.split(sequences, 1, dim=1))
                         if not torch.all(col == 0)]
        sequences = sequences[:, non_zero_cols]
        action_probs = action_probs[:, non_zero_cols]
        action_log_probs = action_log_probs[:, non_zero_cols]
        if self.network._get_name() == 'RNNCritic':
            values = values[:, non_zero_cols]

        return sequences.data, nlls, action_probs, action_log_probs, values

    @torch.no_grad()
    def _batch_sample_partial(self, num=128, batch_size=64, temperature=1.0, partial_smiles=None):
        # ----- Prep-process partial smiles
        at_pts = utils.get_attachment_indexes(partial_smiles)
        n_pts = len(at_pts)
        init_psmiles = []
        for aidx in at_pts:
            opt_psmi, _ = self._optimize_partial_smiles(partial_smiles, aidx)
            opt_psmi_stripped, rem_pts = utils.strip_attachment_points(opt_psmi)
            rem_pts.pop(-1)
            tokens = self.tokenizer.tokenize(opt_psmi_stripped, with_begin_and_end=True)
            pseq = torch.tensor(self.vocabulary.encode(tokens), dtype=torch.long)
            init_psmiles.append((pseq, rem_pts))
        
        # ----- Create placeholders
        sequences = torch.zeros((num, self.max_sequence_length), dtype=torch.long)
        nlls = torch.zeros(num, device=self.device)
        action_probs = torch.zeros((num, self.max_sequence_length), requires_grad=False)
        action_log_probs = torch.zeros((num, self.max_sequence_length), requires_grad=False)
        hidden_state = None
        
        # Sample in batches
        batch_sizes = [batch_size for _ in range(num // batch_size)]
        batch_sizes += [num % batch_size] if num % batch_size != 0 else []
        batch_idx = 0
        
        # ---- Sample
        for size in batch_sizes:
            # Randomize selection in batch
            batch_pseq = []
            batch_at_pts = []
            for _ in range(size):
                i = np.random.choice(list(range(n_pts)), 1)[0]
                batch_pseq.append(init_psmiles[i][0])
                batch_at_pts.append(init_psmiles[i][1])
            batch_pseq = torch.vstack([tf.pad(seq, (0, self.max_sequence_length - len(seq))) for seq in batch_pseq])
            batch_at_pts = np.asarray(batch_at_pts)
            
            # Sample a single batch
            batch_seqs, batch_nlls, batch_action_probs, batch_action_log_probs, _ = self._sample(batch_size=size, temperature=temperature, pseq=batch_pseq.to(self.device))
            
            while batch_at_pts.shape[1]:
                # Convert sequences back to SMILES
                batch_smis = [self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in batch_seqs.cpu().numpy()]
                # Insert attachment point
                batch_psmis = [utils.insert_attachment_points(smi, a)[0] for smi, a in zip(batch_smis, batch_at_pts)]
                # Select another attachment point
                sel_pts = [np.random.choice(a, 1, replace=False)[0] for a in batch_at_pts]
                try:
                # Optimize psmiles
                    opt_psmis = [self._optimize_partial_smiles(psmi, s)[0] for psmi, s in zip(batch_psmis, sel_pts)]
                # Strip (batch_at_pts index may have changed) -> (psmi, at_pts)
                    opt_psmis_stripped = [utils.strip_attachment_points(opt_psmi) for opt_psmi in opt_psmis]
                    batch_at_pts = np.asarray([x[1] for x in opt_psmis_stripped])
                    opt_psmis_stripped = [x[0] for x in opt_psmis_stripped]
                    batch_at_pts = np.delete(batch_at_pts, -1, 1) # Remove last attachment point used
                except:
                    from IPython.display import display
                    display(list(enumerate(batch_at_pts)))
                    display(list(enumerate(batch_smis)))
                    display(list(enumerate(batch_psmis)))
                    display(list(enumerate(opt_psmis)))
                    display(list(enumerate(opt_psmis_stripped)))
                    raise
                # Encode
                batch_pseq_tokens = [self.tokenizer.tokenize(opt_psmi_stripped, with_begin_and_end=True) for opt_psmi_stripped in opt_psmis_stripped]
                batch_pseq = torch.vstack([tf.pad(torch.tensor(self.vocabulary.encode(tokens), dtype=torch.long), (0, self.max_sequence_length - len(tokens))) for tokens in batch_pseq_tokens])
                # Re-sample
                batch_seqs, batch_nlls, batch_action_probs, batch_action_log_probs, _ = self._sample(batch_size=size, temperature=temperature, pseq=batch_pseq.to(self.device))
            
            # Update
            sequences[batch_idx:batch_idx+size, :] = batch_seqs
            nlls[batch_idx:batch_idx+size] += batch_nlls
            action_probs.data[batch_idx:batch_idx+size, :] = batch_action_probs
            action_log_probs.data[batch_idx:batch_idx+size, :] = batch_action_log_probs
        
            # Update batch index
            batch_idx += size
        
        # Trim any completely non zero cols
        non_zero_cols = [col_idx for col_idx, col in enumerate(torch.split(sequences, 1, dim=1))
                            if not torch.all(col == 0)]
        sequences = sequences[:, non_zero_cols]
        action_probs = action_probs[:, non_zero_cols]
        action_log_probs = action_log_probs[:, non_zero_cols]
        
        return sequences.data, nlls, action_probs, action_log_probs, None

    def _beam_search(self, k) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
