import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderGRU(nn.Module):

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        # TODO: Assign to an instance attribute called embedding an embedding with the provided input dimension and embedding dimension.
        self.embedding = nn.Embedding(input_dim, emb_dim)
        

        # TODO: Assign to an instance attribute called GRU a Gated Recurrent Unit with an input size equal to the provided embedding dimension,
        # a hidden size equal to the provided hidden dimension, a number of layers equal to the provided number of layers,
        # input and output tensors organized by batch in their first dimensions,
        # dropout probability equal to the provided dropout probability if the provided number of layers is greater than 1, and
        # dropout probability equal to 0 otherwise.
        self.GRU = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout)


    def forward(self, src, src_lengths):

        # TODO: Assign to a local variable called intermediate the output of passing the provided source to the embedding.
        intermediate = self.embedding(src)

        # TODO: Assign to a local variable called `tensor_of_embedded_sequences` the output of passing intermediate to dropout.
        tensor_of_embedded_sequences = self.dropout(intermediate)

        # TODO: Assign to a local variable called `packed_sequence` the output of packing the tensor of embedded sequences.
        # Specify the list of sequence lengths of each batch element as the provided lengths moved to CPU.
        # Specify that the input tensor is organized by batch in its first dimension.
        # Do not enforce sorting.
        packed_sequence = pack_padded_sequence(tensor_of_embedded_sequences, src_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # TODO: Assign to local variables `packed_sequence_of_output_features` and `tensor_with_final_hidden_states` the output of passing `packed_sequence` to the GRU.
        packed_sequence_of_output_features, tensor_with_final_hidden_states = self.GRU(packed_sequence)

        return tensor_with_final_hidden_states