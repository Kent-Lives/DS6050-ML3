from Attention import Attention
import torch
import torch.nn as nn


class DecoderGRU(nn.Module):

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim

        # TODO: Assign to instance attribute embedding an embedding with the provided output dimension and embedding dimension.
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # TODO: Assign to an instance attribute called GRU a Gated Recurrent Unit with an input size equal to the provided embedding dimension,
        # a hidden size equal to the provided hidden dimension, a number of layers equal to the provided number of layers,
        # input and output tensors organized by batch in their first dimensions,
        # dropout probability equal to the provided dropout probability if the provided number of layers is greater than 1, and
        # dropout probability equal to 0 otherwise.
        self.GRU = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)

        # TODO: Assign to instance attribute attention an object of type Attention.
        self.attention = Attention()

        # TODO: Assign to instance attribute `linear_layer` a linear layer
        # with a number of input features equal to 2 times the provided hidden dimension and
        # a number of output features equal to the provided output dimension.
        self.linear_layer = nn.Linear(2 * hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden, encoder_outputs, src_mask):

        input = input.unsqueeze(1)

        # TODO: Assign to a local variable called intermediate the output of passing the provided input to the embedding.
        intermediate = self.embedding(input)

        # TODO: Assign to a local variable called `tensor_of_embeddings` the output of passing intermediate to dropout.
        tensor_of_embeddings = self.dropout(intermediate)

        # TODO: Assign to local variables `tensor_of_output_features` and `tensor_with_final_hidden_states`
        # the output of passing the tensor of embeddings and the hidden state to the GRU.
        tensor_of_output_features, tensor_with_final_hidden_states = self.GRU(tensor_of_embeddings, hidden)

        # TODO: Remove the sequence dimension from the tensor of output features.
        tensor_of_output_features = tensor_of_output_features.squeeze(1)

        # TODO: Assign to local variables `context_matrix` and `matrix_of_attention_weights`
        # the output of passing the matrix of output features, the provided encoder outputs, and the provided source mask
        # to attention.
        context_matrix, matrix_of_attention_weights = self.attention(tensor_of_output_features, encoder_outputs, src_mask)

        # TODO: Assign to local variable `tensor_of_output_features_and_context`
        # the output of concatenating the matrix of output features and the context matrix.
        tensor_of_output_features_and_context = torch.cat((tensor_of_output_features, context_matrix), dim=1)

        # TODO: Assign to a local variable called `tensor_of_logits`
        # the output of passing the tensor of output features and context to the linear layer.
        tensor_of_logits = self.linear_layer(tensor_of_output_features_and_context)

        return tensor_of_logits, tensor_with_final_hidden_states, matrix_of_attention_weights