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

        # TODO: Assign to instance attribute `linear_layer` a linear layer with the provided hidden dimension and output dimension.
        self.linear_layer = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden):

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

        # TODO: Assign to a local variable called `tensor_of_logits` the output of passing the tensor of output features to the linear layer.
        tensor_of_logits = self.linear_layer(tensor_of_output_features)

        return tensor_of_logits, tensor_with_final_hidden_states