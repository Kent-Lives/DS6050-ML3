import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def forward(self, decoder_hidden, encoder_outputs, src_mask = None):

        # TODO: Assign to local variable `tensor_of_current_decoder_hidden_states`
        # the output of converting the provided tensor of current decoder hidden states
        # from (batch size, number of hidden states) to (batch size, number of hidden states, 1).
        tensor_of_current_decoder_hidden_states = decoder_hidden.unsqueeze(2)
        

        # TODO: Assign to local variable `tensor_of_alignment_scores`
        # the output of batch matrix multiplying the provided tensor of encoder hidden states
        # and the tensor of current decoder hidden states.
        tensor_of_alignment_scores = torch.bmm(encoder_outputs, tensor_of_current_decoder_hidden_states)
        

        # TODO: Assign to local variable `matrix_of_alignment_scores`
        # the output of converting the tensor of alignment scores from (batch size, number of time steps, 1)
        # to (batch size, number of time steps).
        matrix_of_alignment_scores = tensor_of_alignment_scores.squeeze(2)

        # TODO: If the provided mask describing which tokens are real and which are padding exists,
        #   reassign to `matrix_of_alignment_scores` the output of filling the matrix of alignment scores with `-1e9`,
        #   a large negative number, when corresponding values of the mask are false.
        if src_mask is not None:
            matrix_of_alignment_scores = matrix_of_alignment_scores.masked_fill_(src_mask == 0, -1e9)
        
        # TODO: Assign to local variable `matrix_of_attention_weights`
        # the output of passing the matrix of alignment scores to the softmax function from module functional.
        # Compute softmax along dimension 1.
        matrix_of_attention_weights = F.softmax(matrix_of_alignment_scores, dim=1)

        # TODO: Assign to local variable `tensor_of_attention_weights`
        # the output of converting the provided matrix of attention weights
        # from (batch size, number of time steps) to (batch size, 1, number of time steps).
        tensor_of_attention_weights = matrix_of_attention_weights.unsqueeze(1)

        # TODO: Assign to local variable `context_tensor` the output of batch matrix multiplying
        # the tensor of attention weights and the tensor of encoder hidden states.
        context_tensor = torch.bmm(tensor_of_attention_weights, encoder_outputs)
        
        # TODO: Assign to local variable `context_matrix` the output of converting the context tensor
        # from (batch size, 1, number of hidden states) to (batch size, number of hidden states).
        context_matrix = context_tensor.squeeze(1)

        # TODO: Return the context matrix and matrix of attention weights.
        return context_matrix, matrix_of_attention_weights