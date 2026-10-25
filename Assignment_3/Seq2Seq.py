import random
import torch
import torch.nn as nn


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim
        tensor_of_logits_for_many_tokens = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # TODO: Assign to local variable `tensor_with_final_hidden_states` the output of passing the provided source and the provided list of lengths.
        tensor_with_final_hidden_states = self.encoder(src, src_lengths)
        

        SOS_IDX = 1
        input = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=self.device)
        for t in range(0, tgt_len):

            # TODO: Assign to local variables `tensor_of_logits` and `tensor_with_final_hidden_states`
            # the output passing the input and tensor with final hidden states to the decoder.
            tensor_of_logits, tensor_with_final_hidden_states = self.decoder(input, tensor_with_final_hidden_states)
            

            tensor_of_logits_for_many_tokens[:, t, :] = tensor_of_logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = tensor_of_logits.argmax(1)
            if teacher_force:
                 input = tgt[:, t]
            else:
                 input = top1

        return tensor_of_logits_for_many_tokens