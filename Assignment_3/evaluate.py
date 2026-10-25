import torch


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_lengths, tgt = batch

            # TODO: Assign to local variable `tensor_of_logits_for_many_tokens` the output of passing the source,
            # the list of lengths, the target, and a teacher forcing ratio of 0 to the provided model.
            tensor_of_logits_for_many_tokens = model(src, src_lengths, tgt, teacher_forcing_ratio=0)

            number_of_tokens_in_vocabulary = tensor_of_logits_for_many_tokens.shape[-1]
            tensor_of_logits_for_many_tokens = tensor_of_logits_for_many_tokens.reshape(-1, number_of_tokens_in_vocabulary)
            tgt = tgt.reshape(-1)

            # TODO: Assign to local variable loss the output of passing the tensor of logits for many tokens and the target
            # to the provided criterion.
            loss = criterion(tensor_of_logits_for_many_tokens, tgt)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)