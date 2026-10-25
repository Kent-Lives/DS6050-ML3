import torch


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, src_lengths, tgt = batch
        optimizer.zero_grad()

        # TODO: Assign to local variable `tensor_of_logits_for_many_tokens` the output of passing the source,
        # the list of lengths, and the target to the provided model.
        tensor_of_logits_for_many_tokens = model(src, src_lengths, tgt)
        

        number_of_tokens_in_vocabulary = tensor_of_logits_for_many_tokens.shape[-1]
        tensor_of_logits_for_many_tokens = tensor_of_logits_for_many_tokens.reshape(-1, number_of_tokens_in_vocabulary)
        tgt = tgt.reshape(-1)

        # TODO: Assign to local variable loss the output of passing the tensor of logits for many tokens and the target
        # to the provided criterion.
        loss = criterion(tensor_of_logits_for_many_tokens, tgt)

        # TODO: Perform the backward pass.
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # TODO: Update parameters. 
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)