import torch


def evaluate(model, iterator, criterion, pad_idx = 0):
    model.eval()
    epoch_loss = 0

    number_of_correct_non_pad_predictions = 0
    number_of_non_pad_target_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_lengths, tgt = batch

            # TODO: Assign to local variable `tensor_of_logits_for_many_tokens` the output of passing the source,
            # the list of lengths, the target, and a teacher forcing ratio of 0 to the provided model.
            tensor_of_logits_for_many_tokens = model(src, src_lengths, tgt, teacher_forcing_ratio=0)

            number_of_tokens_in_vocabulary = tensor_of_logits_for_many_tokens.shape[-1]

            # TODO: Assign to local variable `matrix_of_predicted_token_indices`
            # the matrix of predicted token indices whose elements correspond to batches, positions in sequences,
            # and the maximum logit for a token in the vocabulary.
            matrix_of_predicted_token_indices = tensor_of_logits_for_many_tokens.argmax(-1)

            # TODO: Assign to local variable `non_pad_mask`
            # a matrix of indicators of whether target token indices are not padding.
            non_pad_mask = tgt != pad_idx

            # TODO: Assign to local variable `correct_predictions_mask`
            # a matrix of indicators of whether predicted token indices are equal to target token indices.
            correct_predictions_mask = matrix_of_predicted_token_indices == tgt

            # TODO: Assign to local variable `correct_non_pad_predictions_mask`
            # a matrix of indicators of whether correct predicted token indices are not padding.
            correct_non_pad_predictions_mask = correct_predictions_mask & non_pad_mask

            number_of_correct_non_pad_predictions_for_batch = correct_non_pad_predictions_mask.sum().item()

            # TODO: Add the number of correct predicted token indices that are not padding for the present batch to
            # the total number of correct predicted token indices that are not padding.
            number_of_correct_non_pad_predictions += number_of_correct_non_pad_predictions_for_batch        
            number_of_non_pad_target_tokens += non_pad_mask.sum().item()
            tensor_of_logits_for_many_tokens = tensor_of_logits_for_many_tokens.reshape(-1, number_of_tokens_in_vocabulary)
            tgt = tgt.reshape(-1)

            # TODO: Assign to local variable loss the output of passing the tensor of logits for many tokens and the target
            # to the provided criterion.
            loss = criterion(tensor_of_logits_for_many_tokens, tgt)

            epoch_loss += loss.item()

    average_loss = epoch_loss / len(iterator)

    # TODO: If the number of target token indices that are not padding is greater than 0,
    #     assign to local variable `token_accuracy` the ratio of
    #     the number of correct predicted token indices that are not padding and
    #     the number of target token indices that are not padding.
    # Otherwise, assign to local variable `token_accuracy` 0.
    if number_of_non_pad_target_tokens > 0:
        token_accuracy = number_of_correct_non_pad_predictions / number_of_non_pad_target_tokens
    else:
        token_accuracy = 0

    return average_loss, token_accuracy