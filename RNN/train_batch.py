"""
Currently ongoing...
"""
from torch.nn.utils.rnn import pad_sequence

# Assuming sentences is a list of sentences
tokenized_sentences = [torch.tensor(tokenizer.encode(sentence).ids) for sentence in sentences]

# Padding the sequences to have the same length
padded_sequences = pad_sequence(tokenized_sentences, batch_first=True, padding_value=tokenizer.token_to_id('[PAD]'))

def train_rnn(model, batch, tokenizer, criterion, optim, h_prev=None):
    if h_prev is None:
        h_prev = model.initHidden(batch.size(0))  # Initialize hidden state for the batch

    loss = 0
    for i in range(batch.size(1) - 1):  # Iterate over time steps
        input = batch[:, i]
        target = batch[:, i + 1]
        h_prev, out = model(input, h_prev)
        loss += criterion(out, target)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item() / batch.size(1)  # Average loss per time step

batch_size = 32  # Define your batch size

# Assuming sentencesdataloader is an iterable of sentences
for batch in sentencesdataloader:
    # Tokenization and padding for each batch
    tokenized_batch = [torch.tensor(tokenizer.encode(sen).ids) for sen in batch]
    padded_batch = pad_sequence(tokenized_batch, batch_first=True, padding_value=tokenizer.token_to_id('[PAD]'))

    # Making sure the batch is not too large
    num_splits = len(padded_batch) // batch_size + (1 if len(padded_batch) % batch_size != 0 else 0)
    for i in range(num_splits):
        batch_subset = padded_batch[i * batch_size:(i + 1) * batch_size]
        losses.append(train_rnn(m, batch_subset, tokenizer, criterion, optim))
