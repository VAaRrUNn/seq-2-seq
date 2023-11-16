"""
Trains the vanilla RNN model

Loading and feeding of the sentences:
  - First it loads a batch of sentences 
  - Secondly it converts them into tokens then to tensor
  - Then it dynamically pads the tensors according to the largest tensor using a padding token
  - Then training...
"""
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from dataset_tokenizer import sentencesdataloader
import sys
import logging
from model import RNN
from tqdm.auto import tqdm 
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from tokenizers import Tokenizer

def tokenize_and_pad_batch(tokenizer, sentences, pad_token='[PAD]'):
    # Tokenize all sentences in the batch
    tokenized_outputs = [tokenizer.encode(sentence) for sentence in sentences]
    
    # Find the length of the longest sentence (after tokenization)
    max_len = max(len(tokens.ids) for tokens in tokenized_outputs)

    # Pad all sentences to the length of the longest one
    padded_tokens = [tokens.ids + [tokenizer.token_to_id(pad_token)] * (max_len - len(tokens.ids)) for tokens in tokenized_outputs]

    # Convert to tensor
    padded_tokens_tensor = torch.tensor(padded_tokens, dtype=torch.long)
    return padded_tokens_tensor

def train_rnn(model, batch_sentences, tokenizer, criterion, optim, scheduler):
    model.to(device)
    model.train()
    h_prev = model.initHidden()

    # Tokenizing and padding the batch
    tokens_tensor = tokenize_and_pad_batch(tokenizer, batch_sentences)
    tokens_tensor = tokens_tensor.to(device)

    loss = 0
    h_prev = h_prev.to(device)
    for i in range(tokens_tensor.size(1) - 1): 
        # Extracting the input and target sequences from the batch
        input_seq = tokens_tensor[:, i]
        target_seq = tokens_tensor[:, i + 1]

        # Forward pass
        h_prev, out = model(input_seq, h_prev)
        # out = out.view(-1, vocab_size)  

        # Compute loss (ignore padding tokens)
        pad_token_id = tokenizer.token_to_id('[PAD]')
        mask = target_seq != pad_token_id
        loss += criterion(out[mask], target_seq[mask])

    # Backward pass and optimization
    optim.zero_grad()
    loss.backward()
    scheduler.step()
    return loss.item() / tokens_tensor.size(0)  # Normalize by batch size


try:
    config_path = Path(__file__).parent / "default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    logging.error(f"Error loading configuration file: {e}")
    sys.exit(1)


script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path of the byteBPE.json file
tokenizer_path = os.path.join(script_dir, 'byteBPE.json')
tokenizer = Tokenizer.from_file(tokenizer_path)

vocab_size = tokenizer.get_vocab_size()
in_embd, h_embd = config["in_embd"], config["h_embd"]
epochs = config["epochs"]
m = RNN(in_embd=in_embd, h_embd=h_embd, vocab_size=vocab_size)

try:
  state_dict = torch.load(os.path.join(script_dir,"model_weights.pth"))
  m.load_state_dict(state_dict)
  logging.info(f"Continuing training from where we left off... 🤗")
except Exception as e:
    logging.error(f"Error loading the weights file: {e}")

logging.info(f"Starting Training")

criterion = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(params=m.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)


device = "cuda" if torch.cuda.is_available() else "cpu"
losses = []
logging.info(f"Training on {device}")
for epoch in tqdm(range(epochs)):
  for sentences in sentencesdataloader:
    losses.append(train_rnn(m, sentences, tokenizer, criterion, optim, scheduler))

logging.info(f"Done training")

plt.plot(range(len(losses)), losses)
plt.show()


# saving...
plt.plot(range(len(losses)), losses)
plt.savefig(os.path.join(script_dir, 'loss_plot.png'))
try:
  torch.save(m.state_dict(), os.path.join(script_dir, 'model_weights.pth'))
  logging.info(f"successfully saved weights...")
except Exception as e:
  logging.error(f"Error saving the weights file: {e}")
