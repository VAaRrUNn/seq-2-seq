from pre_processing import sentencesdataloader
import logging
from tokenizers import Tokenizer
import torch 
from tqdm.auto import tqdm
import torch.nn as nn
from model import RNN
import matplotlib.pyplot as plt


def train_rnn(model, sen, tokenizer, criterion, optim, h_prev=None):
  if h_prev == None:
    h_prev = model.initHidden()

  # tokenizing the sentence
  tokens = torch.tensor(tokenizer.encode(sen).ids)
  loss = 0
  for i in range(len(tokens)-1):
    h_prev, out = m(tokens[i], h_prev)
    out = out.view(-1)
    target = tokens[i+1]
    loss += criterion(out, target)
  optim.zero_grad()
  loss.backward()
  optim.step()
  return loss.item()
    

# 

tokenizer = Tokenizer.from_file("tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
in_embd, h_embd = 10, 12
m = RNN(in_embd=in_embd, h_embd=h_embd, vocab_size=vocab_size)

logging.info(f"Starting Training")

criterion = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(params=m.parameters(), lr=1e-3)
losses = []

for sentences in tqdm(sentencesdataloader):
  for sen in sentences:
    losses.append(train_rnn(m, sen, tokenizer, criterion, optim))
  break


plt.plot(range(len(losses)), losses)