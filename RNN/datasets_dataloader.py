"""
This script will load the dataset with specified rows

"""


# Essential Libraries
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset

# Other libraries
import yaml
import json # to write dictionaries and variables
import string
import warnings
from pathlib import Path
from collections import defaultdict

# for type hint
from typing import Tuple, List, Optional
warnings.filterwarnings("ignore")


# Loading parameters
# Configuration file

config_path = Path(__file__).parent / "default_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


# Loading necessary things from the confic file
BATCH_SIZE = config["BATCH_SIZE"]
NUM_SENTENCES = config["NUM_SENTENCES"]


class CustomDataset(Dataset):
  def __init__(self, 
               num_sentences=NUM_SENTENCES, 
               batch_size=BATCH_SIZE, 
               split="train"):
    self.rawdata = load_dataset("bookcorpus", split=split, streaming=True)
    self.new_data = list()

    # useful things :)
    self.batch_size = batch_size
    self.num_sentences = num_sentences
    self.word2idx = dict()
    self.current_index = 0 # for updating word2idx

    # execute
    self._make_new_dataset()
    self.idx2word = {k: v for v, k in self.word2idx.items()}
    self.vocab_size = len(self.word2idx)
      

  def __len__(self, ):
    return len(self.new_data)

  def __getitem__(self,
                  idx: int):
    return self.new_data[idx]


  def __replace_unnecessary(self, sen):
    to_replace = "1234567890<>?[]{}_-+=&^%$#@!~`:;|"
    for char in to_replace:
      sen = sen.replace(char, '')

    sen = sen.replace('\\', '')
    sen = sen.replace('//', '')
    return sen

  def _clean(self,
             sentences:List[str]):
    """
    Cleans the raw dataset. Extracts the text part of the sentences in the dataset.
    Also updates the word2idx
    """
    out = list()
    for sentence in sentences:
      text = sentence["text"]
      text = self.__replace_unnecessary(text)
      out.append(text) # list of sentences

    # mapping
    for sentence in out:
      for word in sentence.split():
        word = str(word) # there are numbers too which are causing problems later :( it's just a ugly fix
        if word not in self.word2idx.keys():
          self.word2idx[word] = self.current_index
          self.current_index += 1

    return out

  def _make_new_dataset(self):
    """
    Makes new dataset from the given dataset.
    The dataset was really huge, and we only need some sentences.
    """
    temp_dataloader = DataLoader(self.rawdata, batch_size=self.batch_size, collate_fn=self._clean)
    dataset_size = self.num_sentences // self.batch_size

    for sentences in temp_dataloader:
      self.new_data.extend(sentences)
      dataset_size -= 1
      if dataset_size == 0:
        break
      

print("Starting downloading datasets...")
sentences_dataset = CustomDataset(num_sentences=NUM_SENTENCES,
                                  batch_size=BATCH_SIZE)
sentences_dataloader = DataLoader(dataset=sentences_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
print(f"Done with the sentences dataset...")