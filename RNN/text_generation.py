"""
This scripts simply implements the model architecture for text generation using vanilla RNN
"""

# Importing essential libraries
import torch 
import torch.nn as nn
import warnings

# Importing datasets and dataloaders
from datasets_dataloader import sentences_dataloader, sentences_dataset

# For type hint
from typing import Tuple, List, Optional
warnings.filterwarnings("ignore")

