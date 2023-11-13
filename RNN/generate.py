"""
This script runs inference on the RNN model.
It currently generates one sequence at a time (not batched).
"""

from model import RNN
from tokenizers import Tokenizer
import torch
import logging
import yaml
import sys
from pathlib import Path

try:
    config_path = Path(__file__).parent / "default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    logging.error(f"Error loading configuration file: {e}")
    sys.exit(1)

tokenizer = Tokenizer.from_file("tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
in_embd, h_embd = config["in_embd"], config["h_embd"]

m = RNN(in_embd=in_embd, h_embd=h_embd, vocab_size=vocab_size)

try:
  state_dict = torch.load("model_weights.pth")
  m.load_state_dict(state_dict)
except Exception as e:
  logging.error(f"Error loading the weights file: {e}")
  sys.exit(1)

@torch.no_grad()
def generate(model, start_token, max_words=10):
    model.eval()  # Put the model in evaluation mode
    words = [start_token]
    hidden = model.initHidden()  # Initialize hidden state

    for _ in range(max_words - 1):
        input = torch.tensor([tokenizer.token_to_id(words[-1])], dtype=torch.long)
        hidden, output = model(input, hidden)
        next_word_id = output.argmax(dim=1).item()
        next_word = tokenizer.id_to_token(next_word_id)
        words.append(next_word)
        if next_word == '[END]':  # Assuming [END] is your end-of-sequence token
            break

    return ' '.join(words)


import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate text with RNN model")
    parser.add_argument("--max_words", type=int, default=10, help="Maximum number of words to generate")
    args = parser.parse_args()

    start_token = '[START]'  # Replace with your actual start token
    generated_text = generate(m, start_token, args.max_words)
    print(generated_text)

if __name__ == "__main__":
    main()




