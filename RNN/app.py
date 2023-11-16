import streamlit as st
from model import RNN
import os
import yaml
from pathlib import Path
import logging
from tokenizers import Tokenizer
import sys
import torch


try:
    config_path = Path(__file__).parent / "default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    logging.error(f"Error loading configuration file: {e}")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    tokenizer = Tokenizer.from_file(os.path.join(script_dir, "byteBPE.json"))
except Exception as e:
    logging.error(f"Error loading the tokenizer file: {e}")


vocab_size = tokenizer.get_vocab_size()
in_embd, h_embd = config["in_embd"], config["h_embd"]
device = "cuda" if torch.cuda.is_available() else "cpu"
m = RNN(in_embd=in_embd, h_embd=h_embd, vocab_size=vocab_size)
m.to(device)

try:
    state_dict_path = os.path.join(
        script_dir, "model_weights.pth")
    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))
    m.load_state_dict(state_dict)
except Exception as e:
    logging.error(f"Error loading the weights file: {e}")
    sys.exit(1)

def generate(model, start_token, max_words=30, device="cpu"):
    model.to(device)
    model.eval()  # Put the model in evaluation mode
    tokens = [tokenizer.token_to_id(start_token)]
    hidden = model.initHidden().to(device)  # Initialize hidden state

    for _ in range(max_words - 1):
        inp = torch.tensor([tokens[-1]], dtype=torch.long).to(device)
        hidden, output = model(inp, hidden)
        next_word_id = torch.multinomial(output, num_samples=1).item()
        tokens.append(next_word_id)
        if tokenizer.id_to_token(next_word_id) == '[END]' or tokenizer.id_to_token(next_word_id) == '[PAD]': 
            break

    return tokenizer.decode(tokens)



st.title("Vanilla RNN")

if 'generated_texts' not in st.session_state:
    st.session_state.generated_texts = []

# Button for generation
if st.button("Generate"):
    # Call your generation function
    generated_text = generate(m,
                              start_token="[START]",
                              device=device)
    st.session_state.generated_texts.append(generated_text)
    # Display all generated texts
    for text in st.session_state.generated_texts:
        st.write(text)
