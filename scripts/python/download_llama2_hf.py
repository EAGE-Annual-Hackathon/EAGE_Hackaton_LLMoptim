"""
Downloads Llama-2-7b-chat to /workspace/models folder

IMPORTANT:
    Do `huggingface-cli login` prior to running the script. Otherwise it will throw a misleading error regarding checking the correctness of the data storage path
"""

import os

import torch

from transformers import LlamaForCausalLM, LlamaTokenizer

# TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE")
# LLAMA_MODEL_SIZE = os.getenv("LLAMA_MODEL_SIZE")

TRANSFORMERS_CACHE="/workspace/models"
LLAMA_MODEL_SIZE="7"

print(f'Requested to download Llama-2-{LLAMA_MODEL_SIZE}b-chat-hf and save it to {TRANSFORMERS_CACHE}')
SAVE_PATH = os.path.join(
    TRANSFORMERS_CACHE,
    f"Llama-2-{LLAMA_MODEL_SIZE}b-chat-hf")
TOKENIZER_CHECK = os.path.join(SAVE_PATH, "tokenizer.model")
MODEL_CHECK = os.path.join(SAVE_PATH, "pytorch_model.bin.index.json")

if not os.path.exists(TOKENIZER_CHECK):
    print(f"Instantiating tokenizer at {SAVE_PATH}.")
    tokenizer = LlamaTokenizer.from_pretrained(
        f"meta-llama/Llama-2-{LLAMA_MODEL_SIZE}b-chat-hf", legacy=False)
    tokenizer.save_pretrained(SAVE_PATH)
else:
    print(f"{TOKENIZER_CHECK} already present, skipping.")

if not os.path.exists(MODEL_CHECK):
    print(f"Instantiating model weights at {SAVE_PATH}.")
    model = LlamaForCausalLM.from_pretrained(
        f"meta-llama/Llama-2-{LLAMA_MODEL_SIZE}b-chat-hf", torch_dtype=torch.float16)
    model.save_pretrained(SAVE_PATH)
else:
    print(f"{MODEL_CHECK} already present, skipping.")
