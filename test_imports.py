import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

print("Before torch import")
import torch
print("Torch imported")

print("Before datasets import")
from datasets import load_dataset
print("Datasets imported")

print("Before transformers import")
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Transformers imported")

print("Before peft import")
from peft import LoraConfig, get_peft_model
print("PEFT imported")

print("All imports successful!")