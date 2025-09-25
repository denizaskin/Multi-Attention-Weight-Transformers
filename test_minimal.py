import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
print("Before torch import")
import torch
print("Torch imported successfully")
device = torch.device("cpu")
print(f"Using device: {device}")