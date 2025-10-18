import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import time

from customTransformer import Transformer

# Setting the benchmark to Nvidia GPU if you have one
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the vocab sizes and max sequence length with batch parameter for token calculation
src_vocab_size = 10000
tgt_vocab_size = 10000
max_seq_length = 100
batch_size = 64

# Initialize the model
model = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_seq_length=max_seq_length, dropout=0.1).to(device)

# Set the model to evaluation mode
model.eval()

# Generate random input data for validation
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length), device=device)
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length), device=device)

# Define the loss criterion
criterion = nn.CrossEntropyLoss()

# Warms up device for benchmark to get most accurate results
for _ in range(3):
    with torch.no_grad():
        _ = model(val_src_data, val_tgt_data[:, :-1])

# Verifies operations are finished before starting timer
torch.cuda.synchronize()
start_time = time.time()

# Perform validation
with torch.no_grad():
    val_output = model(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))

# Verifies operations are finished before stopping timer
torch.cuda.synchronize()
end_time = time.time()

# Calculations for token and VRAM usage
total_tokens = batch_size * (max_seq_length - 1)
elapsed_time = end_time - start_time
tokens_per_sec = total_tokens / elapsed_time
examples_per_sec = batch_size / elapsed_time

print(f"Validation Loss: {val_loss.item()}")
print(f"Inference Time: {end_time - start_time} seconds")
print(f"Output Shape: {val_output.shape}")
print(f"Tokens per second: {tokens_per_sec}")
print(f"Examples per second: {examples_per_sec}")
print(f"VRAM Allocation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

