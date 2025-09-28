import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

from customTransformer import Transformer

# Define the vocab sizes and max sequence length
src_vocab_size = 10000
tgt_vocab_size = 10000
max_seq_length = 100

# Initialize the model
model = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_seq_length=max_seq_length, dropout=0.1)

# Set the model to evaluation mode
model.eval()

# Generate random input data for validation
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

# Define the loss criterion
criterion = nn.CrossEntropyLoss()

# Perform validation
with torch.no_grad():
    val_output = model(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")
