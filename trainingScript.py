import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from customTransformer import Transformer

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 20 # Number of batches for training round
MAX_SEQ_LEN = 256 # Max length for tokenization
EPOCHS = 3 # Training Epochs
LR = 5e-5 # Learning rate
NUM_WORKERS = 16  # Use your CPU cores
MODEL_DIM = 512 # Dimension of Transformer embed
N_HEADS = 8 # Number of attention heads
N_LAYERS = 6 # Number of encoder/decoder layers
FF_DIM = 2048 # Feed Forward size
DROPOUT = 0.1 # Dropout for regularization

# Datasets
def get_datasets():
    # Loads WikiText-103
    wikitext = load_dataset("wikitext", "wikitext-103-v1", split="train")

    # Loads OpenAssistant
    oasst = load_dataset("OpenAssistant/oasst1", split="train")

    # Loads Dolly 15k
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")

    return wikitext, oasst, dolly

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2") # GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Encodes conversations into tokens
def encode_example(example, field="text"):
    text = example[field] if field in example else None
    if text is None:
        # Dolly / OASST have instruction/response structure
        if "instruction" in example and "response" in example:
            text = f"Instruction: {example['instruction']}\nResponse: {example['response']}"
        elif "text" in example:
            text = example["text"]
        else:
            text = ""

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        return_tensors="pt"
    )
    return tokens["input_ids"].squeeze(0)



# Gets batches ready for DataLoader
def collate_fn(batch):
    input_ids = [encode_example(x) for x in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return input_ids[:, :-1], input_ids[:, 1:]


from datasets import concatenate_datasets

def train():
    wikitext, oasst, dolly = get_datasets()
    dataset = concatenate_datasets([wikitext, oasst, dolly])

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = Transformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        d_model=MODEL_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=FF_DIM,
        max_seq_length=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for src, tgt in loop:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(src, src)  # simple LM objective
                loss = criterion(output.view(-1, output.size(-1)), tgt.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "trainedTransformer.pt")
    print("Training complete.")


if __name__ == "__main__":
    train()
