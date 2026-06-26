import math
import os
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

# CONSTANTS
EMBEDDING_DIM = 384
SEQUENCE_LENGTH = 256
BATCH_SIZE = 64  # Size of batch we grab from corpus
NUM_HEADS = 6  # How many attention heads per block
NUM_LAYERS = 6  # How many times we do this whole thing
DROPOUT = 0.0  # Fights memorisation / overfitting
WARMUP_STEPS = 500
MAX_STEPS = 36000
LEARNING_RATE = 6e-4  # How fast we learn
MIN_LR = LEARNING_RATE / 10

CHECKPOINT_PATH = "/Users/olivergreen/Documents/GitHub/OliverEGreen.github.io/LLM-Learning/GPT-2/checkpoint.pt"
DATA_PATH = "/Users/olivergreen/Documents/GitHub/OliverEGreen.github.io/LLM-Learning/GPT-2/fineweb_train.bin"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class SelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.Wq = nn.Linear(d, d, bias=False)  # Query matrix
        self.Wk = nn.Linear(d, d, bias=False)  # Key matrix
        self.Wv = nn.Linear(d, d, bias=False)  # Value matrix
        self.Wo = nn.Linear(d, d, bias=False)  # Output weights
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.attn_drop = nn.Dropout(DROPOUT)
        self.resid_drop = nn.Dropout(DROPOUT)

    def forward(self, belt):
        B, T = belt.shape[:2]  # Grabbing dims from our 3D belt
        # Passing the belt through our matrices, splitting before we add back together later on
        Q = self.Wq(belt).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(belt).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(belt).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.head_dim**0.5)  # Transposing K and matmulling by Q.
        mask = torch.tril(torch.ones(T, T, device=belt.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))  # Causal masking, i.e diagonally setting values to negative infinity to avoid future-peeking.
        weights = F.softmax(scores, dim=-1)  # Softmax to get values 0-1 where eadonech row adds up to 1.0
        weights = self.attn_drop(weights)
        out = weights @ V  # Multiply weights by Value matrix
        out = out.transpose(1, 2).reshape(B, T, self.d)  # Re-adding back together
        return self.resid_drop(self.Wo(out))


class FeedForward(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))  # Using GELU now instead of ReLU


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = SelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.ff = FeedForward(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)  # Token embeddings
        self.pos_emb = nn.Embedding(SEQUENCE_LENGTH, EMBEDDING_DIM)  # Positional embeddings
        self.blocks = nn.ModuleList([Block(EMBEDDING_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)])  # Creating stacks of blocks per layer
        self.ln_f = nn.LayerNorm(EMBEDDING_DIM)  # Layer norming
        self.head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)  # Output weights
        self.head.weight = self.token_emb.weight  # Weight tying
        self.drop = nn.Dropout(DROPOUT)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        T = ids.shape[1]
        positions = torch.arange(T, device=ids.device)
        x = self.drop(self.token_emb(ids) + self.pos_emb(positions))  # Adding positional and token embeddings
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # Final layer norm
        return self.head(x)


def get_batch(split):
    d = train_data if split == "train" else val_data
    starts = torch.randint(0, len(d) - SEQUENCE_LENGTH, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(d[s : s + SEQUENCE_LENGTH].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(d[s + 1 : s + SEQUENCE_LENGTH + 1].astype(np.int64)) for s in starts])
    return x, y


@torch.no_grad()
def estimate_loss(eval_iters=20):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# Generating samples
@torch.no_grad()
def generate(num_tokens):
    model.eval()
    ids = torch.tensor([[enc.eot_token]], dtype=torch.long, device=DEVICE)
    for _ in range(num_tokens):
        context = ids[:, -SEQUENCE_LENGTH:]  # keep only the last SEQUENCE_LENGTH tokens
        logits = model(context)
        last = logits[:, -1, :]  # logits at the final position only → (1, VOCAB_SIZE)
        probs = F.softmax(last, dim=-1)
        next_id = torch.multinomial(probs, 1)  # sample one token from the distribution
        ids = torch.cat([ids, next_id], dim=1)  # stick it on the end, loop
    return ids


def get_lr(step):
    if step < WARMUP_STEPS:  # linear warmup
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    if step > MAX_STEPS:  # floor after the run
        return MIN_LR
    ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)  # 0→1
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))  # 1→0, cosine
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# This only applies if we're training on NVIDIA GPUs (AKA cuda)
ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else nullcontext()

# Getting tokens from OpenAI's tiktoken
enc = tiktoken.get_encoding("gpt2")

data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

VOCAB_SIZE = enc.n_vocab

model = GPT().to(DEVICE)

# Using AdamW optimisation
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Persistence
best_val = float("inf")

# Resuming from loaded weights
start_step = 0

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = checkpoint["step"]
    best_val = checkpoint["best_val"]
    print(f"resumed from step {start_step}, best val {best_val:.4f}")

for step in range(start_step, MAX_STEPS):
    model.train()
    xb, yb = get_batch("train")
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    with ctx:
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
    optimizer.zero_grad()
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping for training safety
    optimizer.step()
    if step % 500 == 0:
        losses = estimate_loss()
        if losses["val"] < best_val:  # saving to disk
            best_val = losses["val"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val": best_val,
                },
                CHECKPOINT_PATH,
            )
            print(f"  saved checkpoint — step {step}, val {best_val:.4f}")
        print(step, "train", round(losses["train"], 4), "val", round(losses["val"], 4))
        print(enc.decode(generate(200)[0].tolist()))
