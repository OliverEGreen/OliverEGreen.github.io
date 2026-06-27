import math
import os
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# CONSTANTS
EMBEDDING_DIM = 768
SEQUENCE_LENGTH = 1024
NUM_HEADS = 12  # How many attention heads per block
NUM_LAYERS = 12  # How many times we do this whole thing
DROPOUT = 0.0  # Fights memorisation / overfitting
BATCH_SIZE = 12  # Size of batch we grab from corpus
GRAD_ACCUM_STEPS = 5  # Gradient accumulation
WARMUP_STEPS = 700
MAX_STEPS = 20000
LEARNING_RATE = 6e-4  # How fast we learn
MIN_LR = LEARNING_RATE / 10

CHECKPOINT_PATH = "checkpoint.pt"
DATA_PATH = "fineweb_train.bin"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# For DDP (multi-GPU usage)
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)
    master_process = int(os.environ["RANK"]) == 0
    seed_offset = int(os.environ["RANK"])
else:
    master_process = True
    seed_offset = 0

torch.manual_seed(1337 + seed_offset)

# Smoke-testing hyperparameter overrides
if os.environ.get("SMOKE"):
    EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS = 64, 4, 2
    SEQUENCE_LENGTH, BATCH_SIZE = 64, 8
    WARMUP_STEPS, MAX_STEPS = 5, 50
    CHECKPOINT_PATH = "smoke_checkpoint.pt"
    GRAD_ACCUM_STEPS = 4


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
        self.resid_drop = nn.Dropout(DROPOUT)

    def forward(self, belt):
        B, T = belt.shape[:2]  # Grabbing dims from our 3D belt
        # Passing the belt through our matrices, splitting before we add back together later on
        Q = self.Wq(belt).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(belt).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(belt).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=DROPOUT if self.training else 0.0, is_causal=True)  # A serious one-liner!
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

        # Faithfulness hyperparameters
        for name, p in self.named_parameters():
            if name.endswith("Wo.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * NUM_LAYERS) ** 0.5)

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
    raw_model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = raw_model(xb)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    raw_model.train()
    return out


# Generating samples
@torch.no_grad()
def generate(num_tokens):
    raw_model.eval()
    ids = torch.tensor([[enc.eot_token]], dtype=torch.long, device=DEVICE)
    for _ in range(num_tokens):
        context = ids[:, -SEQUENCE_LENGTH:]  # keep only the last SEQUENCE_LENGTH tokens
        logits = raw_model(context)
        last = logits[:, -1, :]  # logits at the final position only → (1, VOCAB_SIZE)
        probs = F.softmax(last, dim=-1)
        next_id = torch.multinomial(probs, 1)  # sample one token from the distribution
        ids = torch.cat([ids, next_id], dim=1)  # stick it on the end, loop
    return ids


# Tracking our learning rate so it can be variable instead of hard-coded
def get_lr(step):
    if step < WARMUP_STEPS:  # linear warmup
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    if step > MAX_STEPS:  # floor after the run
        return MIN_LR
    ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)  # 0→1
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))  # 1→0, cosine
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# This only applies if we're training on NVIDIA GPUs (AKA cuda)
ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if DEVICE.startswith("cuda") else nullcontext()

# Getting tokens from OpenAI's tiktoken
enc = tiktoken.get_encoding("gpt2")

data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

VOCAB_SIZE = enc.n_vocab

model = GPT().to(DEVICE)

# Optimisation for NVIDIA hardware
raw_model = model
if DEVICE.startswith("cuda"):
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Using AdamW optimisation and selective weight decay
decay_params = [p for p in model.parameters() if p.dim() >= 2 and p.requires_grad]
nodecay_params = [p for p in model.parameters() if p.dim() < 2 and p.requires_grad]
optim_groups = [
    {"params": decay_params, "weight_decay": 0.1},
    {"params": nodecay_params, "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95), fused=(DEVICE.startswith("cuda")))

# For persistence
best_val = float("inf")

# Resuming from loaded weights
start_step = 0

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    raw_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = checkpoint["step"]
    best_val = checkpoint["best_val"]
    if master_process:
        print(f"resumed from step {start_step}, best val {best_val:.4f}")

# Training loop
for step in range(start_step, MAX_STEPS):
    model.train()
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad()
    for micro in range(GRAD_ACCUM_STEPS):
        xb, yb = get_batch("train")
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with ctx:
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
            loss = loss / GRAD_ACCUM_STEPS
        if ddp:
            model.require_backward_grad_sync = micro == GRAD_ACCUM_STEPS - 1  # type: ignore
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 500 == 0 and master_process:  # Reporting, saving, generating sample
        losses = estimate_loss()
        if losses["val"] < best_val:  # saving to disk
            best_val = losses["val"]
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val": best_val,
                },
                CHECKPOINT_PATH,
            )
            print(f"  saved checkpoint — step {step}, val {best_val:.4f}")
        print(step, "train", round(losses["train"], 4), "val", round(losses["val"], 4))
        print(enc.decode(generate(200)[0].tolist()))

if ddp:
    destroy_process_group()
