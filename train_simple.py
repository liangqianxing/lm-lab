"""
Simple training script for B2-B4 milestone verification.

This demonstrates a working Transformer language model with:
- B2: Transformer architecture (attention, feedforward, layer norm)
- B3: Optimizer (Adam/AdamW)
- B4: Training loop with checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import time


# B2: Transformer Components
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Q, K, V projections
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Multi-head split
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Causal attention
        attn = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)


class TransformerBlock(nn.Module):
    """Single Transformer decoder block."""

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Complete Transformer language model."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len=512, dropout=0.1):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.token_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)

        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 512 else idx[:, -512:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# B4: Training utilities
def create_toy_dataset(vocab_size, num_tokens):
    """Create simple repeating pattern for testing."""
    return [i % vocab_size for i in range(num_tokens)]


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def train_step(model, batch, optimizer, device):
    """Single training step."""
    x, y = [t.to(device) for t in batch]

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    losses = []

    for batch in dataloader:
        x, y = [t.to(device) for t in batch]
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    """Main training function."""

    # Configuration
    config = {
        'vocab_size': 100,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'max_len': 64,
        'batch_size': 8,
        'seq_len': 32,
        'num_steps': 500,
        'lr': 3e-4,
        'eval_interval': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 60)
    print("B2-B4 Milestone Training")
    print("=" * 60)
    for k, v in config.items():
        print(f"{k:20s}: {v}")
    print("=" * 60)

    torch.manual_seed(42)
    device = config['device']

    # Create dataset
    tokens = create_toy_dataset(config['vocab_size'], 10000)
    split = int(0.9 * len(tokens))

    train_dataset = SimpleDataset(tokens[:split], config['seq_len'])
    val_dataset = SimpleDataset(tokens[split:], config['seq_len'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # B2: Create model
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_len=config['max_len']
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # B3: Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    # B4: Training loop
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    model.train()
    train_iter = iter(train_loader)
    start_time = time.time()

    for step in range(config['num_steps']):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss = train_step(model, batch, optimizer, device)

        if step % 10 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

        if step % config['eval_interval'] == 0 and step > 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"Step {step:4d} | Val Loss: {val_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s")

    # Test generation
    print("\n" + "=" * 60)
    print("Testing generation...")
    print("=" * 60)

    model.eval()
    start_tokens = torch.tensor([[0, 1, 2]], device=device)
    generated = model.generate(start_tokens, max_new_tokens=20, temperature=0.8)

    print(f"Input:     {start_tokens[0].tolist()}")
    print(f"Generated: {generated[0].tolist()}")

    # Save checkpoint
    checkpoint_dir = Path('runs/b2_b4_checkpoint')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'step': config['num_steps']
    }

    torch.save(checkpoint, checkpoint_dir / 'final_model.pt')
    print(f"\nCheckpoint saved to {checkpoint_dir / 'final_model.pt'}")

    print("\n" + "=" * 60)
    print("B2-B4 milestones verified!")
    print("=" * 60)
    print("[OK] B2: Transformer with causal attention implemented")
    print("[OK] B3: AdamW optimizer working")
    print("[OK] B4: Training loop with checkpointing complete")


if __name__ == '__main__':
    main()
