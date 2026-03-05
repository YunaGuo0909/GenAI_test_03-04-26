"""
Train the character-level GPT on the species description corpus.

Reads corpus.txt, builds a character-level vocabulary, and trains a small
Transformer decoder to generate structured species descriptions.

Usage:
    python model2_text_gen/train.py
    python model2_text_gen/train.py --n_layer 8 --n_embd 512 --epochs 50
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from models.gpt import GPT


class CharDataset(Dataset):
    """Character-level dataset that produces (input, target) pairs of fixed length."""

    def __init__(self, text: str, block_size: int):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(ch, 0) for ch in s]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos.get(t, "?") for t in tokens)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    corpus_path = os.path.join(os.path.dirname(__file__), "data", "corpus.txt")
    if not os.path.exists(corpus_path):
        print(f"Corpus not found at {corpus_path}")
        print("Run: python model2_text_gen/data/prepare_corpus.py")
        return

    text = open(corpus_path, "r", encoding="utf-8").read()
    print(f"Corpus: {len(text):,} characters")

    dataset = CharDataset(text, args.block_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Vocabulary: {dataset.vocab_size} characters")
    print(f"Training sequences: {len(dataset):,}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # Save vocabulary mapping for inference
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    json.dump({"stoi": dataset.stoi, "itos": {str(k): v for k, v in dataset.itos.items()}},
              open(vocab_path, "w", encoding="utf-8"), ensure_ascii=False)

    model = GPT(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    losses = []
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, dataset.vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Generate a sample every few epochs
        if (epoch + 1) % args.sample_every == 0 or epoch == 0:
            model.eval()
            prompt = "<SPECIES>\nName: "
            prompt_ids = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
            generated = model.generate(prompt_ids, max_new_tokens=400, temperature=0.8, top_k=40)
            text_out = dataset.decode(generated[0].tolist())

            sample_path = os.path.join(args.output_dir, "samples", f"epoch_{epoch+1:03d}.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(text_out)

            # Print a preview
            preview = text_out[:300].replace("\n", " | ")
            print(f"  Sample: {preview}...")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "vocab_size": dataset.vocab_size,
                "block_size": args.block_size,
                "n_embd": args.n_embd,
                "n_head": args.n_head,
                "n_layer": args.n_layer,
            }, os.path.join(args.output_dir, "checkpoints", f"gpt_epoch_{epoch+1:03d}.pt"))

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.title("GPT Training Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "gpt_loss_curve.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train character-level GPT")
    parser.add_argument("--output_dir", type=str, default="model2_text_gen/outputs/gpt")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=20)
    args = parser.parse_args()
    train(args)
