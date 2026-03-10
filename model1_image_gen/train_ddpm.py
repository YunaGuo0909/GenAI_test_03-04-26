"""
Train DDPM on animal image datasets for class-conditional image generation.

Supports:
- Stage 1: Pre-train on AFHQ (real animal photos)
- Stage 2: Fine-tune on Dungeons & Diffusion (fantasy art) via --resume
- Classifier-Free Guidance (CFG) throughout

Usage:
    # Stage 1: Pre-train on AFHQ
    python model1_image_gen/train_ddpm.py --data_root /transfer/afhq/train

    # Stage 2: Fine-tune on D&D fantasy art
    python model1_image_gen/train_ddpm.py --data_root /transfer/dnd/images \
        --resume model1_image_gen/outputs/ddpm/checkpoints/ddpm_epoch_200.pt \
        --output_dir model1_image_gen/outputs/ddpm_fantasy \
        --epochs 100 --lr 5e-5
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from PIL import ImageFile, Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.dirname(__file__))
from models.ddpm import UNet, GaussianDiffusion


class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder that skips corrupted images instead of crashing."""

    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except (OSError, SyntaxError, ValueError) as e:
                print(f"  Warning: skipping corrupted image at index {index}: {e}")
                index = (index + 1) % len(self)


def get_dataloader(data_root: str, image_size: int, batch_size: int, num_workers: int = 2):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = SafeImageFolder(root=data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    num_classes = len(dataset.classes)
    print(f"Dataset: {len(dataset)} images, {num_classes} classes: {dataset.classes}")
    return loader, num_classes


def save_grid(tensor: torch.Tensor, path: str, nrow: int = 8):
    vutils.save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Intermediate outputs (samples + non-final checkpoints) go to transfer_dir if set
    # Final checkpoint and loss curve always go to output_dir (project/current)
    if getattr(args, "transfer_dir", None):
        transfer_run = os.path.join(args.transfer_dir, os.path.basename(args.output_dir.rstrip(os.sep)))
        os.makedirs(os.path.join(transfer_run, "samples"), exist_ok=True)
        os.makedirs(os.path.join(transfer_run, "checkpoints"), exist_ok=True)
        print(f"Intermediate outputs → {transfer_run}")
    else:
        transfer_run = None
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    loader, num_classes = get_dataloader(args.data_root, args.image_size, args.batch_size)

    model = UNet(in_ch=3, base_ch=args.base_ch, num_classes=num_classes).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    losses = []

    # Resume from checkpoint (for Stage 2 fine-tuning or crash recovery)
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        ckpt_num_classes = ckpt.get("num_classes", 0)

        if ckpt_num_classes != num_classes:
            # Different number of classes: load compatible weights, reinitialise class layers
            print(f"  Class count changed: {ckpt_num_classes} -> {num_classes} (transfer learning)")
            state = ckpt["model_state_dict"]
            compatible_state = {}
            for k, v in state.items():
                if "class_mlp" in k:
                    print(f"  Skipping {k} (class dimension mismatch)")
                    continue
                if k in model.state_dict() and v.shape == model.state_dict()[k].shape:
                    compatible_state[k] = v
                else:
                    print(f"  Skipping {k} (shape mismatch)")
            model.load_state_dict(compatible_state, strict=False)
            print(f"  Loaded {len(compatible_state)}/{len(model.state_dict())} weight tensors")
        else:
            model.load_state_dict(ckpt["model_state_dict"])
            if not args.reset_optimizer:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr
            start_epoch = ckpt.get("epoch", 0)
            losses = ckpt.get("losses", [])
            print(f"  Resumed from epoch {start_epoch}")

    print(f"UNet params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nStarting DDPM training for {args.epochs} epochs (from epoch {start_epoch})...")
    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{start_epoch + args.epochs}")

        for images, labels in pbar:
            images = images.to(device)
            batch_size = images.size(0)

            c_emb = F.one_hot(labels, num_classes).float().to(device)
            mask = (torch.rand(batch_size, device=device) < args.p_uncond).float()
            c_emb = c_emb * (1 - mask[:, None])

            t = torch.randint(0, args.timesteps, (batch_size,), device=device)
            loss = diffusion.p_losses(model, images, t, c_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        if (epoch + 1) % args.sample_every == 0 or epoch == start_epoch:
            model.eval()
            n_per_class = 4
            samples_per_class = []
            for cls_idx in range(num_classes):
                c = F.one_hot(torch.full((n_per_class,), cls_idx), num_classes).float().to(device)
                samples = diffusion.sample(model, (n_per_class, 3, args.image_size, args.image_size), c)
                samples_per_class.append(samples)
            all_samples = torch.cat(samples_per_class, dim=0).cpu()
            samples_dir = (transfer_run if transfer_run else args.output_dir)
            sample_path = os.path.join(samples_dir, "samples", f"epoch_{epoch+1:03d}.png")
            try:
                save_grid(all_samples, sample_path, nrow=n_per_class)
            except OSError as e:
                print(f"  Warning: could not save sample grid ({e})")

        is_final_epoch = (epoch + 1 == start_epoch + args.epochs)
        if (epoch + 1) % args.save_every == 0 or is_final_epoch:
            ckpt_name = f"ddpm_epoch_{epoch+1:03d}.pt"
            # Non-final checkpoints → transfer_dir; final → output_dir (current/project)
            if transfer_run and not is_final_epoch:
                ckpt_path = os.path.join(transfer_run, "checkpoints", ckpt_name)
            else:
                ckpt_path = os.path.join(args.output_dir, "checkpoints", ckpt_name)
            try:
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "losses": losses,
                    "num_classes": num_classes,
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")
            except (OSError, RuntimeError) as e:
                print(f"  Warning: could not save checkpoint ({e}). Free disk space and try --resume from last successful checkpoint.")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    loss_path = os.path.join(args.output_dir, "ddpm_loss_curve.png")
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("DDPM Training Loss")
        if args.resume:
            plt.axvline(x=start_epoch, color="r", linestyle="--", label="Fine-tune start")
            plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        print(f"Loss curve saved to {loss_path}")
    except OSError as e:
        print(f"Warning: could not save loss curve ({e}). Checkpoints and samples are unchanged.")
    finally:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM on animal images")
    parser.add_argument("--data_root", type=str, default="model1_image_gen/data/afhq/train")
    parser.add_argument("--output_dir", type=str, default="model1_image_gen/outputs/ddpm")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_ch", type=int, default=64, help="UNet base channel width")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--p_uncond", type=float, default=0.1,
                        help="Probability of dropping class label for CFG training")
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resuming / fine-tuning")
    parser.add_argument("--reset_optimizer", action="store_true",
                        help="Reset optimizer state when resuming (use for fine-tuning)")
    parser.add_argument("--transfer_dir", type=str, default=None,
                        help="Save intermediate checkpoints and samples here (e.g. /transfer). Final checkpoint only saved to output_dir.")
    args = parser.parse_args()
    train(args)
