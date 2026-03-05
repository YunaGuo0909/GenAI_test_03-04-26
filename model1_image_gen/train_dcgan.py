"""
Train DCGAN on AFHQ (Animal Faces HQ) dataset for 64x64 animal image generation.

Usage:
    python model1_image_gen/train_dcgan.py --data_root model1_image_gen/data/afhq
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from models.dcgan import Generator, Discriminator, weights_init


def get_dataloader(data_root: str, image_size: int, batch_size: int, num_workers: int = 2):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    print(f"Dataset: {len(dataset)} images from {data_root}")
    return loader


def save_grid(tensor: torch.Tensor, path: str, nrow: int = 8):
    """Save a grid of generated images."""
    vutils.save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))


def plot_losses(g_losses: list, d_losses: list, path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("DCGAN Training Loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    loader = get_dataloader(args.data_root, args.image_size, args.batch_size)

    netG = Generator(nz=args.nz, ngf=args.ngf, nc=3).to(device)
    netD = Discriminator(nc=3, ndf=args.ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    print(f"Generator params: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in netD.parameters()):,}")

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Fixed noise for tracking generation quality across epochs
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    g_losses, d_losses = [], []
    real_label, fake_label = 1.0, 0.0

    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (real_imgs, _) in enumerate(pbar):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # --- Train Discriminator: maximize log(D(x)) + log(1 - D(G(z))) ---
            netD.zero_grad()
            label = torch.full((batch_size,), real_label, device=device)
            output = netD(real_imgs)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # --- Train Generator: maximize log(D(G(z))) ---
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            g_losses.append(errG.item())
            d_losses.append(errD.item())

            pbar.set_postfix({
                "D_loss": f"{errD.item():.4f}",
                "G_loss": f"{errG.item():.4f}",
                "D(x)": f"{D_x:.3f}",
                "D(G(z))": f"{D_G_z1:.3f}/{D_G_z2:.3f}",
            })

        # Save samples every epoch
        with torch.no_grad():
            fake_samples = netG(fixed_noise).detach().cpu()
        save_grid(fake_samples, os.path.join(args.output_dir, "samples", f"epoch_{epoch+1:03d}.png"))

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                "epoch": epoch + 1,
                "netG_state_dict": netG.state_dict(),
                "netD_state_dict": netD.state_dict(),
                "optimizerG_state_dict": optimizerG.state_dict(),
                "optimizerD_state_dict": optimizerD.state_dict(),
                "g_losses": g_losses,
                "d_losses": d_losses,
            }, os.path.join(args.output_dir, "checkpoints", f"dcgan_epoch_{epoch+1:03d}.pt"))

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    plot_losses(g_losses, d_losses, os.path.join(args.output_dir, "dcgan_loss_curve.png"))
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCGAN on AFHQ")
    parser.add_argument("--data_root", type=str, default="model1_image_gen/data/afhq/train",
                        help="Path to training images (ImageFolder format)")
    parser.add_argument("--output_dir", type=str, default="model1_image_gen/outputs/dcgan")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nz", type=int, default=100, help="Latent vector dimension")
    parser.add_argument("--ngf", type=int, default=64, help="Generator feature map base size")
    parser.add_argument("--ndf", type=int, default=64, help="Discriminator feature map base size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    args = parser.parse_args()
    train(args)
