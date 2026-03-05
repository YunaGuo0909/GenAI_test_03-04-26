"""
Evaluate and compare DCGAN vs DDPM generated images.

Computes FID (Frechet Inception Distance) between generated and real images,
and produces comparison grids for qualitative evaluation.

Usage:
    python model1_image_gen/evaluate.py --dcgan_ckpt outputs/dcgan/checkpoints/dcgan_epoch_100.pt \
                                         --ddpm_ckpt outputs/ddpm/checkpoints/ddpm_epoch_200.pt \
                                         --data_root data/afhq/val
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, utils as vutils
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from models.dcgan import Generator
from models.ddpm import UNet, GaussianDiffusion


def get_inception_features(images: torch.Tensor, inception: torch.nn.Module,
                           device: torch.device) -> np.ndarray:
    """Extract Inception-v3 pool3 features from a batch of images."""
    # Resize to 299x299 as required by Inception
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    # Denormalize from [-1,1] to [0,1]
    images = (images + 1) / 2
    with torch.no_grad():
        features = inception(images.to(device))
    return features.cpu().numpy()


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute FID between two sets of Inception features."""
    mu_real = real_features.mean(axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_fake = fake_features.mean(axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)


def generate_dcgan_samples(ckpt_path: str, n_samples: int, nz: int = 100,
                           device: torch.device = None) -> torch.Tensor:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    netG = Generator(nz=nz).to(device)
    netG.load_state_dict(checkpoint["netG_state_dict"])
    netG.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, nz, 1, 1, device=device)
        samples = netG(noise).cpu()
    return samples


def generate_ddpm_samples(ckpt_path: str, n_samples: int, image_size: int = 64,
                          base_ch: int = 64, timesteps: int = 1000,
                          device: torch.device = None) -> torch.Tensor:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 0)
    model = UNet(in_ch=3, base_ch=base_ch, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = GaussianDiffusion(timesteps=timesteps, device=device)

    all_samples = []
    batch = min(n_samples, 16)
    for i in tqdm(range(0, n_samples, batch), desc="DDPM sampling"):
        actual_batch = min(batch, n_samples - i)
        c = None
        if num_classes > 0:
            labels = torch.randint(0, num_classes, (actual_batch,))
            c = F.one_hot(labels, num_classes).float().to(device)
        samples = diffusion.sample(model, (actual_batch, 3, image_size, image_size), c)
        all_samples.append(samples.cpu())
    return torch.cat(all_samples, dim=0)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Inception for FID
    print("Loading Inception-v3 for FID computation...")
    inception = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    # Remove the final FC layer to get 2048-d features
    inception.fc = torch.nn.Identity()
    inception = inception.to(device).eval()

    # Get real image features
    print("Computing real image features...")
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    real_dataset = datasets.ImageFolder(root=args.data_root, transform=transform)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=32, shuffle=False)

    real_features = []
    for imgs, _ in tqdm(real_loader, desc="Real features"):
        feats = get_inception_features(imgs, inception, device)
        real_features.append(feats)
    real_features = np.concatenate(real_features, axis=0)

    results = {}

    # DCGAN evaluation
    if args.dcgan_ckpt and os.path.exists(args.dcgan_ckpt):
        print("\nGenerating DCGAN samples...")
        dcgan_samples = generate_dcgan_samples(args.dcgan_ckpt, args.n_samples, device=device)
        vutils.save_image(dcgan_samples[:64], os.path.join(args.output_dir, "dcgan_grid.png"),
                          nrow=8, normalize=True, value_range=(-1, 1))
        print("Computing DCGAN FID...")
        dcgan_features = []
        for i in range(0, len(dcgan_samples), 32):
            batch = dcgan_samples[i:i+32]
            feats = get_inception_features(batch, inception, device)
            dcgan_features.append(feats)
        dcgan_features = np.concatenate(dcgan_features, axis=0)
        results["DCGAN FID"] = compute_fid(real_features, dcgan_features)

    # DDPM evaluation
    if args.ddpm_ckpt and os.path.exists(args.ddpm_ckpt):
        print("\nGenerating DDPM samples...")
        ddpm_samples = generate_ddpm_samples(args.ddpm_ckpt, args.n_samples, device=device)
        vutils.save_image(ddpm_samples[:64], os.path.join(args.output_dir, "ddpm_grid.png"),
                          nrow=8, normalize=True, value_range=(-1, 1))
        print("Computing DDPM FID...")
        ddpm_features = []
        for i in range(0, len(ddpm_samples), 32):
            batch = ddpm_samples[i:i+32]
            feats = get_inception_features(batch, inception, device)
            ddpm_features.append(feats)
        ddpm_features = np.concatenate(ddpm_features, axis=0)
        results["DDPM FID"] = compute_fid(real_features, ddpm_features)

    # Summary
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    for name, fid in results.items():
        print(f"  {name}: {fid:.2f}")
    print("(Lower FID = better quality)")

    with open(os.path.join(args.output_dir, "fid_results.txt"), "w") as f:
        for name, fid in results.items():
            f.write(f"{name}: {fid:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="model1_image_gen/data/afhq/val")
    parser.add_argument("--dcgan_ckpt", type=str, default=None)
    parser.add_argument("--ddpm_ckpt", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="model1_image_gen/outputs/evaluation")
    args = parser.parse_args()
    evaluate(args)
