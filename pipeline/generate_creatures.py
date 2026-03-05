"""
Generate virtual creatures by orchestrating both models.

Loads trained DDPM (image) and GPT (text) models, then generates
a batch of virtual creature entries with paired images and descriptions.

Output: a JSON manifest + image files ready for the website generator.

Usage:
    python pipeline/generate_creatures.py \
        --image_ckpt model1_image_gen/outputs/ddpm/checkpoints/ddpm_epoch_200.pt \
        --text_ckpt model2_text_gen/outputs/gpt/checkpoints/gpt_epoch_100.pt \
        --n 50
"""

import argparse
import hashlib
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
from torchvision import utils as vutils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model1_image_gen.models.ddpm import UNet, GaussianDiffusion
from model2_text_gen.generate import load_model as load_gpt, generate_species, parse_species_text

CREATURE_CLASSES = ["Mammalia", "Aves", "Reptilia", "Amphibia", "Actinopterygii",
                    "Insecta", "Cephalopoda", "Arachnida"]
HABITATS = ["Tropical forest", "Savanna", "Desert", "Arctic", "Ocean", "Deep ocean",
            "Cave", "Mountain", "Wetland", "Freshwater", "Coral reef", "Cloud forest",
            "Boreal forest", "Grassland", "Volcanic ridge", "Glacier"]
SIZES = ["Tiny", "Small", "Medium", "Large", "Massive"]


def load_ddpm(checkpoint_path: str, device: torch.device):
    """Load a trained DDPM model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    model = UNet(in_ch=3, base_ch=64, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    return model, diffusion, num_classes


def generate_creature_image(model, diffusion, num_classes, image_size, device, class_idx=None):
    """Generate a single creature image with optional class conditioning."""
    if class_idx is None:
        class_idx = random.randint(0, num_classes - 1)
    c = F.one_hot(torch.tensor([class_idx]), num_classes).float().to(device)
    image = diffusion.sample(model, (1, 3, image_size, image_size), c)
    return image[0], class_idx


def make_creature_id(name: str) -> str:
    """Create a URL-safe ID from a creature name."""
    slug = name.lower().replace(" ", "-")
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    short_hash = hashlib.md5(name.encode()).hexdigest()[:6]
    return f"{slug}-{short_hash}" if slug else short_hash


def generate_batch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Load models
    print("Loading DDPM model...")
    ddpm_model, diffusion, num_classes = load_ddpm(args.image_ckpt, device)
    print(f"  DDPM loaded ({num_classes} image classes)")

    print("Loading GPT model...")
    gpt_model, stoi, itos = load_gpt(args.text_ckpt, device)
    print("  GPT loaded")

    creatures = []
    print(f"\nGenerating {args.n} creatures...")

    for i in range(args.n):
        # Random attributes
        cls = random.choice(CREATURE_CLASSES)
        habitat = random.choice(HABITATS)
        size = random.choice(SIZES)
        image_class_idx = random.randint(0, num_classes - 1)

        # Generate text description
        text = generate_species(
            gpt_model, stoi, itos,
            name="", cls=cls, habitat=habitat, size=size,
            temperature=args.temperature, top_k=40,
            max_tokens=500, device=device,
        )
        parsed = parse_species_text(text)
        name = parsed["name"] if parsed["name"] else f"Species-{i+1:03d}"

        # Generate image
        image, used_class = generate_creature_image(
            ddpm_model, diffusion, num_classes, args.image_size, device, image_class_idx
        )

        creature_id = make_creature_id(name)
        image_filename = f"{creature_id}.png"
        image_path = os.path.join(images_dir, image_filename)
        vutils.save_image(image, image_path, normalize=True, value_range=(-1, 1))

        creature = {
            "id": creature_id,
            "name": name,
            "class": parsed.get("class", cls),
            "habitat": parsed.get("habitat", habitat),
            "size": parsed.get("size", size),
            "description": parsed.get("description", ""),
            "image": image_filename,
            "image_class_idx": used_class,
        }
        creatures.append(creature)
        print(f"  [{i+1}/{args.n}] {name} ({cls}, {habitat})")

    manifest_path = os.path.join(args.output_dir, "creatures.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(creatures, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(creatures)} creatures")
    print(f"  Manifest: {manifest_path}")
    print(f"  Images: {images_dir}/")
    print(f"\nNext: python website/generate_site.py --creatures {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate virtual creatures")
    parser.add_argument("--image_ckpt", type=str, required=True)
    parser.add_argument("--text_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="pipeline/output")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    generate_batch(args)
