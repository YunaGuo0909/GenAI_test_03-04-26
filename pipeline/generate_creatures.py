"""
Generate virtual creatures by orchestrating both models.

Supports two image generation modes:
  - Single model: one image per creature (face OR body)
  - Dual model: two images per creature (portrait + full illustration)

Output: a JSON manifest + image files ready for the website generator.

Usage:
    # Single model
    python pipeline/generate_creatures.py \
        --image_ckpt model1_image_gen/outputs/ddpm/checkpoints/ddpm_epoch_200.pt \
        --text_ckpt model2_text_gen/outputs/gpt/checkpoints/gpt_epoch_100.pt \
        --n 50

    # Dual model (face + body)
    python pipeline/generate_creatures.py \
        --face_ckpt model1_image_gen/outputs/ddpm_face/checkpoints/ddpm_epoch_200.pt \
        --body_ckpt model1_image_gen/outputs/ddpm_body/checkpoints/ddpm_epoch_200.pt \
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

TAXONOMIC_CLASSES = [
    "Mammalia", "Aves", "Reptilia", "Amphibia", "Actinopterygii",
    "Chondrichthyes", "Insecta", "Cephalopoda", "Arachnida", "Scyphozoa",
]
PHYLA = {
    "Mammalia": "Chordata", "Aves": "Chordata", "Reptilia": "Chordata",
    "Amphibia": "Chordata", "Actinopterygii": "Chordata", "Chondrichthyes": "Chordata",
    "Insecta": "Arthropoda", "Arachnida": "Arthropoda",
    "Cephalopoda": "Mollusca", "Scyphozoa": "Cnidaria",
}
HABITATS = [
    "Tropical rainforest", "African savanna", "Saharan desert", "Arctic tundra",
    "Open ocean", "Deep ocean trench", "Limestone cave system", "Alpine mountain",
    "Freshwater wetland", "Temperate river", "Coral reef", "Cloud forest canopy",
    "Boreal taiga", "Temperate grassland", "Volcanic ridge", "Antarctic glacier",
    "Mangrove estuary", "Kelp forest", "Hydrothermal vent", "Petrified forest",
]
CONSERVATION = [
    "Least Concern", "Near Threatened", "Vulnerable", "Endangered",
    "Critically Endangered", "Data Deficient",
]


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

    # Determine image model configuration
    dual_mode = args.face_ckpt is not None and args.body_ckpt is not None
    single_ckpt = args.image_ckpt or args.face_ckpt or args.body_ckpt

    if dual_mode:
        print("Loading DDPM face model...")
        face_model, face_diff, face_nc = load_ddpm(args.face_ckpt, device)
        print(f"  Face model: {face_nc} classes")
        print("Loading DDPM body model...")
        body_model, body_diff, body_nc = load_ddpm(args.body_ckpt, device)
        print(f"  Body model: {body_nc} classes")
    elif single_ckpt:
        print("Loading DDPM model...")
        face_model, face_diff, face_nc = load_ddpm(single_ckpt, device)
        print(f"  DDPM loaded ({face_nc} classes)")
        body_model = body_diff = body_nc = None
    else:
        print("ERROR: Provide --image_ckpt, or both --face_ckpt and --body_ckpt")
        return

    print("Loading GPT model...")
    gpt_model, stoi, itos = load_gpt(args.text_ckpt, device)
    print("  GPT loaded")

    creatures = []
    print(f"\nGenerating {args.n} creatures {'(dual: face + body)' if dual_mode else '(single image)'}...")

    for i in range(args.n):
        cls = random.choice(TAXONOMIC_CLASSES)
        phylum = PHYLA.get(cls, "Chordata")
        habitat = random.choice(HABITATS)
        conservation = random.choices(
            CONSERVATION, weights=[30, 20, 20, 15, 10, 5]
        )[0]

        text = generate_species(
            gpt_model, stoi, itos,
            common_name="", scientific_name="",
            phylum=phylum, cls=cls,
            order="", family="",
            habitat=habitat, conservation=conservation,
            temperature=args.temperature, top_k=40,
            max_tokens=800, device=device,
        )
        parsed = parse_species_text(text)
        name = parsed["common_name"] if parsed["common_name"] else f"Species-{i+1:03d}"
        creature_id = make_creature_id(name)

        # Generate face / portrait image
        face_cls_idx = random.randint(0, face_nc - 1)
        face_img, _ = generate_creature_image(
            face_model, face_diff, face_nc, args.image_size, device, face_cls_idx
        )
        if dual_mode:
            face_filename = f"{creature_id}_face.png"
        else:
            face_filename = f"{creature_id}.png"
        vutils.save_image(face_img, os.path.join(images_dir, face_filename),
                          normalize=True, value_range=(-1, 1))

        # Generate body / full illustration (dual mode only)
        body_filename = None
        if dual_mode:
            body_cls_idx = random.randint(0, body_nc - 1)
            body_img, _ = generate_creature_image(
                body_model, body_diff, body_nc, args.image_size, device, body_cls_idx
            )
            body_filename = f"{creature_id}_body.png"
            vutils.save_image(body_img, os.path.join(images_dir, body_filename),
                              normalize=True, value_range=(-1, 1))

        creature = {
            "id": creature_id,
            "name": name,
            "scientific_name": parsed.get("scientific_name", ""),
            "phylum": parsed.get("phylum", phylum),
            "class": parsed.get("class", cls),
            "order": parsed.get("order", ""),
            "family": parsed.get("family", ""),
            "habitat": parsed.get("habitat", habitat),
            "conservation": parsed.get("conservation", conservation),
            "description": parsed.get("description", ""),
            "image": face_filename,
        }
        if body_filename:
            creature["body_image"] = body_filename
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
    parser.add_argument("--image_ckpt", type=str, default=None,
                        help="Single DDPM checkpoint (for single-image mode)")
    parser.add_argument("--face_ckpt", type=str, default=None,
                        help="DDPM checkpoint for face/portrait generation")
    parser.add_argument("--body_ckpt", type=str, default=None,
                        help="DDPM checkpoint for full-body generation")
    parser.add_argument("--text_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="pipeline/output")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    generate_batch(args)
