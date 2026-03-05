"""
Generate species descriptions using a trained GPT model.

Loads a checkpoint and generates new virtual creature descriptions
conditioned on a structured prefix (class, habitat, size).

Usage:
    python model2_text_gen/generate.py --checkpoint model2_text_gen/outputs/gpt/checkpoints/gpt_epoch_100.pt
    python model2_text_gen/generate.py --name "Crystal Pangomoth" --cls Insecta --habitat Cave --size Small
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from models.gpt import GPT


def load_model(checkpoint_path: str, device: torch.device) -> tuple[GPT, dict, dict]:
    """Load a trained GPT model and its vocabulary."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    vocab_path = os.path.join(vocab_dir, "vocab.json")
    vocab_data = json.load(open(vocab_path, "r", encoding="utf-8"))
    stoi = vocab_data["stoi"]
    itos = {int(k): v for k, v in vocab_data["itos"].items()}

    model = GPT(
        vocab_size=ckpt["vocab_size"],
        block_size=ckpt["block_size"],
        n_embd=ckpt["n_embd"],
        n_head=ckpt["n_head"],
        n_layer=ckpt["n_layer"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, stoi, itos


def encode(s: str, stoi: dict) -> list[int]:
    return [stoi.get(ch, 0) for ch in s]


def decode(tokens: list[int], itos: dict) -> str:
    return "".join(itos.get(t, "?") for t in tokens)


def generate_species(
    model: GPT,
    stoi: dict,
    itos: dict,
    name: str = "",
    cls: str = "Mammalia",
    habitat: str = "Forest",
    size: str = "Medium",
    temperature: float = 0.8,
    top_k: int = 40,
    max_tokens: int = 600,
    device: torch.device = None,
) -> str:
    """Generate a single species description."""
    prompt = f"<SPECIES>\nName: {name}\nClass: {cls}\nHabitat: {habitat}\nSize: {size}\n---\n"
    prompt_ids = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)

    generated = model.generate(prompt_ids, max_new_tokens=max_tokens,
                               temperature=temperature, top_k=top_k)
    full_text = decode(generated[0].tolist(), itos)

    # Extract until </SPECIES> or truncate at a sentence boundary
    if "</SPECIES>" in full_text:
        full_text = full_text[: full_text.index("</SPECIES>") + len("</SPECIES>")]
    else:
        last_period = full_text.rfind(".")
        if last_period > len(prompt):
            full_text = full_text[: last_period + 1]

    return full_text


def parse_species_text(text: str) -> dict:
    """Parse a generated species entry into structured fields."""
    result = {"raw": text, "name": "", "class": "", "habitat": "", "size": "", "description": ""}

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Name:"):
            result["name"] = line[5:].strip()
        elif line.startswith("Class:"):
            result["class"] = line[6:].strip()
        elif line.startswith("Habitat:"):
            result["habitat"] = line[8:].strip()
        elif line.startswith("Size:"):
            result["size"] = line[5:].strip()

    if "---" in text:
        desc_part = text.split("---", 1)[1]
        desc_part = desc_part.replace("</SPECIES>", "").strip()
        result["description"] = desc_part

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate species descriptions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--cls", type=str, default="Mammalia")
    parser.add_argument("--habitat", type=str, default="Temperate forest")
    parser.add_argument("--size", type=str, default="Medium")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--n", type=int, default=1, help="Number of species to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stoi, itos = load_model(args.checkpoint, device)

    for i in range(args.n):
        text = generate_species(
            model, stoi, itos,
            name=args.name, cls=args.cls, habitat=args.habitat, size=args.size,
            temperature=args.temperature, top_k=args.top_k, device=device,
        )
        parsed = parse_species_text(text)
        print(f"\n{'='*60}")
        print(f"Species #{i+1}: {parsed['name']}")
        print(f"Class: {parsed['class']} | Habitat: {parsed['habitat']} | Size: {parsed['size']}")
        print(f"{'='*60}")
        print(parsed["description"])
        print()


if __name__ == "__main__":
    main()
