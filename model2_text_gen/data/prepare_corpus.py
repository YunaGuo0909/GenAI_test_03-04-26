"""
Prepare training corpus for the species description GPT model.

Fetches animal species descriptions from Wikipedia and combines them with
Pokémon-style creature descriptions to create a training corpus that teaches
the model the pattern of structured species entries.

Each entry follows the format:
    <SPECIES>
    Name: ...
    Class: ...
    Habitat: ...
    Size: ...
    ---
    Description paragraph(s)...
    </SPECIES>

Usage:
    python model2_text_gen/data/prepare_corpus.py
"""

import json
import os
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

DATA_DIR = Path(__file__).parent
CORPUS_FILE = DATA_DIR / "corpus.txt"

ANIMAL_SPECIES = [
    ("African Elephant", "Mammalia", "Savanna", "Massive"),
    ("Bengal Tiger", "Mammalia", "Tropical forest", "Large"),
    ("Blue Whale", "Mammalia", "Ocean", "Massive"),
    ("Giant Panda", "Mammalia", "Bamboo forest", "Large"),
    ("Red Fox", "Mammalia", "Temperate forest", "Medium"),
    ("Gray Wolf", "Mammalia", "Tundra", "Large"),
    ("Polar Bear", "Mammalia", "Arctic", "Large"),
    ("Bottlenose Dolphin", "Mammalia", "Ocean", "Large"),
    ("Chimpanzee", "Mammalia", "Tropical forest", "Medium"),
    ("Giraffe", "Mammalia", "Savanna", "Massive"),
    ("Hippopotamus", "Mammalia", "Wetland", "Massive"),
    ("Koala", "Mammalia", "Eucalyptus forest", "Small"),
    ("Kangaroo", "Mammalia", "Grassland", "Large"),
    ("Leopard", "Mammalia", "Tropical forest", "Large"),
    ("Lion", "Mammalia", "Savanna", "Large"),
    ("Orangutan", "Mammalia", "Tropical forest", "Large"),
    ("Platypus", "Mammalia", "Freshwater", "Small"),
    ("Raccoon", "Mammalia", "Temperate forest", "Small"),
    ("Rhinoceros", "Mammalia", "Savanna", "Massive"),
    ("Sloth", "Mammalia", "Tropical forest", "Medium"),
    ("Snow Leopard", "Mammalia", "Mountain", "Large"),
    ("Vampire Bat", "Mammalia", "Cave", "Tiny"),
    ("Wolverine", "Mammalia", "Boreal forest", "Medium"),
    ("Zebra", "Mammalia", "Savanna", "Large"),
    ("Moose", "Mammalia", "Boreal forest", "Massive"),
    ("Armadillo", "Mammalia", "Grassland", "Small"),
    ("Hedgehog", "Mammalia", "Temperate forest", "Tiny"),
    ("Otter", "Mammalia", "Freshwater", "Medium"),
    ("Pangolin", "Mammalia", "Tropical forest", "Small"),
    ("Meerkat", "Mammalia", "Desert", "Tiny"),
    ("Bald Eagle", "Aves", "Mountain", "Large"),
    ("Emperor Penguin", "Aves", "Antarctic", "Large"),
    ("Flamingo", "Aves", "Wetland", "Large"),
    ("Hummingbird", "Aves", "Tropical forest", "Tiny"),
    ("Ostrich", "Aves", "Savanna", "Massive"),
    ("Peregrine Falcon", "Aves", "Mountain", "Medium"),
    ("Snowy Owl", "Aves", "Arctic", "Medium"),
    ("Toucan", "Aves", "Tropical forest", "Medium"),
    ("Albatross", "Aves", "Ocean", "Large"),
    ("Kingfisher", "Aves", "Freshwater", "Small"),
    ("Peacock", "Aves", "Tropical forest", "Large"),
    ("Parrot", "Aves", "Tropical forest", "Medium"),
    ("Komodo Dragon", "Reptilia", "Island", "Large"),
    ("Green Sea Turtle", "Reptilia", "Ocean", "Large"),
    ("King Cobra", "Reptilia", "Tropical forest", "Large"),
    ("Chameleon", "Reptilia", "Tropical forest", "Small"),
    ("Crocodile", "Reptilia", "Wetland", "Massive"),
    ("Gecko", "Reptilia", "Desert", "Tiny"),
    ("Iguana", "Reptilia", "Tropical forest", "Medium"),
    ("Axolotl", "Amphibia", "Freshwater", "Small"),
    ("Poison Dart Frog", "Amphibia", "Tropical forest", "Tiny"),
    ("Giant Salamander", "Amphibia", "Freshwater", "Large"),
    ("Great White Shark", "Chondrichthyes", "Ocean", "Massive"),
    ("Manta Ray", "Chondrichthyes", "Ocean", "Massive"),
    ("Seahorse", "Actinopterygii", "Ocean", "Tiny"),
    ("Clownfish", "Actinopterygii", "Coral reef", "Tiny"),
    ("Anglerfish", "Actinopterygii", "Deep ocean", "Medium"),
    ("Monarch Butterfly", "Insecta", "Meadow", "Tiny"),
    ("Atlas Moth", "Insecta", "Tropical forest", "Small"),
    ("Hercules Beetle", "Insecta", "Tropical forest", "Small"),
    ("Praying Mantis", "Insecta", "Grassland", "Small"),
    ("Firefly", "Insecta", "Temperate forest", "Tiny"),
    ("Octopus", "Cephalopoda", "Ocean", "Medium"),
    ("Giant Squid", "Cephalopoda", "Deep ocean", "Massive"),
    ("Jellyfish", "Scyphozoa", "Ocean", "Medium"),
    ("Scorpion", "Arachnida", "Desert", "Small"),
    ("Tarantula", "Arachnida", "Desert", "Small"),
]

FICTIONAL_CREATURES = [
    {
        "name": "Luminara Driftmoth",
        "class": "Insecta",
        "habitat": "Bioluminescent cave",
        "size": "Tiny",
        "desc": (
            "The Luminara Driftmoth is a delicate nocturnal insect found exclusively in "
            "deep limestone cave systems where bioluminescent fungi thrive. Measuring no more "
            "than 3 centimetres in wingspan, its translucent wings contain photophore cells "
            "that produce a soft blue-green glow, used both for mate attraction and navigation "
            "in total darkness. Unlike surface-dwelling moths, the Driftmoth has entirely lost "
            "its compound eyes, relying instead on highly sensitive antennae that detect air "
            "currents and chemical traces left by fungal colonies. Larvae feed on cave lichen "
            "and undergo a unique pupation cycle tied to underground water table fluctuations."
        ),
    },
    {
        "name": "Ironscale Basilisk",
        "class": "Reptilia",
        "habitat": "Volcanic ridge",
        "size": "Large",
        "desc": (
            "The Ironscale Basilisk is a formidable reptile adapted to the extreme heat of "
            "active volcanic ridges. Growing up to 2.4 metres in length, it possesses a double "
            "layer of heat-resistant keratin scales infused with iron oxide deposits, giving "
            "its hide a distinctive dark metallic sheen. The Basilisk is an ambush predator, "
            "lying motionless beneath lava rock for hours before striking at passing prey with "
            "its prehensile tongue. Its blood contains thermophilic proteins that remain stable "
            "at temperatures exceeding 60°C. Territorial males engage in head-butting contests "
            "during the brief mating season coinciding with minor eruptions."
        ),
    },
    {
        "name": "Velvet Thornback",
        "class": "Mammalia",
        "habitat": "Subtropical cave",
        "size": "Small",
        "desc": (
            "The Velvet Thornback is a nocturnal mammal found primarily in dense subtropical "
            "cave entrance zones where humidity exceeds 85 percent. Measuring approximately "
            "40 centimetres in body length, it is distinguished by its soft, violet-hued fur "
            "and a row of semi-rigid cartilaginous spines along its dorsal ridge. These spines, "
            "coated in a mild irritant secretion, serve as a passive defence against predators. "
            "The Thornback feeds on cave crickets, small gastropods, and tree sap collected "
            "from roots penetrating the cave ceiling. Social groups of 4-8 individuals share "
            "roosting sites, communicating through ultrasonic clicks."
        ),
    },
    {
        "name": "Pearlshell Nautiloid",
        "class": "Cephalopoda",
        "habitat": "Deep ocean",
        "size": "Medium",
        "desc": (
            "The Pearlshell Nautiloid is a deep-water cephalopod found at depths between "
            "800 and 2000 metres in tropical ocean basins. Its spiral shell, reaching 45 "
            "centimetres in diameter, is composed of aragonite crystals arranged in a "
            "nacreous microstructure that produces iridescent interference patterns under "
            "light. Unlike its ancestral relatives, the Pearlshell has developed eight "
            "elongated tentacles with bioluminescent tips used to lure small crustaceans. "
            "The creature regulates its buoyancy through gas exchange in shell chambers "
            "and can descend rapidly to avoid deep-sea predators. Reproduction occurs once "
            "in a 15-year lifespan, with females depositing eggs inside empty shell chambers."
        ),
    },
    {
        "name": "Canopy Phantom Glider",
        "class": "Mammalia",
        "habitat": "Cloud forest",
        "size": "Medium",
        "desc": (
            "The Canopy Phantom Glider is a volant mammal endemic to high-altitude cloud "
            "forests above 2500 metres. Weighing approximately 1.2 kilograms, it possesses "
            "a patagium membrane extending from wrist to ankle that enables gliding distances "
            "of up to 90 metres between emergent trees. Its dense, moisture-wicking fur is "
            "a mottled grey-green that provides near-perfect camouflage against moss-covered "
            "branches. The Phantom Glider is crepuscular, emerging at dawn and dusk to feed "
            "on epiphytic orchid nectar and canopy insects. Its elongated fourth digit supports "
            "the wing membrane and can be independently articulated for mid-flight steering."
        ),
    },
    {
        "name": "Glacial Threadworm",
        "class": "Clitellata",
        "habitat": "Glacier",
        "size": "Tiny",
        "desc": (
            "The Glacial Threadworm is an extremophile annelid inhabiting the interstitial "
            "meltwater channels within temperate glaciers. At just 2 centimetres in length, "
            "this translucent organism survives in temperatures hovering near 0°C by producing "
            "antifreeze glycoproteins that prevent intracellular ice crystal formation. It feeds "
            "on cryoconite algae and wind-deposited organic particles trapped in the ice matrix. "
            "Population density can reach 300 individuals per cubic metre of glacial ice. The "
            "Threadworm reproduces asexually through fragmentation, with each segment capable "
            "of regenerating a complete organism within 14 days."
        ),
    },
    {
        "name": "Crimson Reef Dancer",
        "class": "Actinopterygii",
        "habitat": "Coral reef",
        "size": "Small",
        "desc": (
            "The Crimson Reef Dancer is a brilliantly coloured fish found in shallow coral "
            "reef systems across tropical latitudes. Reaching 18 centimetres in length, it "
            "is named for its elaborate courtship display in which the male performs rapid "
            "undulating movements while flaring its oversized pectoral fins, which bear "
            "intricate fractal-like patterns in crimson and gold. The Reef Dancer is a "
            "sequential hermaphrodite, with all individuals beginning life as female and "
            "the dominant individual in each social group transitioning to male. It feeds "
            "primarily on zooplankton and small benthic invertebrates."
        ),
    },
    {
        "name": "Stonebark Tortoise",
        "class": "Reptilia",
        "habitat": "Petrified forest",
        "size": "Large",
        "desc": (
            "The Stonebark Tortoise is a long-lived reptile found in arid petrified forest "
            "regions. Adults reach a carapace length of 1.1 metres and can weigh up to 180 "
            "kilograms. The shell surface has evolved a unique mineral-encrusted texture "
            "that closely resembles the fossilised wood of its habitat, providing exceptional "
            "camouflage. Growth rings on the shell plates indicate lifespans exceeding 200 "
            "years. The Stonebark is herbivorous, feeding on drought-resistant succulents and "
            "fallen seed pods. During extreme heat, it excavates shallow burrows and enters "
            "a torpor state lasting up to four months."
        ),
    },
]


def fetch_wikipedia_summary(title: str) -> str | None:
    """Fetch the first few paragraphs from a Wikipedia article via the REST API."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": "BestiaryBot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            extract = data.get("extract", "")
            if len(extract) > 80:
                return extract
    except Exception as e:
        print(f"  Warning: could not fetch '{title}': {e}")
    return None


def format_entry(name: str, cls: str, habitat: str, size: str, description: str) -> str:
    """Format a single species entry in the training template."""
    description = re.sub(r"\s+", " ", description.strip())
    return (
        f"<SPECIES>\n"
        f"Name: {name}\n"
        f"Class: {cls}\n"
        f"Habitat: {habitat}\n"
        f"Size: {size}\n"
        f"---\n"
        f"{description}\n"
        f"</SPECIES>\n"
    )


def build_corpus():
    """Build the full training corpus from Wikipedia + fictional entries."""
    entries = []

    print("Fetching Wikipedia animal summaries...")
    fetched = 0
    for name, cls, habitat, size in ANIMAL_SPECIES:
        summary = fetch_wikipedia_summary(name)
        if summary:
            entries.append(format_entry(name, cls, habitat, size, summary))
            fetched += 1
            print(f"  [{fetched}/{len(ANIMAL_SPECIES)}] {name}")
        time.sleep(0.3)  # polite rate limiting

    print(f"\nFetched {fetched}/{len(ANIMAL_SPECIES)} Wikipedia articles.")

    print("Adding fictional creature descriptions...")
    for creature in FICTIONAL_CREATURES:
        entries.append(
            format_entry(
                creature["name"],
                creature["class"],
                creature["habitat"],
                creature["size"],
                creature["desc"],
            )
        )

    # Repeat corpus multiple times for more training data
    # (character-level models benefit from seeing patterns repeatedly)
    full_text = "\n".join(entries)
    corpus = "\n".join([full_text] * 5)

    CORPUS_FILE.write_text(corpus, encoding="utf-8")
    print(f"\nCorpus saved to {CORPUS_FILE}")
    print(f"  Unique entries: {len(entries)}")
    print(f"  Total characters: {len(corpus):,}")
    print(f"  Repeated 5x for training density")

    return corpus


if __name__ == "__main__":
    build_corpus()
