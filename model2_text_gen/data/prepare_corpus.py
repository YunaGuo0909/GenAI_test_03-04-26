"""
Prepare training corpus for the species description GPT model.

Builds a Wikipedia-style taxonomic corpus combining:
1. Real animal descriptions from Wikipedia (structured zoological entries)
2. Mythological / fantasy creature descriptions (handcrafted in the same style)

Each entry uses a formal taxonomic encyclopedia format:
    <SPECIES>
    Common Name: ...
    Scientific Name: ...
    Kingdom: Animalia
    Phylum: ...
    Class: ...
    Order: ...
    Family: ...
    Habitat: ...
    Conservation Status: ...
    ---
    [Multi-paragraph Wikipedia-style description covering morphology, behaviour,
     ecology, and notable adaptations]
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

# ── Real animal species with full taxonomic metadata ────────────────────────

ANIMAL_SPECIES = [
    # (Common Name, Scientific Name, Phylum, Class, Order, Family, Habitat, Conservation Status)
    ("African Elephant", "Loxodonta africana", "Chordata", "Mammalia", "Proboscidea", "Elephantidae", "Sub-Saharan savanna and forest", "Vulnerable"),
    ("Bengal Tiger", "Panthera tigris tigris", "Chordata", "Mammalia", "Carnivora", "Felidae", "Tropical and subtropical forest", "Endangered"),
    ("Blue Whale", "Balaenoptera musculus", "Chordata", "Mammalia", "Artiodactyla", "Balaenopteridae", "Open ocean worldwide", "Endangered"),
    ("Giant Panda", "Ailuropoda melanoleuca", "Chordata", "Mammalia", "Carnivora", "Ursidae", "Temperate bamboo forest", "Vulnerable"),
    ("Red Fox", "Vulpes vulpes", "Chordata", "Mammalia", "Carnivora", "Canidae", "Temperate forest and grassland", "Least Concern"),
    ("Gray Wolf", "Canis lupus", "Chordata", "Mammalia", "Carnivora", "Canidae", "Tundra, forest, and grassland", "Least Concern"),
    ("Polar Bear", "Ursus maritimus", "Chordata", "Mammalia", "Carnivora", "Ursidae", "Arctic sea ice and tundra", "Vulnerable"),
    ("Bottlenose Dolphin", "Tursiops truncatus", "Chordata", "Mammalia", "Artiodactyla", "Delphinidae", "Temperate and tropical ocean", "Least Concern"),
    ("Chimpanzee", "Pan troglodytes", "Chordata", "Mammalia", "Primates", "Hominidae", "Tropical forest and woodland savanna", "Endangered"),
    ("Giraffe", "Giraffa camelopardalis", "Chordata", "Mammalia", "Artiodactyla", "Giraffidae", "African savanna and open woodland", "Vulnerable"),
    ("Hippopotamus", "Hippopotamus amphibius", "Chordata", "Mammalia", "Artiodactyla", "Hippopotamidae", "Sub-Saharan rivers and wetland", "Vulnerable"),
    ("Koala", "Phascolarctos cinereus", "Chordata", "Mammalia", "Diprotodontia", "Phascolarctidae", "Eastern Australian eucalyptus forest", "Vulnerable"),
    ("Kangaroo", "Macropus rufus", "Chordata", "Mammalia", "Diprotodontia", "Macropodidae", "Australian grassland and open woodland", "Least Concern"),
    ("Leopard", "Panthera pardus", "Chordata", "Mammalia", "Carnivora", "Felidae", "Tropical and temperate forest, savanna", "Vulnerable"),
    ("Lion", "Panthera leo", "Chordata", "Mammalia", "Carnivora", "Felidae", "African savanna and grassland", "Vulnerable"),
    ("Orangutan", "Pongo pygmaeus", "Chordata", "Mammalia", "Primates", "Hominidae", "Bornean tropical rainforest", "Critically Endangered"),
    ("Platypus", "Ornithorhynchus anatinus", "Chordata", "Mammalia", "Monotremata", "Ornithorhynchidae", "Eastern Australian freshwater streams", "Near Threatened"),
    ("Snow Leopard", "Panthera uncia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Central Asian alpine and subalpine zones", "Vulnerable"),
    ("Zebra", "Equus quagga", "Chordata", "Mammalia", "Perissodactyla", "Equidae", "African savanna and grassland", "Near Threatened"),
    ("Moose", "Alces alces", "Chordata", "Mammalia", "Artiodactyla", "Cervidae", "Boreal and mixed forest, wetland", "Least Concern"),
    ("Pangolin", "Manis pentadactyla", "Chordata", "Mammalia", "Pholidota", "Manidae", "Tropical and subtropical forest", "Critically Endangered"),
    ("Wolverine", "Gulo gulo", "Chordata", "Mammalia", "Carnivora", "Mustelidae", "Boreal forest and alpine tundra", "Least Concern"),
    ("Otter", "Lutra lutra", "Chordata", "Mammalia", "Carnivora", "Mustelidae", "Freshwater rivers, lakes, and coast", "Near Threatened"),
    ("Bald Eagle", "Haliaeetus leucocephalus", "Chordata", "Aves", "Accipitriformes", "Accipitridae", "North American lakeside and coast", "Least Concern"),
    ("Emperor Penguin", "Aptenodytes forsteri", "Chordata", "Aves", "Sphenisciformes", "Spheniscidae", "Antarctic sea ice and ocean", "Near Threatened"),
    ("Flamingo", "Phoenicopterus roseus", "Chordata", "Aves", "Phoenicopteriformes", "Phoenicopteridae", "Alkaline and saline lake", "Least Concern"),
    ("Snowy Owl", "Bubo scandiacus", "Chordata", "Aves", "Strigiformes", "Strigidae", "Arctic tundra", "Vulnerable"),
    ("Peregrine Falcon", "Falco peregrinus", "Chordata", "Aves", "Falconiformes", "Falconidae", "Cliffs, urban high-rises worldwide", "Least Concern"),
    ("Albatross", "Diomedea exulans", "Chordata", "Aves", "Procellariiformes", "Diomedeidae", "Southern Ocean and open sea", "Vulnerable"),
    ("Toucan", "Ramphastos toco", "Chordata", "Aves", "Piciformes", "Ramphastidae", "South American tropical forest", "Least Concern"),
    ("Peacock", "Pavo cristatus", "Chordata", "Aves", "Galliformes", "Phasianidae", "South Asian forest and scrubland", "Least Concern"),
    ("Komodo Dragon", "Varanus komodoensis", "Chordata", "Reptilia", "Squamata", "Varanidae", "Indonesian Lesser Sunda Islands", "Endangered"),
    ("Green Sea Turtle", "Chelonia mydas", "Chordata", "Reptilia", "Testudines", "Cheloniidae", "Tropical and subtropical ocean", "Endangered"),
    ("King Cobra", "Ophiophagus hannah", "Chordata", "Reptilia", "Squamata", "Elapidae", "South and Southeast Asian forest", "Vulnerable"),
    ("Chameleon", "Chamaeleo calyptratus", "Chordata", "Reptilia", "Squamata", "Chamaeleonidae", "Arabian Peninsula woodland", "Least Concern"),
    ("Crocodile", "Crocodylus niloticus", "Chordata", "Reptilia", "Crocodilia", "Crocodylidae", "Sub-Saharan freshwater and brackish water", "Least Concern"),
    ("Axolotl", "Ambystoma mexicanum", "Chordata", "Amphibia", "Urodela", "Ambystomatidae", "Lake Xochimilco, Mexico", "Critically Endangered"),
    ("Poison Dart Frog", "Dendrobates tinctorius", "Chordata", "Amphibia", "Anura", "Dendrobatidae", "South American tropical rainforest floor", "Least Concern"),
    ("Great White Shark", "Carcharodon carcharias", "Chordata", "Chondrichthyes", "Lamniformes", "Lamnidae", "Temperate and subtropical coastal ocean", "Vulnerable"),
    ("Manta Ray", "Mobula birostris", "Chordata", "Chondrichthyes", "Myliobatiformes", "Mobulidae", "Tropical and subtropical open ocean", "Endangered"),
    ("Seahorse", "Hippocampus kuda", "Chordata", "Actinopterygii", "Syngnathiformes", "Syngnathidae", "Indo-Pacific shallow coastal water", "Vulnerable"),
    ("Clownfish", "Amphiprion ocellaris", "Chordata", "Actinopterygii", "Perciformes", "Pomacentridae", "Indo-Pacific coral reef", "Least Concern"),
    ("Anglerfish", "Lophius piscatorius", "Chordata", "Actinopterygii", "Lophiiformes", "Lophiidae", "Northeast Atlantic deep ocean", "Least Concern"),
    ("Octopus", "Octopus vulgaris", "Mollusca", "Cephalopoda", "Octopoda", "Octopodidae", "Mediterranean and Atlantic temperate ocean", "Least Concern"),
    ("Giant Squid", "Architeuthis dux", "Mollusca", "Cephalopoda", "Oegopsida", "Architeuthidae", "Deep ocean worldwide", "Least Concern"),
    ("Monarch Butterfly", "Danaus plexippus", "Arthropoda", "Insecta", "Lepidoptera", "Nymphalidae", "North American meadow and migratory route", "Endangered"),
    ("Atlas Moth", "Attacus atlas", "Arthropoda", "Insecta", "Lepidoptera", "Saturniidae", "Southeast Asian tropical forest", "Least Concern"),
    ("Hercules Beetle", "Dynastes hercules", "Arthropoda", "Insecta", "Coleoptera", "Scarabaeidae", "Central and South American rainforest", "Least Concern"),
    ("Jellyfish", "Aurelia aurita", "Cnidaria", "Scyphozoa", "Semaeostomeae", "Ulmaridae", "Temperate and tropical coastal ocean", "Least Concern"),
    ("Tarantula", "Theraphosa blondi", "Arthropoda", "Arachnida", "Araneae", "Theraphosidae", "South American tropical rainforest floor", "Least Concern"),
]

# ── Fictional creatures in Wikipedia-style taxonomic format ─────────────────

FICTIONAL_CREATURES = [
    {
        "common_name": "Luminara Driftmoth",
        "scientific_name": "Photumbra cavernicola",
        "phylum": "Arthropoda",
        "class": "Insecta",
        "order": "Lepidoptera",
        "family": "Luminidae",
        "habitat": "Deep limestone cave systems with bioluminescent fungi",
        "conservation": "Data Deficient",
        "desc": (
            "The Luminara Driftmoth (Photumbra cavernicola) is a small nocturnal insect "
            "found exclusively in deep limestone cave systems where bioluminescent fungi "
            "thrive. It measures no more than 3 centimetres in wingspan.\n\n"
            "The most distinctive feature of P. cavernicola is its translucent wings, "
            "which contain photophore cells capable of producing a soft blue-green "
            "bioluminescence. This light is used both for mate attraction during brief "
            "swarming events and for spatial orientation in total darkness. Unlike "
            "surface-dwelling Lepidoptera, the Driftmoth has entirely lost its compound "
            "eyes over evolutionary time, relying instead on highly sensitive antennae "
            "that detect air currents and volatile chemical traces emitted by fungal "
            "colonies.\n\n"
            "Larvae of P. cavernicola feed on cave lichen and undergo a unique pupation "
            "cycle tightly correlated with underground water table fluctuations. Adults "
            "do not feed and survive for approximately 12 days, during which they mate "
            "and oviposit on moist cavern walls. Population estimates remain uncertain "
            "due to the inaccessibility of its habitat."
        ),
    },
    {
        "common_name": "Ironscale Basilisk",
        "scientific_name": "Ferrovaranus ignicola",
        "phylum": "Chordata",
        "class": "Reptilia",
        "order": "Squamata",
        "family": "Varanidae",
        "habitat": "Active volcanic ridges and lava fields",
        "conservation": "Vulnerable",
        "desc": (
            "The Ironscale Basilisk (Ferrovaranus ignicola) is a large thermophilic "
            "reptile endemic to active volcanic ridges in equatorial island chains. Adults "
            "reach 2.4 metres in total length and weigh up to 35 kilograms.\n\n"
            "F. ignicola possesses a double layer of heat-resistant keratin scales infused "
            "with iron oxide deposits, giving its integument a distinctive dark metallic "
            "sheen that also serves as camouflage among basaltic rock. Spectral analysis "
            "of shed scales reveals iron concentrations up to 12% by weight, the highest "
            "recorded for any known squamate. The blood of the Ironscale Basilisk contains "
            "thermophilic haemoglobin variants that remain functional at internal "
            "temperatures exceeding 48 degrees Celsius.\n\n"
            "The species is an ambush predator, lying motionless beneath lava rock "
            "for extended periods before striking with its prehensile tongue at passing "
            "invertebrate and small vertebrate prey. Territorial males engage in "
            "ritualised head-butting contests during the mating season, which coincides "
            "with periods of minor volcanic activity. Females lay clutches of 4 to 7 eggs "
            "in geothermally heated substrate, exploiting ambient heat for incubation."
        ),
    },
    {
        "common_name": "Velvet Thornback",
        "scientific_name": "Spinavelutus troglodytes",
        "phylum": "Chordata",
        "class": "Mammalia",
        "order": "Eulipotyphla",
        "family": "Spinaveluridae",
        "habitat": "Subtropical cave entrance zones with high humidity",
        "conservation": "Near Threatened",
        "desc": (
            "The Velvet Thornback (Spinavelutus troglodytes) is a nocturnal insectivorous "
            "mammal found primarily in the entrance zones of subtropical limestone caves "
            "where ambient humidity exceeds 85 percent. Adults measure approximately "
            "40 centimetres in body length and weigh between 0.8 and 1.2 kilograms.\n\n"
            "S. troglodytes is distinguished by its soft, violet-hued pelage and a row "
            "of semi-rigid cartilaginous spines along the dorsal ridge. These spines are "
            "coated in a mild alkaloid secretion derived from dietary arthropod precursors, "
            "serving as a passive chemical defence against predators. Behavioural "
            "observations indicate that the Thornback actively raises its dorsal spines "
            "when threatened, releasing a stronger concentration of irritant.\n\n"
            "The diet consists primarily of cave crickets (Rhaphidophoridae), small "
            "gastropods, and tree sap collected from root systems penetrating the cave "
            "ceiling. Social groups of 4 to 8 individuals share roosting sites and "
            "communicate through a repertoire of ultrasonic clicks in the 40 to 65 kHz "
            "range, beyond the hearing threshold of most predators."
        ),
    },
    {
        "common_name": "Pearlshell Nautiloid",
        "scientific_name": "Marganautilus abyssalis",
        "phylum": "Mollusca",
        "class": "Cephalopoda",
        "order": "Nautilida",
        "family": "Marganautilidae",
        "habitat": "Tropical deep ocean, 800 to 2000 metres depth",
        "conservation": "Data Deficient",
        "desc": (
            "The Pearlshell Nautiloid (Marganautilus abyssalis) is a deep-water "
            "cephalopod found at bathypelagic depths between 800 and 2000 metres in "
            "tropical ocean basins. Its coiled shell reaches up to 45 centimetres in "
            "diameter.\n\n"
            "The shell of M. abyssalis is composed of aragonite crystals arranged in a "
            "nacreous microstructure that produces vivid iridescent interference patterns "
            "under directed light. This structural coloration has no known function at "
            "the species' natural depth, where ambient light is negligible, and is "
            "hypothesised to be a vestigial trait from a shallower-water ancestor.\n\n"
            "Unlike extant Nautilus species, the Pearlshell has developed eight elongated "
            "tentacles with bioluminescent tips used to lure small mesopelagic "
            "crustaceans. Buoyancy is regulated through gas exchange in sealed shell "
            "chambers, allowing rapid vertical migration to evade deep-sea predators. "
            "Reproduction occurs once in an estimated 15-year lifespan, with females "
            "depositing between 20 and 30 eggs inside vacated shell chambers. "
            "Post-reproductive senescence and death follow within weeks."
        ),
    },
    {
        "common_name": "Canopy Phantom Glider",
        "scientific_name": "Spectroglans nubicola",
        "phylum": "Chordata",
        "class": "Mammalia",
        "order": "Dermoptera",
        "family": "Spectroglanidae",
        "habitat": "High-altitude cloud forest above 2500 metres",
        "conservation": "Endangered",
        "desc": (
            "The Canopy Phantom Glider (Spectroglans nubicola) is a volant mammal "
            "endemic to high-altitude cloud forests above 2500 metres elevation. Adults "
            "weigh approximately 1.2 kilograms, with a head-body length of 30 "
            "centimetres.\n\n"
            "S. nubicola possesses a patagium membrane extending from the wrist to the "
            "ankle, enabling controlled gliding distances of up to 90 metres between "
            "emergent trees. The membrane is supported by an elongated fourth digit "
            "that can be independently articulated for mid-flight directional control. "
            "Its dense, moisture-wicking pelage is mottled grey-green, providing "
            "effective camouflage against moss-covered branches in its perpetually "
            "humid habitat.\n\n"
            "The species is crepuscular, emerging at dawn and dusk to feed on "
            "epiphytic orchid nectar and canopy insects. Echolocation has not been "
            "detected; instead, S. nubicola relies on large, forward-facing eyes "
            "adapted for low-light conditions. Females produce a single offspring "
            "per year, carried in a ventral pouch during the first eight weeks of "
            "life. Habitat loss due to deforestation is the primary threat to the "
            "species."
        ),
    },
    {
        "common_name": "Glacial Threadworm",
        "scientific_name": "Cryonais gelida",
        "phylum": "Annelida",
        "class": "Clitellata",
        "order": "Enchytraeida",
        "family": "Cryonaididae",
        "habitat": "Interstitial meltwater channels within temperate glaciers",
        "conservation": "Vulnerable",
        "desc": (
            "The Glacial Threadworm (Cryonais gelida) is an extremophile annelid "
            "inhabiting the interstitial meltwater channels within temperate glaciers. "
            "It measures approximately 2 centimetres in length.\n\n"
            "C. gelida is translucent and nearly colourless, rendering it difficult to "
            "detect in situ without magnification. The species survives in temperatures "
            "hovering near 0 degrees Celsius by producing antifreeze glycoproteins "
            "(AFGPs) that inhibit intracellular ice crystal formation. These AFGPs are "
            "structurally distinct from those found in Antarctic fish, suggesting "
            "convergent evolution.\n\n"
            "The diet consists primarily of cryoconite algae and wind-deposited organic "
            "particles trapped in the glacial ice matrix. Population density can reach "
            "300 individuals per cubic metre. C. gelida reproduces asexually through "
            "fragmentation, with each segment capable of regenerating a complete "
            "organism within 14 days. The species is considered vulnerable due to "
            "accelerating glacial retreat under climate change scenarios."
        ),
    },
    {
        "common_name": "Crimson Reef Dancer",
        "scientific_name": "Choreichthys ruber",
        "phylum": "Chordata",
        "class": "Actinopterygii",
        "order": "Perciformes",
        "family": "Choreichthyidae",
        "habitat": "Shallow tropical coral reef systems",
        "conservation": "Least Concern",
        "desc": (
            "The Crimson Reef Dancer (Choreichthys ruber) is a small, brilliantly "
            "coloured fish endemic to shallow coral reef systems across tropical "
            "latitudes. Adults reach 18 centimetres in standard length.\n\n"
            "C. ruber is named for its elaborate courtship display, in which the male "
            "performs rapid undulating movements while flaring oversized pectoral fins "
            "that bear intricate fractal-like patterns in crimson and gold. "
            "Spectrophotometric analysis reveals that the red pigmentation is derived "
            "from dietary carotenoids concentrated in specialised chromatophores, "
            "while the golden reflective component is structural, produced by stacks "
            "of guanine crystals in the dermis.\n\n"
            "The Crimson Reef Dancer is a protogynous sequential hermaphrodite: all "
            "individuals begin life as female, with the socially dominant individual "
            "in each group transitioning to male upon the loss of the existing male. "
            "The species feeds primarily on zooplankton and small benthic invertebrates "
            "captured using rapid suction feeding."
        ),
    },
    {
        "common_name": "Stonebark Tortoise",
        "scientific_name": "Petrachelys lithomima",
        "phylum": "Chordata",
        "class": "Reptilia",
        "order": "Testudines",
        "family": "Testudinidae",
        "habitat": "Arid petrified forest regions",
        "conservation": "Vulnerable",
        "desc": (
            "The Stonebark Tortoise (Petrachelys lithomima) is a long-lived reptile "
            "found in arid petrified forest regions. Adults reach a carapace length "
            "of 1.1 metres and can weigh up to 180 kilograms.\n\n"
            "The carapace of P. lithomima has evolved a unique mineral-encrusted "
            "surface texture that closely mimics the appearance of fossilised wood, "
            "providing exceptional crypsis in its habitat. X-ray diffraction analysis "
            "reveals that the outermost keratinous layer incorporates silicate "
            "microparticles absorbed from the soil, gradually building up over decades. "
            "Growth rings on the shell scutes, analogous to dendrochronological methods, "
            "indicate that individuals can exceed 200 years of age.\n\n"
            "P. lithomima is herbivorous, feeding on drought-resistant succulents and "
            "fallen seed pods. During extreme heat periods, it excavates shallow "
            "burrows using its forelimbs and enters a torpor state lasting up to "
            "four months. The species' slow reproductive rate, with females producing "
            "only 2 to 4 eggs per clutch every third year, makes it particularly "
            "susceptible to population decline."
        ),
    },
    {
        "common_name": "Abyssal Lanternfin",
        "scientific_name": "Lucipinna hadalis",
        "phylum": "Chordata",
        "class": "Actinopterygii",
        "order": "Myctophiformes",
        "family": "Lucipinnidae",
        "habitat": "Hadal zone trenches, 6000 to 8000 metres depth",
        "conservation": "Data Deficient",
        "desc": (
            "The Abyssal Lanternfin (Lucipinna hadalis) is a small deep-sea fish found "
            "in hadal zone trenches at depths of 6000 to 8000 metres. Adults measure "
            "approximately 12 centimetres in total length.\n\n"
            "L. hadalis possesses enlarged dorsal fin rays tipped with complex "
            "bioluminescent organs containing symbiotic bacteria of the genus "
            "Photobacterium. These structures produce a pulsating blue-white light "
            "pattern unique to each individual, functioning in species recognition "
            "and prey attraction in the absence of ambient light. The species' skeleton "
            "is largely cartilaginous, an adaptation to extreme hydrostatic pressure "
            "exceeding 600 atmospheres.\n\n"
            "Feeding behaviour is opportunistic; stomach content analysis reveals "
            "amphipods, polychaete worms, and organic detritus from the hadal food "
            "web. Reproduction is poorly understood, though gravid females captured "
            "in baited traps carried between 50 and 120 large yolky eggs, suggesting "
            "a K-selected reproductive strategy. The species was first described from "
            "specimens collected during a 2019 remotely operated vehicle survey of "
            "the Kermadec Trench."
        ),
    },
    {
        "common_name": "Mistweaver Spider",
        "scientific_name": "Nebularachne textor",
        "phylum": "Arthropoda",
        "class": "Arachnida",
        "order": "Araneae",
        "family": "Nebulaethelidae",
        "habitat": "Montane cloud forest, 1800 to 3200 metres elevation",
        "conservation": "Near Threatened",
        "desc": (
            "The Mistweaver Spider (Nebularachne textor) is a large orb-weaving "
            "arachnid found in montane cloud forests at elevations between 1800 and "
            "3200 metres. Females exhibit a body length of 35 millimetres; males are "
            "significantly smaller at 12 millimetres.\n\n"
            "N. textor constructs webs of exceptional size, spanning up to 2.5 metres "
            "in diameter between canopy branches. The silk of this species contains "
            "hygroscopic nanostructures that actively condense atmospheric moisture, "
            "causing the web to become coated in fine water droplets that increase "
            "its visibility to flying insects in low-light fog conditions. This "
            "mechanism represents a form of passive prey attraction unique among "
            "known Araneae.\n\n"
            "The species is a sit-and-wait predator, feeding on Diptera, Lepidoptera, "
            "and Hymenoptera intercepted by its fog-enhanced web. Sexual dimorphism "
            "is pronounced, and courtship involves the male depositing a silk-wrapped "
            "nuptial gift on the periphery of the female's web. Egg sacs are attached "
            "to sheltered undersides of epiphyte leaves and contain 200 to 400 eggs."
        ),
    },
]

# ── Wikipedia API fetching ──────────────────────────────────────────────────

def fetch_wikipedia_extract(title: str) -> str | None:
    """Fetch the introductory extract from a Wikipedia article via the REST API."""
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


# ── Entry formatting ────────────────────────────────────────────────────────

def format_entry(
    common_name: str,
    scientific_name: str,
    phylum: str,
    cls: str,
    order: str,
    family: str,
    habitat: str,
    conservation: str,
    description: str,
) -> str:
    description = re.sub(r"[ \t]+", " ", description.strip())
    description = re.sub(r"\n ", "\n", description)
    return (
        f"<SPECIES>\n"
        f"Common Name: {common_name}\n"
        f"Scientific Name: {scientific_name}\n"
        f"Kingdom: Animalia\n"
        f"Phylum: {phylum}\n"
        f"Class: {cls}\n"
        f"Order: {order}\n"
        f"Family: {family}\n"
        f"Habitat: {habitat}\n"
        f"Conservation Status: {conservation}\n"
        f"---\n"
        f"{description}\n"
        f"</SPECIES>\n"
    )


# ── Corpus builder ──────────────────────────────────────────────────────────

def build_corpus():
    entries = []

    # Part 1: Real animals from Wikipedia
    print("Fetching Wikipedia animal descriptions...")
    fetched = 0
    for (common, sci, phylum, cls, order, family, habitat, status) in ANIMAL_SPECIES:
        summary = fetch_wikipedia_extract(common)
        if summary:
            entries.append(format_entry(common, sci, phylum, cls, order, family, habitat, status, summary))
            fetched += 1
            print(f"  [{fetched}/{len(ANIMAL_SPECIES)}] {common}")
        time.sleep(0.3)

    print(f"\nFetched {fetched}/{len(ANIMAL_SPECIES)} Wikipedia articles.")

    # Part 2: Fictional creatures (handcrafted in the same taxonomic style)
    print("Adding fictional creature descriptions...")
    for c in FICTIONAL_CREATURES:
        entries.append(
            format_entry(
                c["common_name"], c["scientific_name"], c["phylum"],
                c["class"], c["order"], c["family"],
                c["habitat"], c["conservation"], c["desc"],
            )
        )
    print(f"  Added {len(FICTIONAL_CREATURES)} fictional species.")

    full_text = "\n".join(entries)
    corpus = "\n".join([full_text] * 5)

    CORPUS_FILE.write_text(corpus, encoding="utf-8")
    print(f"\nCorpus saved to {CORPUS_FILE}")
    print(f"  Unique entries: {len(entries)}")
    print(f"  Total characters: {len(corpus):,}")
    print(f"  Repeated 5x for training density")

    # Show a sample entry
    print("\n── Sample entry ──")
    print(entries[-1][:600] + "...")

    return corpus


if __name__ == "__main__":
    build_corpus()
