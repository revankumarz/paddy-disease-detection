"""Curated agronomy knowledge base for the 10 paddy leaf conditions.

Used two ways:
  1. As a deterministic fallback when no LLM is running.
  2. As grounding context injected into the LLM prompt so advice stays factual.
"""

KB = {
    "bacterial_leaf_blight": {
        "name": "Bacterial Leaf Blight",
        "pathogen": "Bacterium — Xanthomonas oryzae pv. oryzae",
        "severity": "High",
        "symptoms": [
            "Water-soaked yellow stripes along leaf margins that turn grey-white and dry",
            "Lesions spread from the leaf tip downward, wavy edges",
            "In severe 'kresek' stage seedlings wilt and die",
        ],
        "favoured_by": "Warm humid weather, heavy rain/wind injury, excess nitrogen, standing water",
        "treatment": [
            "Drain the field and stop applying nitrogen until it is controlled",
            "Spray copper-based bactericide (copper hydroxide/oxychloride) or streptocycline where permitted",
            "Remove and destroy severely infected plants and weed hosts",
        ],
        "prevention": [
            "Plant resistant varieties (e.g. IRBB lines)",
            "Use certified disease-free seed; treat seed with hot water/approved chemical",
            "Balanced fertiliser — avoid excess nitrogen; ensure good field drainage",
        ],
    },
    "bacterial_leaf_streak": {
        "name": "Bacterial Leaf Streak",
        "pathogen": "Bacterium — Xanthomonas oryzae pv. oryzicola",
        "severity": "Moderate to High",
        "symptoms": [
            "Narrow, dark-green water-soaked streaks between leaf veins",
            "Streaks turn yellow-brown to grey; tiny amber bacterial beads may ooze",
            "Streaks stay confined by veins (unlike blight's marginal spread)",
        ],
        "favoured_by": "High humidity, rain splash, wounds, dense canopy",
        "treatment": [
            "Avoid overhead/splash irrigation; improve drainage and air flow",
            "Copper-based sprays can slow spread; remove infected debris",
            "Reduce nitrogen top-dressing during an active outbreak",
        ],
        "prevention": [
            "Certified clean seed and seed treatment",
            "Resistant varieties and proper plant spacing",
            "Sanitation — remove crop residue and grassy weed hosts",
        ],
    },
    "bacterial_panicle_blight": {
        "name": "Bacterial Panicle Blight",
        "pathogen": "Bacterium — Burkholderia glumae",
        "severity": "High",
        "symptoms": [
            "Florets/grains discolour to straw or brown; panicles stay upright and unfilled",
            "Grain sterility and blanking, uneven ripening",
            "Sheath and upper leaf discolouration in hot spells",
        ],
        "favoured_by": "High night temperatures and heat stress at flowering, dense stands",
        "treatment": [
            "No fully effective curative spray — manage the crop to reduce stress",
            "Maintain adequate water to buffer heat stress during flowering",
            "Avoid excess nitrogen which worsens severity",
        ],
        "prevention": [
            "Use tolerant varieties and clean seed",
            "Adjust planting date so flowering avoids peak heat",
            "Balanced nutrition and moderate seeding density",
        ],
    },
    "blast": {
        "name": "Rice Blast",
        "pathogen": "Fungus — Magnaporthe oryzae (Pyricularia oryzae)",
        "severity": "High",
        "symptoms": [
            "Spindle/diamond-shaped lesions with grey centres and brown-red margins",
            "Lesions on leaves, collar, node and neck of the panicle",
            "Neck blast breaks the panicle base causing whiteheads and major yield loss",
        ],
        "favoured_by": "Cool nights with dew/long leaf wetness, high humidity, excess nitrogen",
        "treatment": [
            "Spray a systemic fungicide (tricyclazole, or azoxystrobin/isoprothiolane) at first lesions",
            "Repeat at booting/heading to protect the neck if pressure is high",
            "Stop further nitrogen and drain to reduce canopy humidity",
        ],
        "prevention": [
            "Grow resistant varieties; rotate/mix genes to avoid breakdown",
            "Split nitrogen doses, avoid heavy single applications",
            "Silicon amendment and good spacing to speed leaf drying",
        ],
    },
    "brown_spot": {
        "name": "Brown Spot",
        "pathogen": "Fungus — Bipolaris oryzae (Cochliobolus miyabeanus)",
        "severity": "Moderate",
        "symptoms": [
            "Numerous small oval brown spots with a tan/grey centre and dark margin",
            "Spots evenly scattered over the leaf; a classic sign of nutrient-poor soil",
            "Dark spots also on grains, reducing quality",
        ],
        "favoured_by": "Nutrient (especially potassium) deficiency, drought stress, poor unfertile soils",
        "treatment": [
            "Correct the underlying nutrient deficiency (K and micronutrients)",
            "Foliar fungicide (mancozeb or propiconazole) if it reaches grain-fill",
            "Improve irrigation to relieve drought stress",
        ],
        "prevention": [
            "Balanced fertilisation, adequate potassium; add organic matter to poor soils",
            "Clean, treated seed",
            "Steady water supply to avoid moisture stress",
        ],
    },
    "dead_heart": {
        "name": "Dead Heart (Stem Borer damage)",
        "pathogen": "Insect pest — rice stem borer larvae (Scirpophaga / Chilo spp.)",
        "severity": "Moderate to High",
        "symptoms": [
            "Central shoot dries and dies while outer leaves stay green ('dead heart')",
            "The dead central tiller pulls out easily",
            "At panicle stage the same damage shows as empty 'whiteheads'",
        ],
        "favoured_by": "Continuous rice cropping, standing stubble, moderate temperatures",
        "treatment": [
            "This is insect damage, not a disease — target the borer, not a fungicide",
            "Clip and destroy egg masses; release Trichogramma parasitoids",
            "If threshold exceeded, apply a recommended granular insecticide (e.g. cartap/chlorantraniliprole)",
        ],
        "prevention": [
            "Plough under stubble and destroy volunteer plants after harvest",
            "Synchronised planting and light traps to catch adult moths",
            "Avoid excess nitrogen which attracts egg-laying",
        ],
    },
    "downy_mildew": {
        "name": "Downy Mildew",
        "pathogen": "Fungus-like oomycete — Sclerophthora macrospora",
        "severity": "Low to Moderate",
        "symptoms": [
            "Chlorotic yellow mottling and twisting/thickening of leaves",
            "Excessive tillering, stunting and distorted 'crazy top' growth",
            "Malformed or sterile panicles in severe cases",
        ],
        "favoured_by": "Flooding/waterlogging of young seedlings, cool wet conditions",
        "treatment": [
            "Improve drainage immediately — remove standing water from seedlings",
            "Rogue out badly distorted plants",
            "Metalaxyl-based seed treatment/soil drench where available",
        ],
        "prevention": [
            "Avoid flooding of very young seedlings; level fields for even drainage",
            "Clean seed and destruction of infected debris",
            "Control grassy weeds that host the pathogen",
        ],
    },
    "hispa": {
        "name": "Rice Hispa",
        "pathogen": "Insect pest — Dicladispa armigera (leaf beetle)",
        "severity": "Moderate",
        "symptoms": [
            "White parallel scraped streaks where adults feed on the leaf surface",
            "Larvae tunnel (mine) inside the leaf making irregular translucent blotches",
            "Heavy attack gives the field a scorched whitish look",
        ],
        "favoured_by": "High humidity, dense weedy fields, excess nitrogen, early crop stage",
        "treatment": [
            "Again an insect — clip and destroy heavily mined leaf tips (removes larvae)",
            "Avoid further nitrogen which favours the beetle",
            "Spray a recommended insecticide only if damage crosses threshold",
        ],
        "prevention": [
            "Keep bunds and field weed-free; balanced nitrogen",
            "Manual collection of adult beetles in early infestation",
            "Encourage natural predators; avoid broad-spectrum sprays early",
        ],
    },
    "normal": {
        "name": "Normal (Healthy)",
        "pathogen": "None — no disease detected",
        "severity": "None",
        "symptoms": [
            "Uniform green leaves, no lesions, streaks, spots or mining",
            "Healthy tillering and upright vigorous growth",
        ],
        "favoured_by": "Balanced nutrition, good water and pest management",
        "treatment": [
            "No treatment needed — the leaf looks healthy",
        ],
        "prevention": [
            "Keep scouting weekly, especially after rain or heat",
            "Maintain balanced fertiliser and clean, well-drained fields",
            "Use certified seed and resistant varieties as a baseline",
        ],
    },
    "tungro": {
        "name": "Tungro",
        "pathogen": "Viral disease (RTBV + RTSV) spread by green leafhoppers",
        "severity": "High",
        "symptoms": [
            "Yellow-orange discolouration from the leaf tip downward",
            "Stunted plants with reduced tillering and delayed flowering",
            "Mottled young leaves; partly filled, discoloured grain",
        ],
        "favoured_by": "Green leafhopper populations, overlapping/continuous rice crops",
        "treatment": [
            "No cure for the virus — remove and destroy infected plants promptly",
            "Control the green leafhopper vector (recommended insecticide / neem)",
            "Do not replant into the same standing infected crop",
        ],
        "prevention": [
            "Plant tungro-resistant varieties and use a synchronised planting window",
            "Manage leafhoppers early; keep a rice-free fallow to break the cycle",
            "Rogue infected plants as soon as symptoms appear",
        ],
    },
}


def get_disease_info(disease_key: str) -> dict:
    """Return the KB entry for a class key (e.g. 'blast')."""
    return KB.get(disease_key, {})


def format_disease_report(disease_key: str, confidence: float | None = None) -> str:
    """Human-readable, structured advice card — the no-LLM fallback output."""
    info = KB.get(disease_key)
    if not info:
        return f"No knowledge-base entry found for '{disease_key}'."

    conf = f" ({confidence * 100:.1f}% confidence)" if confidence is not None else ""
    lines = [
        f"### {info['name']}{conf}",
        f"**Cause:** {info['pathogen']}  ",
        f"**Severity:** {info['severity']}",
        "",
        "**What you're seeing (symptoms):**",
    ]
    lines += [f"- {s}" for s in info["symptoms"]]
    lines += ["", f"**Conditions that favour it:** {info['favoured_by']}", "", "**What to do now (treatment):**"]
    lines += [f"- {s}" for s in info["treatment"]]
    lines += ["", "**How to prevent it next season:**"]
    lines += [f"- {s}" for s in info["prevention"]]
    return "\n".join(lines)
