"""Central configuration for the paddy-disease agent."""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Preferred model file, with fallbacks (first that exists wins).
MODEL_CANDIDATES = [
    "efficientnet_final.keras",
    "efficientnet_best_phase3.keras",
    "efficientnet_phase3.keras",
    "efficientnet_best_phase2.keras",
]

# --- Model input ---
IMG_SIZE = 300          # fallback; actual size is read from the model at runtime
RESCALE = 1.0 / 255.0   # training used ImageDataGenerator(rescale=1./255)

# --- Classes (alphabetical order == flow_from_directory order used in training) ---
CLASS_NAMES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]

# --- LLM (Ollama, free & local) ---
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"   # `ollama pull llama3.2` — swap for any local model
LLM_TIMEOUT = 120           # seconds
