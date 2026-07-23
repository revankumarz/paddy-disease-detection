"""Loads the trained EfficientNet model and exposes a single predict() tool.

Preprocessing intentionally mirrors training exactly:
    resize -> 224x224, rescale pixels by 1/255, add batch dimension.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import config

_model = None          # lazily loaded singleton
_model_path = None


def _resolve_model_path():
    for name in config.MODEL_CANDIDATES:
        p = config.MODELS_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No model file found in {config.MODELS_DIR}. "
        f"Expected one of: {', '.join(config.MODEL_CANDIDATES)}"
    )


def load_model():
    """Load (once) and return the Keras model."""
    global _model, _model_path
    if _model is None:
        import tensorflow as tf  # imported lazily so the app starts fast
        _model_path = _resolve_model_path()
        _model = tf.keras.models.load_model(_model_path, compile=False)
    return _model


def get_model_name() -> str:
    return _model_path.name if _model_path else "(not loaded)"


def _input_size() -> int:
    """Read the model's expected HxW so we always match how it was trained."""
    try:
        shape = load_model().inputs[0].shape  # (None, H, W, 3)
        if shape[1]:
            return int(shape[1])
    except Exception:
        pass
    return config.IMG_SIZE


def _preprocess(image: Image.Image) -> np.ndarray:
    size = _input_size()
    img = image.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) * config.RESCALE
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def predict(image: Image.Image, top_k: int = 3) -> dict:
    """Run the classifier on a PIL image.

    Returns:
        {
          "disease": "blast",
          "confidence": 0.82,
          "top_k": [("blast", 0.82), ("brown_spot", 0.09), ...],
          "all_probs": {class: prob, ...},
        }
    """
    model = load_model()
    batch = _preprocess(image)
    probs = model.predict(batch, verbose=0)[0]

    order = np.argsort(probs)[::-1]
    all_probs = {config.CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
    top = [(config.CLASS_NAMES[i], float(probs[i])) for i in order[:top_k]]

    best_idx = int(order[0])
    return {
        "disease": config.CLASS_NAMES[best_idx],
        "confidence": float(probs[best_idx]),
        "top_k": top,
        "all_probs": all_probs,
    }
