"""Thin client for a local, free Ollama LLM.

If Ollama isn't installed / running, is_available() returns False and the
agent falls back to the deterministic knowledge-base responses. No API keys,
no accounts, no cost — everything runs on this machine.
"""
from __future__ import annotations

import requests

from . import config


def is_available() -> bool:
    """True if an Ollama server is reachable and the model is pulled."""
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        models = [m.get("name", "") for m in r.json().get("models", [])]
        # match "llama3.2" against "llama3.2:latest" etc.
        return any(m.split(":")[0] == config.OLLAMA_MODEL.split(":")[0] for m in models)
    except Exception:
        return False


def _build_messages(messages, system):
    payload = []
    if system:
        payload.append({"role": "system", "content": system})
    payload.extend(messages)
    return payload


# Options tuned for speed on CPU: shorter context + capped output length.
_OPTIONS = {"temperature": 0.3, "num_ctx": 2048, "num_predict": 400}


def chat(messages: list[dict], system: str | None = None) -> str:
    """Send a chat conversation to Ollama and return the full assistant reply."""
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "messages": _build_messages(messages, system),
            "stream": False,
            "options": _OPTIONS,
        },
        timeout=config.LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def chat_stream(messages: list[dict], system: str | None = None):
    """Yield the assistant reply token-by-token as it is generated."""
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "messages": _build_messages(messages, system),
            "stream": True,
            "options": _OPTIONS,
        },
        timeout=config.LLM_TIMEOUT,
        stream=True,
    )
    resp.raise_for_status()
    import json as _json
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = _json.loads(line)
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece
        if chunk.get("done"):
            break
