# 🌾 PaddyNet — Paddy Disease Agent

An **agentic AI layer** built on top of the EfficientNet paddy-disease classifier.
It turns the model from "predicts a label" into an assistant a farmer can talk to.

## What it does

1. **Sees** — a farmer uploads a rice-leaf photo; the trained CNN diagnoses the disease.
2. **Knows** — a curated agronomy knowledge base covers all 10 conditions
   (cause, symptoms, severity, treatment, prevention).
3. **Reasons & talks** — a **free local LLM (Ollama)** explains the diagnosis and
   answers follow-up questions ("is it safe to spray now?", "will it spread?").

The three pieces are wired together in `agent.py` (VISION → KNOWLEDGE → REASONING).

## 100% free — no API keys, no accounts

- The chat brain runs **locally via Ollama** — no cloud, no billing.
- If Ollama isn't installed, the app **still works fully** using the built-in
  knowledge base (diagnosis + structured treatment/prevention advice).

## Run it

```bat
:: from the PR\ folder
run_agent.bat
```
or manually:
```bat
venv\Scripts\activate
streamlit run agent\app.py
```

## Enable the conversational AI (optional, free)

1. Install Ollama → https://ollama.com/download
2. `ollama pull llama3.2`
3. Reload the app — the sidebar will show "Local AI online".

## Files

| File | Role |
|------|------|
| `config.py` | Paths, class names, model + LLM settings |
| `model_service.py` | Loads the `.keras` model, preprocesses, predicts |
| `knowledge_base.py` | Agronomy facts for all 10 classes + fallback report |
| `llm.py` | Free local Ollama client (with availability check) |
| `agent.py` | The agent: orchestrates vision + knowledge + LLM |
| `app.py` | Streamlit web UI (upload + diagnose + chat) |

## Note on preprocessing

The model input size is **read from the model itself** (EfficientNetB3 → 300×300)
and pixels are scaled by `1/255` to match how it was trained.
