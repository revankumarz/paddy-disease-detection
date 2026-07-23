"""The agentic layer.

Ties three capabilities together:
  * VISION  -> the trained EfficientNet classifier (model_service.predict)
  * KNOWLEDGE -> the curated agronomy KB (knowledge_base)
  * REASONING -> a local free LLM (llm) with a KB-only fallback

The agent keeps the current diagnosis as session state so the farmer can
ask natural follow-up questions ("is it safe to spray now?", "will it
spread to my other field?") and get grounded answers.
"""
from __future__ import annotations

from PIL import Image

from . import knowledge_base as kb
from . import llm, model_service

SYSTEM_PROMPT = """You are 'PaddyNet', a friendly, practical rice-farming advisor \
helping smallholder farmers. An image classifier has already diagnosed the disease \
from a leaf photo — you do NOT identify the disease yourself; you explain it and advise.

Rules:
- Base your facts ONLY on the DIAGNOSIS and KNOWLEDGE provided below. Do not invent \
chemicals, brand names or dosages.
- Be concise, warm and practical. Use simple language a farmer can act on today.
- For any pesticide/fungicide, remind the farmer to follow the local label and consult \
their agricultural extension officer for exact dosage and safety.
- If confidence is low (below 60%), gently say the photo is uncertain and suggest a \
clearer close-up in good light, or a second opinion.
- If the leaf is 'Normal', reassure them and give light monitoring tips.
"""


def _grounding_block(diagnosis: dict) -> str:
    disease = diagnosis["disease"]
    conf = diagnosis["confidence"]
    info = kb.get_disease_info(disease)
    top = ", ".join(f"{d} {p*100:.0f}%" for d, p in diagnosis["top_k"])

    return (
        f"DIAGNOSIS: {info.get('name', disease)} at {conf*100:.1f}% confidence.\n"
        f"Top predictions: {top}.\n\n"
        f"KNOWLEDGE (facts you may use):\n"
        f"- Cause: {info.get('pathogen')}\n"
        f"- Severity: {info.get('severity')}\n"
        f"- Symptoms: {'; '.join(info.get('symptoms', []))}\n"
        f"- Favoured by: {info.get('favoured_by')}\n"
        f"- Treatment: {'; '.join(info.get('treatment', []))}\n"
        f"- Prevention: {'; '.join(info.get('prevention', []))}\n"
    )


class PaddyAgent:
    def __init__(self):
        self.diagnosis: dict | None = None
        self.history: list[dict] = []   # chat history for the LLM
        self.use_llm = llm.is_available()

    # ---- capability 1 + 2: diagnose then advise ----
    def diagnose(self, image: Image.Image) -> dict:
        """Run the vision model and produce an opening assessment message."""
        self.diagnosis = model_service.predict(image)
        self.history = []

        if self.use_llm:
            user_kickoff = (
                "Here is the diagnosis for the farmer's leaf photo. Give a short, "
                "friendly opening assessment: what it likely is, how serious, and the "
                "top 2-3 things to do now. Then invite them to ask questions."
            )
            opening = self._llm_reply(user_kickoff, record_user=False)
        else:
            opening = self._fallback_opening()

        self.history.append({"role": "assistant", "content": opening})
        return {"diagnosis": self.diagnosis, "message": opening}

    # ---- capability 3: conversational follow-ups ----
    def chat(self, user_message: str) -> str:
        if self.diagnosis is None:
            return "Please upload a leaf photo first so I can diagnose it, then I can help."

        if self.use_llm:
            reply = self._llm_reply(user_message, record_user=True)
        else:
            reply = self._fallback_chat(user_message)
            self.history.append({"role": "user", "content": user_message})

        self.history.append({"role": "assistant", "content": reply})
        return reply

    def chat_stream(self, user_message: str):
        """Streaming version of chat(): yields text pieces, updates history at end.

        Falls back to yielding the whole KB answer at once when no LLM is available.
        """
        if self.diagnosis is None:
            yield "Please upload a leaf photo first so I can diagnose it, then I can help."
            return

        if not self.use_llm:
            reply = self._fallback_chat(user_message)
            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": reply})
            yield reply
            return

        system = SYSTEM_PROMPT + "\n\n" + _grounding_block(self.diagnosis)
        messages = list(self.history) + [{"role": "user", "content": user_message}]
        self.history.append({"role": "user", "content": user_message})

        collected = []
        try:
            for piece in llm.chat_stream(messages, system=system):
                collected.append(piece)
                yield piece
        except Exception as e:
            self.use_llm = False
            fb = self._fallback_chat(user_message)
            collected = [fb + f"\n\n_(Local AI unavailable — {e}. Showing knowledge-base answer.)_"]
            yield collected[0]
        self.history.append({"role": "assistant", "content": "".join(collected)})

    # ---- LLM path ----
    def _llm_reply(self, user_message: str, record_user: bool) -> str:
        system = SYSTEM_PROMPT + "\n\n" + _grounding_block(self.diagnosis)
        messages = list(self.history)
        messages.append({"role": "user", "content": user_message})
        if record_user:
            self.history.append({"role": "user", "content": user_message})
        try:
            return llm.chat(messages, system=system)
        except Exception as e:
            # network hiccup / model unloaded -> degrade gracefully
            self.use_llm = False
            return (self._fallback_chat(user_message)
                    + f"\n\n_(Note: local AI unavailable — {e}. Showing knowledge-base answer.)_")

    # ---- deterministic fallback path ----
    def _fallback_opening(self) -> str:
        d = self.diagnosis
        report = kb.format_disease_report(d["disease"], d["confidence"])
        header = ""
        if d["disease"] != "normal" and d["confidence"] < 0.60:
            header = ("> ⚠️ The photo is a bit uncertain. For a firmer answer, retake a "
                      "sharp close-up of the affected leaf in good daylight.\n\n")
        return header + report

    def _fallback_chat(self, user_message: str) -> str:
        """Keyword-routed answers from the KB when no LLM is available."""
        info = kb.get_disease_info(self.diagnosis["disease"])
        q = user_message.lower()

        def bullets(items):
            return "\n".join(f"- {x}" for x in items)

        if any(w in q for w in ["treat", "cure", "spray", "control", "fix", "medicine", "fungicide", "pesticide"]):
            return f"**Treatment for {info['name']}:**\n{bullets(info['treatment'])}\n\n" \
                   "_Always follow the local product label and ask your extension officer for exact dosage._"
        if any(w in q for w in ["prevent", "avoid", "next season", "stop it", "protect"]):
            return f"**Prevention for {info['name']}:**\n{bullets(info['prevention'])}"
        if any(w in q for w in ["symptom", "sign", "look", "identify", "spot"]):
            return f"**Symptoms of {info['name']}:**\n{bullets(info['symptoms'])}"
        if any(w in q for w in ["cause", "why", "pathogen", "reason", "bacteria", "fungus", "virus"]):
            return f"**Cause:** {info['pathogen']}\n**Favoured by:** {info['favoured_by']}"
        if any(w in q for w in ["serious", "severe", "bad", "danger", "yield", "loss"]):
            return f"**Severity of {info['name']}: {info['severity']}.**\nFavoured by: {info['favoured_by']}"
        if any(w in q for w in ["spread", "contagious", "other field", "neighbour"]):
            return f"{info['name']} is favoured by: {info['favoured_by']}. " \
                   "Manage those conditions and follow the prevention steps to limit spread:\n" \
                   f"{bullets(info['prevention'])}"

        # default: full card
        return kb.format_disease_report(self.diagnosis["disease"], self.diagnosis["confidence"])
