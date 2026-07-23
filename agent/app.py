"""Streamlit UI for the Paddy Disease Agent.

Run from the project root (PR/):
    venv\\Scripts\\streamlit run agent/app.py
"""
import sys
from pathlib import Path

# allow "import agent.*" when run via `streamlit run agent/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from agent import config, llm
from agent.agent import PaddyAgent
from agent.knowledge_base import KB

st.set_page_config(page_title="PaddyNet — Paddy Disease Agent", page_icon="🌾", layout="centered")


@st.cache_resource
def get_agent():
    return PaddyAgent()


def confidence_color(c: float) -> str:
    return "🟢" if c >= 0.8 else ("🟡" if c >= 0.6 else "🔴")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🌾 PaddyNet")
    st.caption("An AI agent that diagnoses paddy leaf diseases and advises on treatment.")

    llm_ok = llm.is_available()
    if llm_ok:
        st.success(f"🧠 Local AI online ({config.OLLAMA_MODEL})")
        st.caption("Full conversational advice enabled.")
    else:
        st.warning("🧠 Local AI offline — using built-in knowledge base")
        with st.expander("Turn on conversational AI (free)"):
            st.markdown(
                "1. Install **Ollama** → https://ollama.com/download\n"
                f"2. In a terminal: `ollama pull {config.OLLAMA_MODEL}`\n"
                "3. Reload this page.\n\n"
                "It runs fully on your PC — no account, no cost."
            )

    st.divider()
    st.caption("Detects 10 conditions:")
    st.caption(", ".join(info["name"] for info in KB.values()))


# ---------------- Main ----------------
st.title("🌾 Paddy Disease Agent")
st.caption("Upload a photo of a rice leaf. The agent diagnoses it and helps you treat it.")

agent = get_agent()

uploaded = st.file_uploader("Upload a rice leaf photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Your leaf", use_container_width=True)
    with col2:
        # diagnose only once per uploaded file
        if st.session_state.get("last_file") != uploaded.name:
            with st.spinner("🔬 Diagnosing…"):
                result = agent.diagnose(image)
            st.session_state["last_file"] = uploaded.name
            st.session_state["result"] = result
            st.session_state["messages"] = [
                {"role": "assistant", "content": result["message"]}
            ]

        result = st.session_state["result"]
        diag = result["diagnosis"]
        name = KB[diag["disease"]]["name"]
        c = diag["confidence"]
        st.metric("Diagnosis", name, f"{confidence_color(c)} {c*100:.1f}% confidence")
        with st.expander("All class probabilities"):
            for cls, p in sorted(diag["all_probs"].items(), key=lambda x: -x[1]):
                st.write(f"{KB[cls]['name']}")
                st.progress(p)

    st.divider()
    st.subheader("💬 Ask the agent")

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"], avatar="🌾" if msg["role"] == "assistant" else "🧑‍🌾"):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about treatment, spread, prevention…")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🌾"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🌾"):
            reply = st.write_stream(agent.chat_stream(prompt))
        st.session_state["messages"].append({"role": "assistant", "content": reply})
else:
    st.info("👆 Upload a leaf photo to begin. No photo handy? Grab one from the `archive/` folder.")
