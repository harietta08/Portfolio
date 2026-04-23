import os
import sys
from typing import Tuple, List, Set

import streamlit as st

# Make backend importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.orchestrator import chat_turn


# ---------- Page setup (only once, must be near top) ----------
st.set_page_config(page_title="IIT International Student Chatbot (Week 3)", layout="wide")


# ---------- Helpers ----------
def reset_chat_state():
    # Reset only the keys we use (safer than deleting everything)
    for k in ["history", "memory"]:
        if k in st.session_state:
            del st.session_state[k]


def split_sections(md: str) -> Tuple[str, str]:
    marker = "### Sources"
    if md and marker in md:
        before, after = md.split(marker, 1)
        return before.strip(), (marker + after).strip()
    return (md or "").strip(), ""


def dedupe_sources_block(sources_md: str) -> str:
    if not sources_md:
        return ""

    lines = sources_md.splitlines()
    out: List[str] = []
    seen: Set[str] = set()

    for line in lines:
        if line.strip().lower() == "### sources":
            out.append("### Sources")
            continue
        if line.strip().startswith("- "):
            key = line.strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(line)
        else:
            out.append(line)

    # Normalize multiple blank lines
    cleaned: List[str] = []
    for l in out:
        if cleaned and cleaned[-1].strip() == "" and l.strip() == "":
            continue
        cleaned.append(l)

    return "\n".join(cleaned).strip()


# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")

    if st.button("🔄 Reset chat (history + memory)"):
        reset_chat_state()
        st.rerun()

    if st.button("Reset memory only"):
        st.session_state["memory"] = {}

    st.divider()
    st.caption("Tip: Use 'Reset chat' before running evaluation questions.")


# ---------- UI header ----------
st.title("IIT International Student Chatbot")


# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = {}


# ---------- Render existing chat ----------
for role, content in st.session_state["history"]:
    with st.chat_message(role):
        st.markdown(content, unsafe_allow_html=True)


# ---------- Chat input ----------
q = st.chat_input("Ask about CPT, OPT, STEM OPT, travel, enrollment, RCL, insurance waiver, SSN...")
if q:
    st.session_state["history"].append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    result = chat_turn(q, memory=st.session_state["memory"])  # returns dict

    if result.get("memory") is not None:
        st.session_state["memory"] = result["memory"]

    mode = result.get("mode", "unknown")
    topic = result.get("topic")
    conf = result.get("topic_confidence")

    answer_md = result.get("answer_markdown", "")
    main_md, sources_md = split_sections(answer_md)
    sources_md = dedupe_sources_block(sources_md)

    with st.chat_message("assistant"):
        st.caption(f"Mode: {mode} | Topic: {topic} | Confidence: {conf}")
        st.markdown(main_md, unsafe_allow_html=True)

        decision = result.get("decision")
        # Show decision only for meaningful rules output (avoid showing unknown/null policy decisions)
        if mode == "rules" and isinstance(decision, dict) and decision.get("policy_id"):
            with st.expander("Decision (rule engine JSON)", expanded=False):
                st.json(decision)

        if sources_md:
            with st.expander("Sources", expanded=False):
                st.markdown(sources_md, unsafe_allow_html=True)

        with st.expander("Debug", expanded=False):
            st.json({
                "topic": topic,
                "topic_confidence": conf,
                "mode": mode,
                "retrieved_count": result.get("retrieved_count"),
                "memory": st.session_state["memory"],
                "decision": decision,
            })

    st.session_state["history"].append(("assistant", answer_md))