import os
import re
import time

import streamlit as st
from groq import Groq

APP_TITLE = "CardioSense Chatbot"
APP_CAPTION = "Supportive mental health companion for people living with cardiovascular disease."

DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
SECRET_NAME = "CS_KEY_9F3A7C"

SYSTEM_PROMPT = (
    "You are CardioSense, a supportive mental health companion for people living with cardiovascular disease. "
    "Use a warm, calm, supportive tone. Be practical. Ask 1 short clarifying question when needed. "
    "Keep replies concise. Do not diagnose. Do not claim certainty. "
    "If the user mentions self-harm or suicide, urge urgent help and encourage contacting local emergency services. "
    "If the user describes emergency symptoms (possible heart attack or stroke), urge urgent medical care."
)

SELF_HARM_PAT = re.compile(
    r"(suicide|kill\s+myself|self[-\s]*harm|end\s+my\s+life|want\s+to\s+die|overdose)",
    re.IGNORECASE,
)

EMERGENCY_PAT = re.compile(
    r"(heart\s*attack|chest\s*pain|pressure\s+in\s+chest|pain\s+spreading|left\s+arm|jaw\s+pain|"
    r"shortness\s+of\s+breath|sweating|fainting|stroke|face\s+droop|slurred\s+speech|one\s+side\s+weakness)",
    re.IGNORECASE,
)

def _clean_text(s):
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def detect_route(user_text):
    if SELF_HARM_PAT.search(user_text):
        return "self_harm"
    if EMERGENCY_PAT.search(user_text):
        return "emergency"
    return "normal"

def emergency_reply():
    return (
        "If you have chest pain, pressure, spreading pain, severe shortness of breath, fainting, or stroke signs, "
        "please seek urgent medical help now. Call local emergency services or go to the nearest emergency unit. "
        "If you can, tell me your symptoms, when they started, and your age."
    )

def self_harm_reply():
    return (
        "I’m really sorry you’re feeling this way. You deserve support right now. "
        "If you are in immediate danger or might act on these thoughts, please call your local emergency number now. "
        "If you can, tell me your country so I can suggest crisis contact options. "
        "Are you safe right now?"
    )

def get_api_key():
    if SECRET_NAME in st.secrets:
        return str(st.secrets[SECRET_NAME]).strip()
    return str(os.environ.get(SECRET_NAME, "")).strip()

@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return Groq(api_key=api_key)

def build_messages(turns):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in turns[-10:]:
        role = t.get("role", "")
        content = t.get("content", "")
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    return msgs

def groq_chat(client: Groq, messages, temperature: float, top_p: float, max_tokens: int):
    r = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,
        stream=False,
    )
    txt = ""
    if r and getattr(r, "choices", None):
        m = r.choices[0].message
        if m and getattr(m, "content", None):
            txt = m.content
    return _clean_text(txt)

def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    api_key = get_api_key()
    if not api_key:
        st.stop()

    client = get_client(api_key)

    with st.sidebar:
        st.subheader("Generation settings")
        max_new_tokens = st.slider("Max new tokens", 64, 512, 220, 16)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

    if "turns" not in st.session_state:
        st.session_state.turns = []

    for msg in st.session_state.turns:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask CardioSense anything...")
    if user_input is None:
        return

    cleaned = _clean_text(user_input)
    if not cleaned:
        return

    st.session_state.turns.append({"role": "user", "content": cleaned})
    with st.chat_message("user"):
        st.markdown(cleaned)

    route = detect_route(cleaned)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(5)

            if route == "emergency":
                reply = emergency_reply()
            elif route == "self_harm":
                reply = self_harm_reply()
            else:
                messages = build_messages(st.session_state.turns)
                reply = groq_chat(
                    client=client,
                    messages=messages,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(max_new_tokens),
                )
                if not reply:
                    reply = "I’m here with you. What feels hardest right now?"

            st.markdown(reply)

    st.session_state.turns.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
