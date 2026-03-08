"""
Component 4 — Chatbot service.

"""

import logging
import re
import time
from datetime import datetime
from groq import Groq
import config
from services.firebase_service import get_db

logger = logging.getLogger(__name__)

DEFAULT_MODEL = config.GROQ_CHAT_MODEL

SYSTEM_PROMPT = (
    "You are CardioSense, a supportive health companion STRICTLY focused on "
    "cardiovascular health and mental/emotional well-being for people living "
    "with or at risk of cardiovascular disease.\n\n"
    "IMPORTANT RULES:\n"
    "1. ONLY discuss topics related to: heart health, cardiovascular disease, "
    "blood pressure, cholesterol, heart medications, cardiac symptoms, "
    "exercise for heart health, cardiac diet/nutrition, stress management, "
    "anxiety, depression, mental health, emotional well-being, sleep, "
    "and lifestyle changes for heart health.\n"
    "2. If the user asks about ANY topic outside cardiovascular or mental "
    "health (e.g. politics, geography, coding, general knowledge, sports, "
    "entertainment, history, science unrelated to heart health), you MUST "
    "politely decline and redirect them. Say something like: "
    "'I appreciate your curiosity, but I'm specifically designed to help "
    "with cardiovascular and mental health topics. Is there anything about "
    "your heart health or emotional well-being I can help with?'\n"
    "3. NEVER answer general knowledge questions, even if simple.\n"
    "4. Use a warm, calm, supportive tone. Be practical.\n"
    "5. Ask 1 short clarifying question when needed.\n"
    "6. Keep replies concise. Do not diagnose. Do not claim certainty.\n"
    "7. If the user mentions self-harm or suicide, urge urgent help and "
    "encourage contacting local emergency services.\n"
    "8. If the user describes emergency symptoms (possible heart attack or "
    "stroke), urge urgent medical care."
)

SELF_HARM_PAT = re.compile(
    r"(suicide|kill\s+myself|self[-\s]*harm|end\s+my\s+life|"
    r"want\s+to\s+die|overdose)",
    re.IGNORECASE,
)

EMERGENCY_PAT = re.compile(
    r"(heart\s*attack|chest\s*pain|pressure\s+in\s+chest|pain\s+spreading|"
    r"left\s+arm|jaw\s+pain|shortness\s+of\s+breath|sweating|fainting|"
    r"stroke|face\s+droop|slurred\s+speech|one\s+side\s+weakness)",
    re.IGNORECASE,
)

# ── Singleton client ───────────────────────────────────────────────
_client = None


def _get_client():
    global _client
    if _client is None and config.CARDIOSENSE_KEY:
        _client = Groq(api_key=config.CARDIOSENSE_KEY)
    return _client


def _clean(s):
    return re.sub(r"\s+", " ", str(s or "").strip())


def detect_route(text):
    if SELF_HARM_PAT.search(text):
        return "self_harm"
    if EMERGENCY_PAT.search(text):
        return "emergency"
    return "normal"


def _emergency_reply():
    return (
        "If you have chest pain, pressure, spreading pain, severe shortness "
        "of breath, fainting, or stroke signs, please seek urgent medical "
        "help now. Call local emergency services or go to the nearest "
        "emergency unit."
    )


def _self_harm_reply():
    return (
        "I'm really sorry you're feeling this way. You deserve support right "
        "now. If you are in immediate danger, please call your local emergency "
        "number. Are you safe right now?"
    )


# ── Chat sessions ──────────────────────────────────────────────────

def get_or_create_session(uid):
    """Get the latest open chat session or create one."""
    db = get_db()
    if not db:
        return None
    # Check for existing session
    docs = list(db.collection("chat_sessions")
            .where("uid", "==", uid)
            .stream())
    if docs:
        docs.sort(key=lambda d: d.to_dict().get("created_at", ""), reverse=True)
        d = docs[0].to_dict()
        d["session_id"] = docs[0].id
        return d

    # Create new session
    now = datetime.utcnow().isoformat()
    ref = db.collection("chat_sessions").add({
        "uid": uid,
        "created_at": now,
        "updated_at": now,
    })
    return {"session_id": ref[1].id, "uid": uid, "created_at": now}


def create_new_session(uid):
    """Force-create a new chat session."""
    db = get_db()
    if not db:
        return None
    now = datetime.utcnow().isoformat()
    ref = db.collection("chat_sessions").add({
        "uid": uid,
        "created_at": now,
        "updated_at": now,
    })
    return {"session_id": ref[1].id, "uid": uid, "created_at": now}


def get_session_messages(session_id):
    """Fetch all messages for a session."""
    db = get_db()
    if not db:
        return []
    docs = (db.collection("chat_messages")
            .where("session_id", "==", session_id)
            .stream())
    results = [doc.to_dict() for doc in docs]
    results.sort(key=lambda x: x.get("created_at", ""))
    return results


def _store_message(session_id, uid, role, content):
    db = get_db()
    if not db:
        return
    now = datetime.utcnow().isoformat()
    db.collection("chat_messages").add({
        "session_id": session_id,
        "uid": uid,
        "role": role,
        "content": content,
        "created_at": now,
    })
    # Update session timestamp
    db.collection("chat_sessions").document(session_id).update({
        "updated_at": now,
    })


def send_message(session_id, uid, user_text):
    """
    Process a user chat message:
    1. Store user message
    2. Safety-route
    3. Generate response (with 3-5s delay)
    4. Store assistant message
    Returns the assistant reply text.
    """
    cleaned = _clean(user_text)
    if not cleaned:
        return "I'm here. What's on your mind?"

    _store_message(session_id, uid, "user", cleaned)

    route = detect_route(cleaned)

    # Fixed delay for natural feel
    time.sleep(4)

    if route == "emergency":
        reply = _emergency_reply()
    elif route == "self_harm":
        reply = _self_harm_reply()
    else:
        client = _get_client()
        if not client:
            reply = (
                "I'm here for you, but my connection is temporarily unavailable. "
                "Please try again in a moment."
            )
        else:
            # Build context from recent messages
            history = get_session_messages(session_id)
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            for m in history[-10:]:
                if m.get("role") in ("user", "assistant") and m.get("content"):
                    msgs.append({"role": m["role"], "content": m["content"]})

            try:
                resp = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=msgs,
                    temperature=0.7,
                    top_p=0.9,
                    max_completion_tokens=220,
                    stream=False,
                )
                reply = _clean(
                    resp.choices[0].message.content if resp.choices else ""
                )
                if not reply:
                    reply = "I'm here with you. What feels hardest right now?"
            except Exception:
                reply = (
                    "I'm sorry, I had trouble processing that. Could you "
                    "try rephrasing?"
                )

    _store_message(session_id, uid, "assistant", reply)
    return reply


def get_patient_chat_sessions(uid):
    """Get all chat sessions for a patient (for doctor view)."""
    db = get_db()
    if not db:
        return []
    docs = (db.collection("chat_sessions")
            .where("uid", "==", uid)
            .stream())
    sessions = []
    for doc in docs:
        d = doc.to_dict()
        d["session_id"] = doc.id
        sessions.append(d)
    sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return sessions
