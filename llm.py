"""llm.py
Gemini-based English tutor logic with context + topic management.
- .env: GEMINI_API_KEY
- Exposes:
    refine_transcript(raw_transcript: str,
                      alt_segments: list[list[str]],
                      chat_history: list[dict],
                      current_topic: str|None) -> str
    gemini_analyze(transcript: str, chat_history: list[dict], current_topic: str|None) -> dict
"""

import os, re, json
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

from google import generativeai as genai

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your .env")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
MAX_HISTORY_TURNS = 6

# ============== Helpers ==============
def _safe_read_text(resp) -> str:
    """Avoid resp.text quick accessor; concatenate parts safely."""
    out = ""
    try:
        if getattr(resp, "candidates", None):
            for c in resp.candidates:
                if getattr(c, "content", None) and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        if getattr(p, "text", None):
                            out += p.text
    except Exception:
        pass
    return out.strip()

def _robust_json_parse(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s or "")
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

# ============== Refinement ==============
REFINE_SYSTEM = (
    "You are an ASR post-processor. Given conversation context, topic, and N-best hypotheses, "
    "produce ONE fluent English sentence that best reflects the user's intent. "
    "Strongly prefer coherent, idiomatic phrases; repair malformed or nonsensical fragments. "
    "If multiple partial phrases exist, merge and lightly paraphrase to make natural English. "
    'Output STRICT JSON: {"refined": "..."} only.'
)

def refine_transcript(raw_transcript: str,
                      alt_segments: List[List[str]],
                      chat_history: List[Dict[str, str]],
                      current_topic: Optional[str]) -> str:
    """Use LLM + context to pick the best overall transcript."""
    model = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=REFINE_SYSTEM)

    payload = {
        "current_topic": current_topic,
        "chat_history_tail": chat_history[-MAX_HISTORY_TURNS:],
        "raw_transcript": raw_transcript,
        "alt_segments": alt_segments[:20],  # safety cap
        "instructions": (
            "Choose or lightly edit text from the alternatives to form ONE coherent sentence. "
            "Prefer named entities that fit the topic/context."
        ),
    }

    resp = model.generate_content(
        [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
        generation_config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 640},
    )
    raw = _safe_read_text(resp)
    if not raw:
        # safety fallback
        return raw_transcript or ""

    data = _robust_json_parse(raw)
    refined = (data.get("refined") or "").strip()
    return refined or (raw_transcript or "")

# ============== Tutor ==============
SYSTEM_PROMPT = (
    "You are an expert English tutor for Korean learners.\n"
    "Maintain conversational context using the provided chat_history and current_topic.\n"
    "If current_topic is null, first propose 3 short topics in English and ask the learner to choose one; otherwise continue naturally on the topic.\n"
    "Do three things for each user turn:\n"
    "1) FEEDBACK: (reasons in Korean) 자세한 이유/규칙을 한국어로 설명하시오. 간단한 영어 예시는 포함해도 됨.\n"
    "2) CORRECTED_EN: Provide a concise corrected English version of what they tried to say.\n"
    "3) REPLY_EN: Continue the conversation naturally in English (1–2 sentences), referencing context and topic.\n"
    "Output STRICT JSON (no markdown) with keys: {\"corrected_en\", \"feedback\", \"reply_en\", \"meta\"}.\n"
    "meta must be an object with: {\"detected_language\": \"en\"|\"ko\"|\"mixed\", \"topic\": string|null}.\n"
)

def gemini_analyze(transcript: str, chat_history: List[Dict[str, str]], current_topic: Optional[str]) -> Dict[str, str]:
    print("\n=== LLM via Gemini (English tutor analysis) ===")
    model = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)

    history_payload = json.dumps(chat_history[-MAX_HISTORY_TURNS:], ensure_ascii=False)
    user_payload = (
        "current_topic: " + (current_topic or "null") + "\n"
        "chat_history (most recent first is last):\n" + history_payload + "\n\n"
        "User transcript (post-refined):\n" + (transcript or "")
    )

    resp = model.generate_content(
        [{"role": "user", "parts": [{"text": user_payload}]}],
        generation_config={"temperature": 0.6, "top_p": 0.9, "top_k": 40, "max_output_tokens": 640},
    )

    raw = _safe_read_text(resp)
    if not raw:
        # safety fallback
        return {
            "corrected_en": "",
            "feedback": "모델 응답이 제한되어 간단히 이어갈게요. 방금 내용을 한 번 더 명확히 말해줄래요?",
            "reply_en": "Could you please restate that in a short sentence?",
            "meta": "{}",
        }

    data = _robust_json_parse(raw)
    return {
        "corrected_en": data.get("corrected_en") or "",
        "feedback": data.get("feedback") or "",
        "reply_en": data.get("reply_en") or "Could you tell me more in English?",
        "meta": json.dumps(data.get("meta") or {}),
    }
