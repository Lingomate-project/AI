"""llm.py
Gemini 기반 영어 튜터 로직 (대화 맥락 + 주제 관리)
- .env에서 GEMINI_API_KEY 를 읽어 사용

difficulty: "easy" | "medium" | "hard"
register  : "casual" | "formal"
"""

import os, re, json
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

from google import generativeai as genai

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("로컬 개발 시 .env 파일에 GEMINI_API_KEY 를 설정하세요.")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
MAX_HISTORY_TURNS = 6


def _safe_read_text(resp) -> str:
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


REFINE_SYSTEM = (
    "당신은 ASR(음성 인식) 후처리기입니다. 대화 맥락, 현재 주제, 그리고 N-best 가설 목록을 바탕으로 "
    "사용자의 의도를 가장 잘 반영하는 **자연스러운 영어 문장 1개**를 만들어 주세요. "
    "비문/어색한 조각은 고쳐 쓰고, 여러 조각이 섞여 있으면 자연스럽게 합치되 과도한 의역은 피하세요. "
    "고유명사나 지명은 맥락에 맞는 표기로 선택하세요. "
    '출력은 **엄격한 JSON**만 허용합니다. 형식: {"refined": "..."} (마크다운/설명 금지)'
)


def refine_transcript(
    raw_transcript: str,
    alt_segments: List[List[str]],
    chat_history: List[Dict[str, str]],
    current_topic: Optional[str],
) -> str:
    model = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=REFINE_SYSTEM)

    payload = {
        "current_topic": current_topic,
        "chat_history_tail": chat_history[-MAX_HISTORY_TURNS:],
        "raw_transcript": raw_transcript,
        "alt_segments": alt_segments[:20],
        "instructions": (
            "대안들에서 선택하거나 가볍게 수정하여 자연스러운 영어 **한 문장**으로 합치세요. "
            "주제/맥락에 맞는 고유명사·지명을 우선하세요."
        ),
    }

    resp = model.generate_content(
        [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
        generation_config={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 640,
        },
    )
    raw = _safe_read_text(resp)
    if not raw:
        return raw_transcript or ""

    data = _robust_json_parse(raw)
    refined = (data.get("refined") or "").strip()
    return refined or (raw_transcript or "")


SYSTEM_PROMPT = (
    "당신은 한국인 영어 학습자를 돕는 전문가 튜터입니다.\n"
    "아래로 전달되는 chat_history 와 current_topic 을 활용해 대화 맥락을 유지하세요.\n"
    "만약 current_topic 이 null 이면, 먼저 영어로 3개의 짧은 대화 주제를 제안하고 사용자에게 선택을 요청하세요.\n"
    "이미 주제가 있으면 그 주제로 자연스럽게 이어가세요.\n"
    "\n"
    "매 사용자 발화에 대해 아래 3가지를 수행하세요:\n"
    "1) FEEDBACK: 사용자의 영어에서 어색하거나 틀린 부분이 있으면 **이유/규칙을 한국어로 자세히 설명**하세요.\n"
    "2) CORRECTED_EN: 사용자가 말하려던 내용을 간결하고 자연스러운 **영어 한 문장**으로 수정하여 제시하세요.\n"
    "3) REPLY_EN: 대화가 자연스럽게 이어지도록 **영어 문장**으로 반응하세요.\n"
    "\n"
    "매 요청마다 difficulty 와 register 값이 함께 전달됩니다. 반드시 이를 반영해 문장을 구성하세요.\n"
    "- difficulty:\n"
    "  - \"easy\": 쉬운 어휘, 짧은 문장, 문법 설명도 쉽게.\n"
    "  - \"medium\": 일반 성인 학습자 기준 일상 영어.\n"
    "  - \"hard\": 고급 어휘와 복잡한 문장 구조.\n"
    "- register:\n"
    "  - \"casual\": 친한 친구에게 말하듯, 구어체.\n"
    "  - \"formal\": 정중하고 공손한 톤.\n"
    "\n"
    "출력은 **JSON**만 허용합니다.\n"
    "형식: {\"corrected_en\": \"...\", \"feedback\": \"...\", \"reply_en\": \"...\", \"meta\": { ... }}\n"
    "meta 객체에는 다음 키가 반드시 포함되어야 합니다:\n"
    "  - \"detected_language\": \"en\" | \"ko\" | \"mixed\"\n"
    "  - \"topic\": string | null\n"
)


def gemini_analyze(
    transcript: str,
    chat_history: List[Dict[str, str]],
    current_topic: Optional[str],
    difficulty: str,
    register: str,
) -> Dict[str, str]:
    """
    - difficulty, register는 외부에서 주입
    """
    model = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)

    history_payload = json.dumps(chat_history[-MAX_HISTORY_TURNS:], ensure_ascii=False)
    settings_block = (
        "settings:\n"
        f"  difficulty: {difficulty}  # easy | medium | hard\n"
        f"  register: {register}      # casual | formal\n"
    )
    user_payload = (
        settings_block
        + "\n"
        + "current_topic: " + (current_topic or "null") + "\n"
        + "chat_history (most recent first is last):\n" + history_payload + "\n\n"
        + "User transcript (post-refined):\n" + (transcript or "")
    )

    resp = model.generate_content(
        [{"role": "user", "parts": [{"text": user_payload}]}],
        generation_config={
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 640,
        },
    )

    raw = _safe_read_text(resp)
    if not raw:
        return {
            "corrected_en": "",
            "feedback": "모델 응답이 일시적으로 제한되어 간단히 이어갈게요. 방금 내용을 한 번 더 명확히 말해 줄래요?",
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
