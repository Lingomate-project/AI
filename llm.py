# llm.py
"""
Gemini 기반 영어 튜터 로직 (대화 맥락 + 주제 관리)

difficulty: "easy" | "medium" | "hard"
register  : "casual" | "formal"
"""

import json
import logging
from typing import List, Dict, Optional, Any

from llm_utils import call_gemini_json

logger = logging.getLogger(__name__)

# =========================
# 1. STT 후처리용 시스템 프롬프트
# =========================

REFINE_SYSTEM = (
    "당신은 ASR(음성 인식) 후처리기입니다. 대화 맥락, 현재 주제, 그리고 N-best 가설 목록을 바탕으로 "
    "사용자의 의도를 가장 잘 반영하는 **자연스러운 영어 문장 1개**를 만들어 주세요.\n"
    "- chat_history: 지금까지의 전체 대화 리스트\n"
    "문법적으로 어색한 조각은 자연스럽게 고쳐 쓰고, 여러 조각이 섞여 있으면 한 문장으로 자연스럽게 합치되 "
    "과도한 의역은 피하세요. 고유명사나 지명은 앞선 대화 맥락에 맞게 선택하세요.\n"
    "출력은 **엄격한 JSON 문자열**만 허용합니다. 형식: {\"refined\": \"...\"} (마크다운/설명 금지)"
)


def refine_transcript(
    raw_transcript: str,
    alt_segments: List[List[str]],
    chat_history: List[Dict[str, str]],
    current_topic: Optional[str],
) -> str:
    """
    STT 결과를 한 문장으로 정제.
    - chat_history 전체를 맥락 힌트로 제공.
    """
    payload = {
        "current_topic": current_topic,
        "chat_history": chat_history,
        "raw_transcript": raw_transcript,
        "alt_segments": alt_segments[:20],
        "instructions": (
            "대안들에서 선택하거나 가볍게 수정하여 자연스러운 영어 **한 문장**으로 합치세요. "
            "주제/맥락에 맞는 고유명사·지명을 우선하세요."
        ),
    }

    data = call_gemini_json(
        system_prompt=REFINE_SYSTEM,
        user_payload=json.dumps(payload, ensure_ascii=False),
        temperature=0.2,
        top_p=0.9,
        max_tokens=640,
    )

    if not data:
        return raw_transcript or ""

    refined = (data.get("refined") or "").strip()
    return refined or (raw_transcript or "")


# =========================
# 2. 회화 튜터용 시스템 프롬프트
# =========================

SYSTEM_PROMPT = (
    "당신은 한국인 영어 학습자를 돕는 전문가 튜터입니다.\n"
    "아래로 전달되는 chat_history(전체)와 current_topic 을 활용해 대화 맥락을 유지하세요.\n"
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
    "출력은 **JSON 문자열**만 허용합니다.\n"
    "형식: {\"corrected_en\": \"...\", \"feedback\": \"...\", \"reply_en\": \"...\", \"meta\": { ... }}\n"
    "meta 객체에는 다음 키가 반드시 포함되어야 합니다:\n"
    "  - \"detected_language\": \"en\" | \"ko\" | \"mixed\"\n"
    "  - \"topic\": string | null\n"
    "  - \"needs_correction\": true | false  # 사용자의 발화가 의미/문법/자연스러움 면에서 수정이 필요한지 여부\n"
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
    - chat_history 전체를 맥락으로 제공
    """
    history_payload = json.dumps(chat_history, ensure_ascii=False)

    settings_block = (
        "settings:\n"
        f"  difficulty: {difficulty}  # easy | medium | hard\n"
        f"  register: {register}      # casual | formal\n"
    )
    user_payload = (
        settings_block
        + "\n"
        + f"current_topic: {current_topic or 'null'}\n"
        + "chat_history:\n" + history_payload + "\n\n"
        + "User transcript (post-refined):\n" + (transcript or "")
    )

    data = call_gemini_json(
        system_prompt=SYSTEM_PROMPT,
        user_payload=user_payload,
        temperature=0.6,
        top_p=0.9,
        max_tokens=640,
    )

    if not data:
        return {
            "corrected_en": "",
            "feedback": "모델 응답이 일시적으로 제한되어 간단히 이어갈게요. 방금 내용을 한 번 더 명확히 말해 줄래요?",
            "reply_en": "Could you please restate that in a short sentence?",
            "meta": "{}",
        }

    return {
        "corrected_en": data.get("corrected_en") or "",
        "feedback": data.get("feedback") or "",
        "reply_en": data.get("reply_en") or "Could you tell me more in English?",
        "meta": json.dumps(data.get("meta") or {}),
    }


# =========================
# 3. 문장 교정 피드백 LLM (버튼용)
# =========================

FEEDBACK_SYSTEM_PROMPT = (
    "당신은 한국인 영어 학습자를 위한 **문장 교정·피드백 도우미**입니다.\n"
    "입력으로 주어지는 것은 학습자가 실제로 말한 영어 문장 한 줄입니다.\n"
    "\n"
    "당신의 역할은 다음 세 가지입니다:\n"
    "1) corrected_en: 학습자가 말하려던 의미를 유지하면서,\n"
    "   더 자연스럽고 문법적으로 올바른 **영어 문장 1개**로 고쳐 주세요.\n"
    "2) reason_ko: 왜 그렇게 고쳤는지, 어떤 문법/어휘 선택이 문제였는지를\n"
    "   한국어로 2~4문장 정도 자세히 설명해 주세요.\n"
    "3) needs_correction: 입력 문장이 교정이 필요한지 여부를 true/false 로 표시하세요.\n"
    "   - 이미 충분히 자연스럽고 문법적으로도 괜찮으면 false 로 두세요.\n"
    "\n"
    "주의사항:\n"
    "- corrected_en은 너무 과도하게 길게 바꾸지 말고, 원래 문장의 의도를 최대한 유지하세요.\n"
    "- 학습자가 이미 자연스럽게 말했으면 needs_correction 을 false 로 설정하고,\n"
    "  corrected_en 은 원문과 거의 같게 두세요.\n"
    "\n"
    "반드시 **JSON 문자열만** 반환해야 합니다. 마크다운/설명 문구는 금지입니다.\n"
    "형식은 정확히 다음과 같습니다:\n"
    "{\n"
    "  \"corrected_en\": \"...\",\n"
    "  \"reason_ko\": \"...\",\n"
    "  \"needs_correction\": true\n"
    "}\n"
)


def generate_feedback(text: str) -> Dict[str, Any]:
    """
    /api/ai/feedback 에서 사용하는 LLM 래퍼 (버튼 기반 문장 교정용).

    입력: 사용자가 실제로 말한 영어 문장(또는 구)
    반환 형태:
      {
        "corrected_en": str,       # 교정된 자연스러운 영어 문장
        "reason_ko": str,          # 왜 그렇게 수정했는지 한국어 설명
        "needs_correction": bool   # 교정이 필요한지 여부
      }
    """
    data = call_gemini_json(
        system_prompt=FEEDBACK_SYSTEM_PROMPT,
        user_payload=text,
        temperature=0.5,
        top_p=0.9,
        max_tokens=512,
    )

    if not data:
        return {}

    corrected_en = (data.get("corrected_en") or "").strip()
    reason_ko = (data.get("reason_ko") or "").strip()
    needs_correction = bool(data.get("needs_correction"))

    return {
        "corrected_en": corrected_en,
        "reason_ko": reason_ko,
        "needs_correction": needs_correction,
    }


# =========================
# 4. 사용자 예시 응답 생성 LLM
# =========================

EXAMPLE_REPLY_SYSTEM_PROMPT = (
    "당신은 한국인 영어 학습자를 위한 회화 튜터입니다.\n"
    "AI가 방금 말한 마지막 영어 문장(ai_last_reply)과 전체 대화 히스토리(chat_history)를 보고,\n"
    "학습자가 이어서 말해볼 수 있는 자연스러운 영어 문장 예시를 **정확히 한 문장**만 제안하세요.\n"
    "\n"
    "응답은 반드시 JSON 형식 문자열로만 반환해야 합니다.\n"
    '형식: {\"reply_example\": \"...\"}\n'
)


def generate_example_reply(
    ai_text: str,
    chat_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    /api/ai/example-reply 에서 사용하는 LLM 래퍼.
    반환 형태:
      { "reply_example": str }
    """
    payload = {
        "chat_history": chat_history,
        "ai_last_reply": ai_text,
    }

    data = call_gemini_json(
        system_prompt=EXAMPLE_REPLY_SYSTEM_PROMPT,
        user_payload=json.dumps(payload, ensure_ascii=False),
        temperature=0.6,
        top_p=0.9,
        max_tokens=256,
    )

    if not data:
        return {}

    example = (data.get("reply_example") or "").strip()
    if not example:
        example = "Let me share my opinion about that."

    return {"reply_example": example}


# =========================
# 5. 사전 엔트리(단어/숙어 설명) LLM
# =========================

DICTIONARY_SYSTEM_PROMPT = (
    "당신은 영어-한국어 이중언어 사전 도우미입니다.\n"
    "하나의 영어 단어 또는 숙어(phrase)를 입력으로 받으면,\n"
    "1) 그 의미를 한국어로 1~2문장으로 간단히 설명하고,\n"
    "2) 그 단어/숙어를 실제로 사용하는 자연스러운 영어 예문을 정확히 2개 만들어 주세요.\n"
    "\n"
    "응답은 반드시 JSON 형식 문자열로만 반환해야 합니다.\n"
    '형식: {\"term\": \"...\", \"meaning_ko\": \"...\", \"examples\": [\"...\", \"...\"]}\n'
)


def generate_dictionary_entry(term: str) -> Dict[str, Any]:
    """
    /api/ai/dictionary 에서 사용하는 LLM 래퍼.
    반환 형태:
      { "term": str, "meaning_ko": str, "examples": List[str 길이=2] }
    """
    data = call_gemini_json(
        system_prompt=DICTIONARY_SYSTEM_PROMPT,
        user_payload=term,
        temperature=0.4,
        top_p=0.9,
        max_tokens=512,
    )

    if not data:
        return {}

    out_term = (data.get("term") or term).strip()
    meaning_ko = (data.get("meaning_ko") or "").strip()
    examples = data.get("examples") or []

    if not isinstance(examples, list):
        examples = [str(examples)]

    examples = [str(e).strip() for e in examples if str(e).strip()]

    # 예문 2개로 정규화
    if len(examples) < 2:
        while len(examples) < 2:
            examples.append(f"I often use the phrase '{out_term}' in my daily conversations.")
    elif len(examples) > 2:
        examples = examples[:2]

    return {
        "term": out_term,
        "meaning_ko": meaning_ko,
        "examples": examples,
    }
