# llm_utils.py
import json
import logging
import re
from typing import Any, Dict

from google import generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

# 공통 Gemini 설정
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set. LLM calls will fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)


def safe_read_text(resp) -> str:
    """Gemini 응답에서 text 파트만 안전하게 추출."""
    out = ""
    try:
        if getattr(resp, "candidates", None):
            for c in resp.candidates:
                if getattr(c, "content", None) and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        if getattr(p, "text", None):
                            out += p.text
    except Exception as e:
        logger.error("safe_read_text error: %s", e, exc_info=True)
    return out.strip()


def robust_json_parse(s: str) -> Dict[str, Any]:
    """
    응답이 JSON 하나로 딱 떨어지지 않을 때, 본문 안에서 {...} 블록을 찾아 파싱.
    실패하면 {} 반환.
    """
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


def call_gemini_json(
    *,
    system_prompt: str,
    user_payload: str,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """
    공통 Gemini 호출 래퍼.
    - system_prompt: system_instruction
    - user_payload: user에 들어갈 텍스트 (이미 JSON string 등)
    - JSON 파싱까지 해서 dict로 반환, 실패 시 {} 반환
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY missing in call_gemini_json")
        return {}

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt,
        )
        resp = model.generate_content(
            [{"role": "user", "parts": [{"text": user_payload}]}],
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_tokens,
            },
        )
        raw = safe_read_text(resp)
    except Exception as e:
        logger.error("Gemini call error: %s", e, exc_info=True)
        return {}

    if not raw:
        return {}

    return robust_json_parse(raw)
