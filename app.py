#영어 회화 코치용 FastAPI 서버 (AI 서버 전용)


import base64
import json
import logging
from typing import List, Dict, Optional, Literal, Any

from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from tts import synthesize_pcm_bytes, TTSError
from llm import (
    gemini_analyze,
    generate_feedback,
    generate_example_reply,
    generate_session_review_vocab,
)
from stt import stt_transcribe, SAMPLE_RATE, STTError

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 공통 응답 유틸 (ok / err)
# ─────────────────────────────────────────────────────────────

def ok(data: Any) -> Dict[str, Any]:
    """
    성공 응답 공통 포맷 (ApiResponse)

    {
      "success": true,
      "data": { ... },
      "error": null
    }
    """
    return {
        "success": True,
        "data": data,
        "error": None,
    }


def err(message: str, data: Optional[Any] = None) -> Dict[str, Any]:
    """
    오류 응답 공통 포맷 (ApiResponse)

    {
      "success": false,
      "data": { ... }  # 없으면 {}
      "error": "에러 메시지"
    }
    """
    return {
        "success": False,
        "data": data if data is not None else {},
        "error": message,
    }


# ─────────────────────────────────────────────────────────────
# FastAPI 앱 설정
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="English Conversation Coach AI Server",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# 헬스 체크 / 루트
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return ok({"ok": True})


@app.get("/")
def root():
    return ok({"service": "English Conversation Coach AI Server", "status": "ok"})


# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────

def _session_key(session_id: Optional[str]) -> str:
    """멀티 세션 상태 관리를 위한 키. sessionId 없으면 'anonymous'."""
    return session_id or "anonymous"


# ─────────────────────────────────────────────────────────────
# 전역 대화 상태 + 정확도 통계 (sessionId 별)
# ─────────────────────────────────────────────────────────────

# sessionId -> 대화 히스토리 / topic / 정확도 통계
CHAT_HISTORY: Dict[str, List[Dict[str, str]]] = {}
CURRENT_TOPIC: Dict[str, Optional[str]] = {}

ACCURACY_TOTAL_TURNS: Dict[str, int] = {}
ACCURACY_NEEDS_CORRECTION: Dict[str, int] = {}

FEEDBACK_HISTORY: Dict[str, List[Dict[str, str]]] = {}


def _update_accuracy_stats(session_key: str, meta_json_str: str) -> None:
    """
    meta_json_str 안의 meta.needs_correction 값을 보고 통계를 업데이트.
    (ai_chat 안에서만 사용, /api/ai/feedback 과는 무관)
    """
    total = ACCURACY_TOTAL_TURNS.get(session_key, 0) + 1
    ACCURACY_TOTAL_TURNS[session_key] = total

    try:
        meta_obj = json.loads(meta_json_str or "{}")
    except Exception:
        meta_obj = {}

    if isinstance(meta_obj, dict) and meta_obj.get("needs_correction"):
        ACCURACY_NEEDS_CORRECTION[session_key] = ACCURACY_NEEDS_CORRECTION.get(session_key, 0) + 1


def _compute_accuracy_score(session_key: str) -> dict:
    """
    점수 공식:
      score = 100 - (피드백(=수정 필요) / 전체 사용자 응답) * 25

    - 최소 0, 최대 100 으로 클램핑
    - 아직 한 번도 응답이 없으면 100점으로 처리
    """
    total = ACCURACY_TOTAL_TURNS.get(session_key, 0)
    corrected = ACCURACY_NEEDS_CORRECTION.get(session_key, 0)

    if total <= 0:
        return {"total": 0, "corrected": 0, "score": 100.0}

    ratio = corrected / total
    score = 100.0 - ratio * 25.0
    if score < 0.0:
        score = 0.0
    if score > 100.0:
        score = 100.0
    return {"total": total, "corrected": corrected, "score": score}


# ─────────────────────────────────────────────────────────────
# 0) STT - 오디오(PCM) → 텍스트
#     POST /api/stt/recognize
# ─────────────────────────────────────────────────────────────

@app.post("/api/stt/recognize")
async def stt_recognize(
    pcm: bytes = Body(..., media_type="application/octet-stream"),
):
    """
    16kHz, mono, LINEAR16 PCM 바이트를 그대로 받아서 STT를 수행하는 엔드포인트.
    """
    logger.info("STT request received: %d bytes (16kHz PCM)", len(pcm))

    try:
        transcript, alt_candidates = await stt_transcribe(pcm, sr=SAMPLE_RATE)
    except STTError as e:
        logger.error(
            "STTError in endpoint: stage=%s, detail=%s",
            e.stage,
            e.detail,
            exc_info=True,
        )
        return err(
            "STT error",
            data={
                "stage": e.stage,
                "detail": e.detail,
            },
        )
    except Exception as e:
        logger.error("Unknown STT error in endpoint: %s", e, exc_info=True)
        return err(
            "STT error",
            data={
                "stage": "unknown",
                "detail": str(e),
            },
        )

    return ok(
        {
            "transcript": transcript,
        }
    )


# ─────────────────────────────────────────────────────────────
# 1) AI 텍스트 응답 - POST /api/ai/chat
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    text: str
    sessionId: Optional[str] = None

    # 메인 백엔드에서 회화 설정을 기반으로 내려주는 옵션 값들
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    register: Optional[Literal["casual", "formal"]] = None


@app.post("/api/ai/chat")
async def ai_chat(req: ChatRequest):
    """
    요청 예시:
    {
      "text": "Can you help me practice English?",
      "sessionId": "s_123",
      "difficulty": "medium",   // optional, 없으면 medium
      "register": "casual"      // optional, 없으면 casual
    }
    """
    session_key = _session_key(req.sessionId)

    difficulty = req.difficulty or "medium"
    register = req.register or "casual"

    transcript = req.text

    # session별 상태 가져오기 (전체 히스토리)
    history = CHAT_HISTORY.setdefault(session_key, [])
    current_topic = CURRENT_TOPIC.get(session_key)

    # gemini_analyze 사용
    try:
        result = gemini_analyze(
            transcript=transcript,
            chat_history=history,
            current_topic=current_topic,
            difficulty=difficulty,
            register=register,
        )
    except Exception as e:
        logger.error("ai_chat LLM error: %s", e, exc_info=True)
        return err("LLM error")

    reply_en = result["reply_en"]
    corrected = result["corrected_en"]

    # meta에서 topic 갱신 + 정확도 통계 업데이트
    meta_str = result.get("meta", "{}")
    try:
        meta_obj = json.loads(meta_str or "{}")
        new_topic = meta_obj.get("topic") if isinstance(meta_obj, dict) else None
        if new_topic:
            CURRENT_TOPIC[session_key] = new_topic
    except Exception as e:
        logger.warning("Failed to parse meta in ai_chat: %s", e)
    _update_accuracy_stats(session_key, meta_str)

    # 전체 대화 히스토리 업데이트 (자르지 않고 append)
    history.append(
        {"user": transcript, "corrected_en": corrected, "reply_en": reply_en}
    )

    return ok({"text": reply_en})


# ─────────────────────────────────────────────────────────────
# 2) AI 피드백 - 문장 교정 + 교정 이유 제공
#     POST /api/ai/feedback
# ─────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    text: str               # 사용자가 실제로 말한 영어 문장
    sessionId: Optional[str] = None  # 어떤 세션에서 받은 피드백인지


@app.post("/api/ai/feedback")
async def ai_feedback(req: FeedbackRequest):
    """
    사용자가 '피드백 보기' 버튼을 눌렀을 때 특정 문장에 대한 교정을 요청하는 엔드포인트.
    """
    session_key = _session_key(req.sessionId)

    data = generate_feedback(req.text)

    if not data:
        return err("LLM response empty")

    needs_correction = bool(data.get("needs_correction"))

    # 이미 자연스러운 문장 → 한 줄 메시지만 반환 (히스토리에는 저장 안 함)
    if not needs_correction:
        return ok(
            {
                "natural": True,
                "message": "자연스러운 문장이에요!",
            }
        )

    # 교정 필요 → 교정 문장 + 이유 반환
    corrected_en = data.get("corrected_en", "")
    reason_ko = data.get("reason_ko", "")

    # 해당 세션의 피드백 히스토리에 기록
    feedback_list = FEEDBACK_HISTORY.setdefault(session_key, [])
    feedback_list.append(
        {
            "original": req.text,
            "corrected_en": corrected_en,
            "reason_ko": reason_ko,
        }
    )

    return ok(
        {
            "natural": False,
            "corrected_en": corrected_en,
            "reason_ko": reason_ko,
        }
    )


# ─────────────────────────────────────────────────────────────
# 3) TTS - 텍스트 → 오디오 (base64 pcm)
#     POST /api/ai/tts
# ─────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str

    # 메인 백엔드가 회화 설정으로부터 매핑해서 넘겨주는 값
    # accent: us / uk / au
    accent: Optional[Literal["us", "uk", "au"]] = None

    # gender: male / female  → 내부에서 f/m 으로 변환
    gender: Optional[Literal["male", "female"]] = None


@app.post("/api/ai/tts")
async def tts_endpoint(req: TTSRequest):
    """
    텍스트를 TTS로 변환해 base64-encoded PCM(LINEAR16)을 반환.
    """
    accent = (req.accent or "us").lower()
    gender_long = (req.gender or "female").lower()
    gender_short = "f" if gender_long == "female" else "m"

    try:
        pcm_bytes, sample_rate = synthesize_pcm_bytes(
            text=req.text,
            accent=accent,
            gender=gender_short,
            sr=16_000,
        )
    except TTSError as e:
        logger.error(
            "TTSError in /api/ai/tts: stage=%s, detail=%s",
            e.stage,
            e.detail,
            exc_info=True,
        )
        return err(
            "TTS error",
            data={
                "stage": e.stage,
                "detail": e.detail,
            },
        )
    except Exception as e:
        logger.error("Unknown TTS error in /api/ai/tts: %s", e, exc_info=True)
        return err(
            "TTS error",
            data={
                "stage": "unknown",
                "detail": str(e),
            },
        )

    audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")

    return ok(
        {
            "audio": audio_b64,
            "sampleRate": sample_rate,
            "encoding": "LINEAR16",
        }
    )


@app.post("/api/ai/tts/pcm", response_class=Response)
async def tts_pcm_endpoint(req: TTSRequest):
    """
    텍스트를 TTS로 변환해 순수 PCM(LINEAR16) 바이트 스트림으로 반환.
    """
    accent = (req.accent or "us").lower()
    gender_long = (req.gender or "female").lower()
    gender_short = "f" if gender_long == "female" else "m"

    try:
        pcm_bytes, sample_rate = synthesize_pcm_bytes(
            text=req.text,
            accent=accent,
            gender=gender_short,
            sr=16_000,
        )
    except TTSError as e:
        logger.error(
            "TTSError in /api/ai/tts/pcm: stage=%s, detail=%s",
            e.stage,
            e.detail,
            exc_info=True,
        )
        msg = f"TTS error [{e.stage}] {e.detail}"
        return Response(
            content=msg.encode("utf-8"),
            status_code=500,
            media_type="text/plain; charset=utf-8",
        )
    except Exception as e:
        logger.error("Unknown TTS error in /api/ai/tts/pcm: %s", e, exc_info=True)
        msg = f"TTS error [unknown] {e}"
        return Response(
            content=msg.encode("utf-8"),
            status_code=500,
            media_type="text/plain; charset=utf-8",
        )

    headers = {
        "X-Audio-Sample-Rate": str(sample_rate),
        "X-Audio-Encoding": "LINEAR16",
    }

    return Response(
        content=pcm_bytes,
        media_type="application/octet-stream",
        headers=headers,
    )


# ─────────────────────────────────────────────────────────────
# 4) 사용자 응답 예시 생성 - POST /api/ai/example-reply
# ─────────────────────────────────────────────────────────────

class ExampleReplyRequest(BaseModel):
    ai_text: str  # AI가 방금 말한 문장
    sessionId: Optional[str] = None


@app.post("/api/ai/example-reply")
async def ai_example_reply(req: ExampleReplyRequest):
    """
    AI 응답 텍스트와 전체 대화 맥락을 바탕으로,
    사용자가 말해볼 만한 영어 예시 한 문장을 생성.
    난이도/말투는 이 엔드포인트에서 별도로 관리하지 않는다.
    """
    session_key = _session_key(req.sessionId)

    history = CHAT_HISTORY.get(session_key, [])

    data = generate_example_reply(
        ai_text=req.ai_text,
        chat_history=history,
    )

    if not data:
        return err("LLM response empty")

    example = data.get("reply_example", "").strip()
    if not example:
        example = "Let me share my opinion about that."

    return ok({"reply_example": example})


# ─────────────────────────────────────────────────────────────
# 5) 세션 복습 - POST /api/ai/review
# ─────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    sessionId: Optional[str] = None
    maxItems: Optional[int] = 5


@app.post("/api/ai/review")
async def ai_review(req: ReviewRequest):
    """
    세션 기준 복습 엔드포인트.
    """
    session_key = _session_key(req.sessionId)
    max_items = req.maxItems or 5

    # 1) 세션 전체 대화 (AI가 사용한 reply_en 포함)
    chat_history = CHAT_HISTORY.get(session_key, [])

    # 2) LLM으로부터 '어려운 단어/숙어' 리스트 받기
    vocab_data = generate_session_review_vocab(
        chat_history=chat_history,
        max_items=max_items,
    )
    raw_items = vocab_data.get("items", [])

    # 3) term + meaning_ko만 추려서 반환
    vocab_items = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        term = item.get("term")
        meaning_ko = item.get("meaning_ko")
        if not term or not meaning_ko:
            continue
        vocab_items.append(
            {
                "term": term,
                "meaning_ko": meaning_ko,
            }
        )

    return ok(
        {
            "sessionId": session_key,
            "vocabulary": vocab_items,
        }
    )


# ─────────────────────────────────────────────────────────────
# 6) 정확도 조회 - GET /api/stats/accuracy
# ─────────────────────────────────────────────────────────────

@app.get("/api/stats/accuracy")
async def get_accuracy(sessionId: Optional[str] = Query(default=None)):
    """
    전체 사용자 응답 중 LLM이 '수정 필요(needs_correction)'로 판단한 비율을 기반으로
    100 - (피드백 개수 / 전체 사용자 응답) * 25 점수를 계산해서 반환.
    sessionId 없으면 'anonymous' 키 기준으로 계산.
    """
    session_key = _session_key(sessionId)
    stats = _compute_accuracy_score(session_key)
    return ok(
        {
            "sessionId": session_key,
            "accuracy": stats["score"],
        }
    )


# ─────────────────────────────────────────────────────────────
# 7) 전체 대화 히스토리 조회 - GET /api/conversation/history
# ─────────────────────────────────────────────────────────────

@app.get("/api/conversation/history")
async def get_conversation_history(sessionId: Optional[str] = Query(default=None)):
    """
    sessionId별 전체 대화 히스토리를 반환하고,
    그 즉시 해당 세션의 상태(CHAT_HISTORY, CURRENT_TOPIC, 정확도, 피드백)를 초기화한다.

    - 메인 백엔드에서 '세션 종료' 시 이 엔드포인트를 호출하면
      1) 전체 대화 로그를 받아서 저장하고
      2) AI 서버 쪽 세션 상태는 자동으로 reset 된다.

    변경 사항:
      - 응답 바디에는 sessionId와 간단한 history만 포함한다.
      - history 항목에는 user 발화와 AI 응답(reply_en)만 포함한다.
        (corrected_en, feedback, accuracy 등은 응답에서 제외)
    """
    session_key = _session_key(sessionId)

    raw_history = CHAT_HISTORY.get(session_key, [])

    # user + reply_en만 노출
    history_simple: List[Dict[str, str]] = []
    for turn in raw_history:
        if not isinstance(turn, dict):
            continue
        history_simple.append(
            {
                "user": turn.get("user", ""),
                "reply_en": turn.get("reply_en", ""),
            }
        )

    response_body = ok(
        {
            "sessionId": session_key,
            "history": history_simple,
        }
    )

    # 세션 상태 초기화 (응답에는 안 나오지만 메모리 정리)
    CHAT_HISTORY.pop(session_key, None)
    CURRENT_TOPIC.pop(session_key, None)
    ACCURACY_TOTAL_TURNS.pop(session_key, None)
    ACCURACY_NEEDS_CORRECTION.pop(session_key, None)
    FEEDBACK_HISTORY.pop(session_key, None)

    return response_body
