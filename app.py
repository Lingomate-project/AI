# app.py
"""
영어 회화 코치용 FastAPI 서버 (AI 서버 전용)

담당 범위 (LingoMate 명세서 기준):
- 4. AI & 음성 (AI / NLP / Speech) 영역을 담당하는 마이크로서비스

엔드포인트:
- POST /api/ai/chat                  : AI 텍스트 응답 (대화, userId별 히스토리/정확도 관리)
- POST /api/ai/feedback              : 문장 교정 + 교정 이유 제공 (버튼 기반, 자동 교정 아님)
- POST /api/ai/tts                   : 텍스트 → 오디오(wav, base64)  ※ 회화 설정은 백엔드에서 내려줌
- POST /api/ai/example-reply         : AI 응답에 대한 사용자 예시 응답 생성
- POST /api/ai/dictionary            : 사용자 사전 등재용 단어/숙어 설명
- POST /api/stt/recognize            : 오디오(PCM) → STT 텍스트 변환
- GET  /api/stats/accuracy           : 정확도 통계 조회 (userId별)
- GET  /api/conversation/history     : userId별 전체 대화 히스토리 조회

※ 회화 설정(country/style/gender)은 이 서버에서 관리하지 않는다.
   메인 백엔드가 설정값을 조회한 뒤, 이 서버 호출 시 difficulty/register/accent/gender 를 함께 넘겨준다.
"""

import base64
import json
import logging
from typing import List, Dict, Optional, Literal

from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tts import synthesize_wav_bytes
from llm import (
    gemini_analyze,
    generate_feedback,
    generate_example_reply,
    generate_dictionary_entry,
)
from stt import stt_transcribe, SAMPLE_RATE

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FastAPI 앱 설정
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="English Conversation Coach AI Server",
    version="1.0.0",
)

# 필요시 CORS 허용 (프론트/백엔드에서 호출할 수 있도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포 시에는 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────

class ApiResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


def _user_key(user_id: Optional[str]) -> str:
    """멀티 유저 상태 관리를 위한 키. userId 없으면 'anonymous'."""
    return user_id or "anonymous"


# ─────────────────────────────────────────────────────────────
# 전역 대화 상태 + 정확도 통계 (userId 별)
# ─────────────────────────────────────────────────────────────

# userId -> 대화 히스토리 / topic / 정확도 통계
CHAT_HISTORY: Dict[str, List[Dict[str, str]]] = {}
CURRENT_TOPIC: Dict[str, Optional[str]] = {}

ACCURACY_TOTAL_TURNS: Dict[str, int] = {}
ACCURACY_NEEDS_CORRECTION: Dict[str, int] = {}


def _update_accuracy_stats(user_key: str, meta_json_str: str) -> None:
    """
    meta_json_str 안의 meta.needs_correction 값을 보고 통계를 업데이트.
    (ai_chat 안에서만 사용, /api/ai/feedback 과는 무관)
    """
    total = ACCURACY_TOTAL_TURNS.get(user_key, 0) + 1
    ACCURACY_TOTAL_TURNS[user_key] = total

    try:
        meta_obj = json.loads(meta_json_str or "{}")
    except Exception:
        meta_obj = {}

    if isinstance(meta_obj, dict) and meta_obj.get("needs_correction"):
        ACCURACY_NEEDS_CORRECTION[user_key] = ACCURACY_NEEDS_CORRECTION.get(user_key, 0) + 1


def _compute_accuracy_score(user_key: str) -> dict:
    """
    점수 공식:
      score = 100 - (피드백(=수정 필요) / 전체 사용자 응답) * 25

    - 최소 0, 최대 100 으로 클램핑
    - 아직 한 번도 응답이 없으면 100점으로 처리
    """
    total = ACCURACY_TOTAL_TURNS.get(user_key, 0)
    corrected = ACCURACY_NEEDS_CORRECTION.get(user_key, 0)

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
# 새 대화 세션 시작시 상태 초기화
# ─────────────────────────────────────────────────────────────

@app.post("/api/conversation/reset", response_model=ApiResponse)
async def reset_conversation(userId: Optional[str] = Body(default=None)):
    """
    특정 userId(또는 anonymous)에 대한 대화 상태/정확도 통계를 초기화.
    - 프론트/백엔드에서 '새 세션 시작' 버튼 눌렀을 때 호출하면 됨.
    """
    user_key = _user_key(userId)
    CHAT_HISTORY.pop(user_key, None)
    CURRENT_TOPIC.pop(user_key, None)
    ACCURACY_TOTAL_TURNS.pop(user_key, None)
    ACCURACY_NEEDS_CORRECTION.pop(user_key, None)

    return ApiResponse(
        success=True,
        data={"userId": user_key},
    )


# ─────────────────────────────────────────────────────────────
# 0) STT - 오디오(PCM) → 텍스트
#     POST /api/stt/recognize
# ─────────────────────────────────────────────────────────────

class STTRequest(BaseModel):
    """
    audio: base64로 인코딩된 LINEAR16 PCM 모노 오디오
    sampleRate: Hz (생략 시 STT 기본 SAMPLE_RATE 사용)
    """
    audio: str
    sampleRate: Optional[int] = None


@app.post("/api/stt/recognize", response_model=ApiResponse)
async def stt_recognize(req: STTRequest):
    """
    백엔드에서 오디오를 전송받아 STT로 텍스트를 반환하는 엔드포인트.

    요청 예시:
    {
      "audio": "<base64-encoded PCM16 mono>",
      "sampleRate": 48000
    }

    응답 예시:
    {
      "success": true,
      "data": {
        "transcript": "Hello, how are you?",
        "altSegments": [
          ["Hello, how are you?", "Hello, how are u?"],
          ...
        ]
      }
    }
    """
    # 1) base64 → bytes 디코딩
    try:
        pcm_bytes = base64.b64decode(req.audio)
    except Exception as e:
        logger.error("STT decode error: %s", e, exc_info=True)
        return ApiResponse(success=False, error="Invalid base64 audio")

    sr = req.sampleRate or SAMPLE_RATE

    # 2) STT 실행
    try:
        transcript, alt_segments = await stt_transcribe(pcm_bytes, sr=sr)
    except Exception as e:
        logger.error("STT recognize error in endpoint: %s", e, exc_info=True)
        return ApiResponse(success=False, error="STT error")

    return ApiResponse(
        success=True,
        data={
            "transcript": transcript,
            "altSegments": alt_segments,
        },
    )


# ─────────────────────────────────────────────────────────────
# 1) AI 텍스트 응답 - POST /api/ai/chat
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    text: str
    userId: Optional[str] = None

    # 메인 백엔드에서 회화 설정을 기반으로 내려주는 옵션 값들
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    register: Optional[Literal["casual", "formal"]] = None


@app.post("/api/ai/chat", response_model=ApiResponse)
async def ai_chat(req: ChatRequest):
    """
    요청 예시:
    {
      "text": "Can you help me practice English?",
      "userId": "u_123",
      "difficulty": "medium",   // optional, 없으면 medium
      "register": "casual"      // optional, 없으면 casual
    }

    응답:
    {
      "success": true,
      "data": {
        "text": "Sure! What part of English would you like to practice?"
      }
    }
    """
    user_key = _user_key(req.userId)

    # 회화 설정은 메인 백엔드가 관리 → 이 서버는 단지 파라미터만 사용
    difficulty = req.difficulty or "medium"
    register = req.register or "casual"

    transcript = req.text

    # user별 상태 가져오기 (전체 히스토리)
    history = CHAT_HISTORY.setdefault(user_key, [])
    current_topic = CURRENT_TOPIC.get(user_key)

    # gemini_analyze 사용 (내부에서 전체 히스토리 활용)
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
        return ApiResponse(success=False, error="LLM error")

    reply_en = result["reply_en"]
    corrected = result["corrected_en"]

    # meta에서 topic 갱신 + 정확도 통계 업데이트
    meta_str = result.get("meta", "{}")
    try:
        meta_obj = json.loads(meta_str or "{}")
        new_topic = meta_obj.get("topic") if isinstance(meta_obj, dict) else None
        if new_topic:
            CURRENT_TOPIC[user_key] = new_topic
    except Exception as e:
        logger.warning("Failed to parse meta in ai_chat: %s", e)
    _update_accuracy_stats(user_key, meta_str)

    # 전체 대화 히스토리 업데이트 (자르지 않고 append)
    history.append(
        {"user": transcript, "corrected_en": corrected, "reply_en": reply_en}
    )

    return ApiResponse(
        success=True,
        data={"text": reply_en},
    )


# ─────────────────────────────────────────────────────────────
# 2) AI 피드백 - 문장 교정 + 교정 이유 제공
#     POST /api/ai/feedback
# ─────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    text: str  # 사용자가 실제로 말한 영어 문장


@app.post("/api/ai/feedback", response_model=ApiResponse)
async def ai_feedback(req: FeedbackRequest):
    """
    사용자가 '피드백 보기' 버튼을 눌렀을 때 특정 문장에 대한 교정을 요청하는 엔드포인트.

    자연스럽지 않은 문장 예시 응답:
    {
      "success": true,
      "data": {
        "natural": false,
        "corrected_en": "I really like playing soccer.",
        "reason_ko": "왜 이렇게 고쳤는지 한국어 설명..."
      }
    }

    이미 자연스러운 문장 예시 응답:
    {
      "success": true,
      "data": {
        "natural": true,
        "message": "자연스러운 문장이에요!"
      }
    }

    ※ 이 엔드포인트는 정확도 통계(ACCURACY_*)를 전혀 건드리지 않으므로
       자연스러운 문장은 정확도 계산에서도 제외된다.
    """
    data = generate_feedback(req.text)

    if not data:
        return ApiResponse(
            success=False,
            error="LLM response empty",
        )

    needs_correction = bool(data.get("needs_correction"))

    # 이미 자연스러운 문장 → 한 줄 메시지만 반환
    if not needs_correction:
        return ApiResponse(
            success=True,
            data={
                "natural": True,
                "message": "자연스러운 문장이에요!",
            },
        )

    # 교정 필요 → 교정 문장 + 이유 반환
    corrected_en = data.get("corrected_en", "")
    reason_ko = data.get("reason_ko", "")

    return ApiResponse(
        success=True,
        data={
            "natural": False,
            "corrected_en": corrected_en,
            "reason_ko": reason_ko,
        },
    )


# ─────────────────────────────────────────────────────────────
# 3) TTS - 텍스트 → 오디오 (base64 wav)
#     POST /api/ai/tts
# ─────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str

    # 메인 백엔드가 회화 설정으로부터 매핑해서 넘겨주는 값
    # accent: us / uk / au
    accent: Optional[Literal["us", "uk", "au"]] = None

    # gender: male / female  → 내부에서 f/m 으로 변환
    gender: Optional[Literal["male", "female"]] = None


@app.post("/api/ai/tts", response_model=ApiResponse)
async def tts_endpoint(req: TTSRequest):
    """
    요청 예:
      {
        "text": "Hello! How can I help you today?",
        "accent": "uk",          // optional, 없으면 "us"
        "gender": "female"       // optional, 없으면 "female"
      }

    응답:
      {
        "success": true,
        "data": {
          "audio": "<base64-encoded-wav>",
          "mime": "audio/wav"
        }
      }
    """
    # 기본값 적용
    accent = (req.accent or "us").lower()          # us / uk / au
    gender_long = (req.gender or "female").lower() # male / female
    gender_short = "f" if gender_long == "female" else "m"

    try:
        pcm_wav_bytes = synthesize_wav_bytes(
            text=req.text,
            accent=accent,          # "us" / "uk" / "au"
            gender=gender_short,    # "f" / "m"
        )
    except Exception as e:
        logger.error("TTS synth error in endpoint: %s", e, exc_info=True)
        return ApiResponse(success=False, error="TTS error")

    audio_b64 = base64.b64encode(pcm_wav_bytes).decode("ascii")

    return ApiResponse(
        success=True,
        data={
            "audio": audio_b64,
            "mime": "audio/wav",
        },
    )


# ─────────────────────────────────────────────────────────────
# 4) 사용자 응답 예시 생성 - POST /api/ai/example-reply
# ─────────────────────────────────────────────────────────────

class ExampleReplyRequest(BaseModel):
    ai_text: str  # AI가 방금 말한 문장
    userId: Optional[str] = None


@app.post("/api/ai/example-reply", response_model=ApiResponse)
async def ai_example_reply(req: ExampleReplyRequest):
    """
    AI 응답 텍스트와 전체 대화 맥락을 바탕으로,
    사용자가 말해볼 만한 영어 예시 한 문장을 생성.
    난이도/말투는 이 엔드포인트에서 별도로 관리하지 않는다.
    """
    user_key = _user_key(req.userId)

    history = CHAT_HISTORY.get(user_key, [])

    data = generate_example_reply(
        ai_text=req.ai_text,
        chat_history=history,
    )

    if not data:
        return ApiResponse(success=False, error="LLM response empty")

    example = data.get("reply_example", "").strip()
    if not example:
        example = "Let me share my opinion about that."

    return ApiResponse(
        success=True,
        data={"reply_example": example},
    )


# ─────────────────────────────────────────────────────────────
# 5) 사용자 사전 등재용 단어/숙어 설명 - POST /api/ai/dictionary
# ─────────────────────────────────────────────────────────────

class DictionaryEntryRequest(BaseModel):
    term: str  # 영단어 또는 숙어


@app.post("/api/ai/dictionary", response_model=ApiResponse)
async def ai_dictionary(req: DictionaryEntryRequest):
    """
    사용자 사전 등재용 엔드포인트.
    영단어나 숙어 텍스트를 입력받아 한국어 뜻과 예문 2개를 반환.
    """
    data = generate_dictionary_entry(req.term)

    if not data:
        return ApiResponse(success=False, error="LLM response empty")

    term = data.get("term", req.term)
    meaning_ko = data.get("meaning_ko", "")
    examples = data.get("examples", [])

    return ApiResponse(
        success=True,
        data={
            "term": term,
            "meaning_ko": meaning_ko,
            "examples": examples,
        },
    )


# ─────────────────────────────────────────────────────────────
# 6) 정확도 조회 - GET /api/stats/accuracy
# ─────────────────────────────────────────────────────────────

@app.get("/api/stats/accuracy", response_model=ApiResponse)
async def get_accuracy(userId: Optional[str] = Query(default=None)):
    """
    전체 사용자 응답 중 LLM이 '수정 필요(needs_correction)'로 판단한 비율을 기반으로
    100 - (피드백 개수 / 전체 사용자 응답) * 25 점수를 계산해서 반환.
    userId 없으면 'anonymous' 키 기준으로 계산.
    """
    user_key = _user_key(userId)
    stats = _compute_accuracy_score(user_key)
    return ApiResponse(
        success=True,
        data={
            "userId": user_key,
            "totalTurns": stats["total"],
            "correctedTurns": stats["corrected"],
            "accuracy": stats["score"],
        },
    )


# ─────────────────────────────────────────────────────────────
# 7) 전체 대화 히스토리 조회 - GET /api/conversation/history
# ─────────────────────────────────────────────────────────────

@app.get("/api/conversation/history", response_model=ApiResponse)
async def get_conversation_history(userId: Optional[str] = Query(default=None)):
    """
    userId별 전체 대화 히스토리를 반환.
    메인 백엔드에서 '세션 종료' 시 이 엔드포인트를 호출해서
    해당 세션 동안의 전체 대화 로그를 가져갈 수 있다.
    """
    user_key = _user_key(userId)
    history = CHAT_HISTORY.get(user_key, [])
    return ApiResponse(
        success=True,
        data={
            "userId": user_key,
            "history": history,
        },
    )

