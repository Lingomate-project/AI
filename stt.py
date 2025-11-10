"""stt.py
Batch STT (English-only) using Google Cloud Speech **v2**.

Return:
  (transcript, alt_segments)
    - transcript: top-1 후보들을 이어 붙인 최종 문자열
    - alt_segments: 각 세그먼트의 N-best 후보 문자열 리스트들
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
from google.cloud import speech_v2 as speech
from google.oauth2 import service_account
import google.auth

# =========================
# Fixed policy & tunables
# =========================
STT_PRIMARY: str = "en-US"   # 영어만 인식 (원하면 "en-US"로 A/B 테스트)
SAMPLE_RATE: int = 16000
N_BEST: int = 8
RECOGNIZER_LOCATION: str = "global"

# 가벼운 전처리(원치 않으면 coeff=0.0, target=None)
PREEMPHASIS: float = 0.97         # 0 = off, 0.95~0.98 권장
TARGET_RMS_DBFS: float | None = -20  # 레벨이 너무 작/큰 경우만 ±12 dB 내에서 보정

# Windows event loop policy (sounddevice 대비)
if sys.platform.startswith("win"):
    import asyncio as _asyncio
    _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())


# =========================
# Utilities
# =========================
def log(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def _load_speech_client() -> speech.SpeechClient:
    """Prefer GOOGLE_APPLICATION_CREDENTIALS (service account); else ADC."""
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and os.path.exists(cred_path):
        creds = service_account.Credentials.from_service_account_file(cred_path)
        return speech.SpeechClient(credentials=creds)
    return speech.SpeechClient()


def _project_id_from_adc() -> str:
    project_id = google.auth.default()[1]
    if not project_id:
        raise RuntimeError("Failed to resolve GCP project id from ADC/credentials.")
    return project_id


# =========================
# Audio pre-processing
# =========================
def _preemphasis(x: np.ndarray, coeff: float) -> np.ndarray:
    if not (0.0 < coeff < 1.0):
        return x
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    xf = x.astype(np.float32)
    y[1:] = xf[1:] - coeff * xf[:-1]
    y = np.clip(y, -32768, 32767).astype(np.int16)
    return y


def _rms_normalize(x: np.ndarray, target_dbfs: float | None) -> np.ndarray:
    if target_dbfs is None:
        return x
    xf = x.astype(np.float32)
    rms = float(np.sqrt(np.mean(xf * xf) + 1e-9))
    if rms <= 1e-6:
        return x
    cur_db = 20.0 * np.log10(rms / 32767.0 + 1e-12)
    gain_db = max(-12.0, min(12.0, target_dbfs - cur_db))
    g = 10 ** (gain_db / 20.0)
    y = np.clip(xf * g, -32768, 32767).astype(np.int16)
    return y


def _preprocess_pcm(pcm: bytes) -> bytes:
    if (PREEMPHASIS <= 0.0) and (TARGET_RMS_DBFS is None):
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16)
    if PREEMPHASIS > 0.0:
        x = _preemphasis(x, PREEMPHASIS)
    x = _rms_normalize(x, TARGET_RMS_DBFS)
    return x.tobytes()


# =========================
# v2 Recognize config
# =========================
def _build_recognize_config(sr: int) -> speech.RecognitionConfig:
    decoding = speech.ExplicitDecodingConfig(
        encoding=speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sr,
        audio_channel_count=1,
    )
    features = speech.RecognitionFeatures(
        enable_automatic_punctuation=True,
        max_alternatives=N_BEST,
    )
    return speech.RecognitionConfig(
        explicit_decoding_config=decoding,
        language_codes=[STT_PRIMARY],
        model="latest_short",  # 길게 말하면 "latest_long"으로 바꿔서 A/B
        features=features,
    )


# =========================
# Public API
# =========================
async def stt_transcribe(pcm: bytes, sr: int = SAMPLE_RATE) -> Tuple[str, List[List[str]]]:
    """
    Batch recognize in **English only** (v2 Recognize API).
    Args:
        pcm: raw PCM16 mono audio bytes.
        sr : sample rate (Hz), 실제 오디오와 일치해야 함.

    Returns:
        transcript: str
        alt_segments: List[List[str]] — 각 세그먼트의 N-best 후보.
    """
    log(f"STT via Google Cloud Speech-to-Text v2 (batch recognize, language={STT_PRIMARY})")

    def _do_recognize() -> Tuple[str, List[List[str]]]:
        client = _load_speech_client()
        project_id = _project_id_from_adc()
        recognizer = f"projects/{project_id}/locations/{RECOGNIZER_LOCATION}/recognizers/_"

        # 전처리
        pcm_clean = _preprocess_pcm(pcm)

        # Recognize 요청
        config = _build_recognize_config(sr)
        request = speech.RecognizeRequest(
            recognizer=recognizer,
            config=config,
            content=pcm_clean,   # v2는 별도 RecognitionAudio 없이 content 필드에 직접 바이트 전달
        )
        response = client.recognize(request=request)

        # 결과 파싱
        alt_segments: List[List[str]] = []
        top_pieces: List[str] = []
        for result in response.results:
            if not result.alternatives:
                continue
            alts = [a.transcript.strip() for a in result.alternatives if a.transcript]
            if not alts:
                continue
            alt_segments.append(alts)
            top_pieces.append(alts[0])

        transcript = " ".join(p for p in top_pieces if p).strip()
        print(f"→ TRANSCRIPT(top-1): {transcript or '[EMPTY]'}")
        return transcript, alt_segments

    import asyncio
    return await asyncio.to_thread(_do_recognize)
