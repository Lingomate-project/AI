#stt.py

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
from google.cloud import speech_v2 as speech
from google.oauth2 import service_account
import google.auth
from google.api_core.client_options import ClientOptions


# =========================
# 고정 정책 & 튜닝 값
# =========================
STT_PRIMARY: str = "en-US"              # 인식 언어(영어 고정)
SAMPLE_RATE: int = 48000                # 입력 PCM 샘플레이트(Hz)
N_BEST: int = 8                         # 세그먼트별 대안 후보 개수

RECOGNIZER_LOCATION: str = "asia-northeast1"

# 사용할 STT 모델 ID
STT_MODEL: str = "chirp_3"

# 가벼운 전처리
PREEMPHASIS: float = 0.97               # 프리엠퍼시스 계수
TARGET_RMS_DBFS: float | None = -18     # 대략적 RMS 타깃(dBFS)

# Windows 에서 sounddevice 사용 시 이벤트 루프 정책 설정
if sys.platform.startswith("win"):
    import asyncio as _asyncio
    _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())


# =========================
# 유틸리티
# =========================
def log(title: str) -> None:
    """콘솔에 구분선과 함께 타이틀 로그 출력."""
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def _load_speech_client() -> speech.SpeechClient:
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    client_options = None
    location = RECOGNIZER_LOCATION
    # global 이 아닌 경우 리전 엔드포인트로 접속
    if location and location != "global":
        client_options = ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")

    if cred_path and os.path.exists(cred_path):
        creds = service_account.Credentials.from_service_account_file(cred_path)
        return speech.SpeechClient(credentials=creds, client_options=client_options)

    # ADC 사용
    return speech.SpeechClient(client_options=client_options)


def _project_id_from_adc() -> str:
    """ADC에서 현재 GCP 프로젝트 ID를 해석. 실패 시 예외 발생."""
    project_id = google.auth.default()[1]
    if not project_id:
        raise RuntimeError("ADC/자격증명에서 GCP 프로젝트 ID를 찾지 못했습니다.")
    return project_id


# =========================
# 오디오 전처리
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
# v2 Recognize 구성 (Chirp 3)
# =========================

def _build_recognize_config(sr: int) -> speech.RecognitionConfig:
    decoding = speech.ExplicitDecodingConfig(
        encoding=speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sr,
        audio_channel_count=1,
    )
    features = speech.RecognitionFeatures(
        enable_automatic_punctuation=True,
        max_alternatives=N_BEST
    )
    return speech.RecognitionConfig(
        explicit_decoding_config=decoding,
        language_codes=[STT_PRIMARY], 
        model=STT_MODEL,
        features=features,
    )


async def stt_transcribe(pcm: bytes, sr: int = SAMPLE_RATE) -> Tuple[str, List[List[str]]]:
    log(f"STT (v2 Recognize + {STT_MODEL}) 실행 — language={STT_PRIMARY}, location={RECOGNIZER_LOCATION}")

    def _do_recognize() -> Tuple[str, List[List[str]]]:
        client = _load_speech_client()
        project_id = _project_id_from_adc()
        recognizer = f"projects/{project_id}/locations/{RECOGNIZER_LOCATION}/recognizers/_"

        # 1) 전처리 적용
        pcm_clean = _preprocess_pcm(pcm)

        # 2) Recognize 요청 구성 & 호출
        config = _build_recognize_config(sr)
        request = speech.RecognizeRequest(
            recognizer=recognizer,
            config=config,
            content=pcm_clean,  # v2는 별도 RecognitionAudio 없이 content에 직접 바이트 전달
        )
        response = client.recognize(request=request)

        # 3) 결과 파싱
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
        return transcript, alt_segments

    import asyncio
    return await asyncio.to_thread(_do_recognize)
