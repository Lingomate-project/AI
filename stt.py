from __future__ import annotations

import os
import sys
from typing import List, Tuple

import logging
import numpy as np
from google.cloud import speech_v2 as speech
from google.api_core.client_options import ClientOptions

from gcp_auth import get_credentials, GCPAuthError

logger = logging.getLogger(__name__)

STT_PRIMARY: str = "en-US"
SAMPLE_RATE: int = 16000          # 16kHz
N_BEST: int = 8

# 위치/프로젝트/리코그나이저 설정
RECOGNIZER_LOCATION: str = os.getenv("STT_LOCATION", "asia-northeast1")

STT_MODEL: str = os.getenv("STT_MODEL", "chirp_3")

RECOGNIZER_ID: str = "_"

# ====== (전처리 관련, 지금은 사용 안 함) ======
PREEMPHASIS: float = 0.97
TARGET_RMS_DBFS: float | None = -18


class STTError(Exception):
    """STT 처리 단계 에러 (어디서 터졌는지 알려주는용)."""

    def __init__(self, stage: str, detail: str):
        self.stage = stage      # 예: "auth", "client", "input", "preprocess", "request", "api", "parse"
        self.detail = detail
        super().__init__(f"[{stage}] {detail}")


def _get_speech_client() -> speech.SpeechClient:
    """SpeechClient 생성 (에러 시 STTError로 감싸서 던짐)."""
    try:
        creds = get_credentials()
    except GCPAuthError as e:
        logger.error("STT auth error: %s", e, exc_info=True)
        raise STTError(f"auth/{e.stage}", e.detail)

    try:
        # location에 따라 endpoint 결정
        if RECOGNIZER_LOCATION == "global":
            api_endpoint = "speech.googleapis.com"
        else:
            api_endpoint = f"{RECOGNIZER_LOCATION}-speech.googleapis.com"

        logger.info("STT api_endpoint = %s", api_endpoint)

        client_options = ClientOptions(api_endpoint=api_endpoint)
        client = speech.SpeechClient(
            credentials=creds,
            client_options=client_options,
        )
    except Exception as e:
        logger.error("STT client init failed: %s", e, exc_info=True)
        raise STTError("client", f"SpeechClient 생성 실패: {e}")

    return client


def _pre_emphasis(signal: np.ndarray, coef: float = PREEMPHASIS) -> np.ndarray:
    if coef is None or coef == 0:
        return signal
    return np.append(signal[0], signal[1:] - coef * signal[:-1])


def _rms_dbfs(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(x), axis=-1))
    return 20 * np.log10(rms / 32768.0 + 1e-12)


def _gain_to_target_dbfs(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    if target_dbfs is None:
        return x

    current = _rms_dbfs(x)
    diff_db = target_dbfs - current
    gain = 10 ** (diff_db / 20.0)
    return x * gain


async def stt_transcribe(
    pcm_data: bytes,
    sr: int = SAMPLE_RATE,
    language_code: str = STT_PRIMARY,
    n_best: int = N_BEST,
) -> Tuple[str, List[str]]:
    """
    16bit mono PCM 바이트를 받아 STT 수행.
    에러 발생 시 STTError(stage, detail)를 던진다.

    반환:
      - best_text: 가장 확률 높은 transcript
      - nbest_texts: 대안 transcript 리스트
    """
    if not pcm_data:
        raise STTError("input", "PCM 데이터가 비어 있습니다.")

    # 1) numpy 변환
    try:
        audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    except Exception as e:
        raise STTError("input", f"PCM → numpy 변환 실패: {e}")

    logger.info(
        "STT input pcm: samples=%d, min=%.1f, max=%.1f, mean=%.3f",
        audio_np.shape[0],
        float(audio_np.min()) if audio_np.size > 0 else 0.0,
        float(audio_np.max()) if audio_np.size > 0 else 0.0,
        float(audio_np.mean()) if audio_np.size > 0 else 0.0,
    )

    # 2) 전처리 없이 원본 그대로 사용
    processed_bytes = pcm_data

    # 3) 클라이언트 & recognizer 경로
    client = _get_speech_client()

    project_id = os.getenv("GCP_PROJECT_ID", "_")
    recognizer = f"projects/{project_id}/locations/{RECOGNIZER_LOCATION}/recognizers/{RECOGNIZER_ID}"
    logger.info("STT recognizer path = %s", recognizer)
    logger.info("STT model = %s, language = %s, sr = %d", STT_MODEL, language_code, sr)

    # 4) explicit_decoding_config 로 RAW PCM 형식 명시
    try:
        config = speech.RecognitionConfig(
            explicit_decoding_config=speech.ExplicitDecodingConfig(
                encoding=speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sr,
                audio_channel_count=1,
            ),
            language_codes=[language_code],
            model=STT_MODEL,  
            features=speech.RecognitionFeatures(
                enable_word_time_offsets=False,
                enable_automatic_punctuation=True,
            ),
        )

        request = speech.RecognizeRequest(
            recognizer=recognizer,
            config=config,
            content=processed_bytes,
        )
    except Exception as e:
        raise STTError("request", f"RecognizeRequest 구성 실패: {e}")

    logger.info(
        "Calling STT recognize: recognizer=%s, lang=%s, n_best=%d, sr=%d",
        recognizer,
        language_code,
        n_best,
        sr,
    )

    # 5) API 호출
    try:
        response = client.recognize(request)
    except Exception as e:
        logger.error("STT recognize() API 호출 실패: %s", e, exc_info=True)
        raise STTError("api", f"STT API 호출 실패: {e}")

    # 6) 응답 파싱 + 디버그용 전체 로그
    try:
        nbest_texts: List[str] = []
        debug_results = []

        for r_idx, result in enumerate(response.results):
            for a_idx, alt in enumerate(result.alternatives[:n_best]):
                debug_results.append(
                    {
                        "result_index": r_idx,
                        "alt_index": a_idx,
                        "transcript": alt.transcript,
                        "confidence": alt.confidence,
                    }
                )
                # 공백만 있는 transcript 는 제외
                if alt.transcript and alt.transcript.strip():
                    nbest_texts.append(alt.transcript.strip())

        logger.info("STT raw alternatives: %s", debug_results)

    except Exception as e:
        raise STTError("parse", f"STT 응답 파싱 실패: {e}")

    if not nbest_texts:
        logger.warning("STT empty transcript. Raw results: %s", debug_results)
        raise STTError("parse", "STT 결과가 비어 있습니다.")

    best_text = nbest_texts[0]
    return best_text, nbest_texts


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("SAMPLE_RATE = %d", SAMPLE_RATE)
    logger.info("GCP_PROJECT_ID = %s", os.getenv("GCP_PROJECT_ID"))
    logger.info("STT_LOCATION = %s", RECOGNIZER_LOCATION)
    logger.info("STT_MODEL = %s", STT_MODEL)
