"""
- 외부에서 accent, gender 값을 받아 목소리를 선택.
- 억양: us / uk / au
- 성별: f(여성) / m(남성)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Final, Tuple, Optional

import logging
from google.cloud import texttospeech as tts

logger = logging.getLogger(__name__)

# 억양/성별 조합에 따른 보이스 이름 맵핑
VOICE_MAP = {
    ("us", "f"): "en-US-Chirp-HD-F",
    ("us", "m"): "en-US-Chirp-HD-M",
    ("uk", "f"): "en-GB-Neural2-A",
    ("uk", "m"): "en-GB-Neural2-D",
    ("au", "f"): "en-AU-Neural2-A",
    ("au", "m"): "en-AU-Neural2-B",
}

CHIRP_RECOMMENDED_SR: Final[int] = 24_000
SILENCE_TAIL_SEC: Final[float] = 0.25
WAV_CHANNELS: Final[int] = 1
WAV_SAMPWIDTH_BYTES: Final[int] = 2


def _lang_code_from_voice(name: str, fallback: str = "en-US") -> str:
    parts = name.split("-")
    if len(parts) >= 2 and len(parts[0]) == 2 and len(parts[1]) == 2:
        return f"{parts[0]}-{parts[1]}"
    return fallback


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _play_pcm16(pcm: bytes, sr: int) -> None:
    """
    서버(Elastic Beanstalk) 환경에서는 로컬 스피커 재생을 하지 않으므로 NO-OP.
    """
    logger.info("Skipping local audio playback (no sounddevice in server environment).")
    return


def _save_wav(pcm: bytes, path: str, sr: int) -> None:
    import wave

    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(WAV_CHANNELS)
        wf.setsampwidth(WAV_SAMPWIDTH_BYTES)
        wf.setframerate(sr)
        wf.writeframes(pcm)


def _ensure_sentence_final_punct(text: str) -> str:
    text = (text or "").strip()
    if text and text[-1] not in ".!?":
        return text + "."
    return text


def _append_silence_tail(pcm: bytes, sr: int, seconds: float = SILENCE_TAIL_SEC) -> bytes:
    tail_bytes = int(sr * seconds) * WAV_SAMPWIDTH_BYTES
    return pcm + (b"\x00" * tail_bytes)


def _select_voice(accent: str, gender: str) -> Tuple[str, str]:
    accent = (accent or "us").lower()
    gender = (gender or "f").lower()
    if accent not in ("us", "uk", "au"):
        accent = "us"
    if gender not in ("f", "m"):
        gender = "f"

    voice_name = VOICE_MAP.get((accent, gender), VOICE_MAP[("us", "f")])
    lang_code = _lang_code_from_voice(voice_name, fallback="en-US")
    return voice_name, lang_code


def speak_reply_en(
    text: str,
    sr: int = 16_000,
    output_dir: str = "outputs",
    accent: str = "us",
    gender: str = "f",
) -> str:
    """
    로컬 테스트용: TTS 재생 + wav 파일 저장.
    (서버에서는 _play_pcm16이 NO-OP이므로 파일만 저장됨)
    """
    voice_name, lang_for_tts = _select_voice(accent, gender)

    if "Chirp" in voice_name and sr < CHIRP_RECOMMENDED_SR:
        sr = CHIRP_RECOMMENDED_SR

    text = _ensure_sentence_final_punct(text)

    logger.info("TTS (voice=%s, pcm@%d, lang=%s)", voice_name, sr, lang_for_tts)

    client = tts.TextToSpeechClient()

    synthesis_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(name=voice_name, language_code=lang_for_tts)
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, sample_rate_hertz=sr)

    try:
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )
    except Exception as e:
        logger.error("TTS synth error: %s", e, exc_info=True)
        raise

    pcm = _append_silence_tail(response.audio_content, sr, seconds=SILENCE_TAIL_SEC)

    _play_pcm16(pcm, sr)
    out_path = os.path.abspath(os.path.join(output_dir, f"reply_{_timestamp()}.wav"))
    _save_wav(pcm, out_path, sr)
    logger.info("TTS saved: %s", out_path)
    return out_path


# ========= FastAPI용 공통 TTS 함수 =========

_tts_client: Optional[tts.TextToSpeechClient] = None


def _get_tts_client() -> tts.TextToSpeechClient:
    global _tts_client
    if _tts_client is None:
        _tts_client = tts.TextToSpeechClient()
    return _tts_client


def synthesize_wav_bytes(
    text: str,
    accent: str = "us",
    gender: str = "f",
    sr: int = 16_000,
) -> bytes:
    """
    FastAPI 엔드포인트에서 사용하는 TTS 합성 함수.
    - 입력: text, accent(us/uk/au), gender(f/m), sample rate
    - 출력: WAV 포맷의 바이트
    """
    voice_name, lang_for_tts = _select_voice(accent, gender)

    if "Chirp" in voice_name and sr < CHIRP_RECOMMENDED_SR:
        sr = CHIRP_RECOMMENDED_SR

    text = _ensure_sentence_final_punct(text)

    client = _get_tts_client()

    synthesis_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(name=voice_name, language_code=lang_for_tts)
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        sample_rate_hertz=sr,
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    pcm = _append_silence_tail(response.audio_content, sr, seconds=SILENCE_TAIL_SEC)

    # WAV 래핑
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(WAV_CHANNELS)
        wf.setsampwidth(WAV_SAMPWIDTH_BYTES)
        wf.setframerate(sr)
        wf.writeframes(pcm)

    return buf.getvalue()
