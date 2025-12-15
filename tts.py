from __future__ import annotations

import logging
from typing import Optional, Literal, Tuple, Dict

from google.cloud import texttospeech
from gcp_auth import get_credentials, GCPAuthError

logger = logging.getLogger(__name__)

# (language_code, ssml_gender) -> voice_name 캐시
_CHIRP_VOICE_CACHE: Dict[tuple, str] = {}


class TTSError(Exception):
    def __init__(self, stage: str, detail: str):
        self.stage = stage
        self.detail = detail
        super().__init__(f"[{stage}] {detail}")


def _get_tts_client() -> texttospeech.TextToSpeechClient:
    try:
        creds = get_credentials()
    except GCPAuthError as e:
        logger.error("TTS auth error: %s", e, exc_info=True)
        raise TTSError(f"auth/{e.stage}", e.detail)

    try:
        client = texttospeech.TextToSpeechClient(credentials=creds)
    except Exception as e:
        logger.error("TTS client init failed: %s", e, exc_info=True)
        raise TTSError("client", f"TextToSpeechClient 생성 실패: {e}")

    return client


def _pick_chirp_voice_name(
    client: texttospeech.TextToSpeechClient,
    language_code: str,
    ssml_gender: texttospeech.SsmlVoiceGender,
) -> str:
    """
    해당 language_code에서 Chirp(Chirp 3/HD) 보이스를 찾아 name 반환.
    가능한 경우 gender까지 맞춰서 고른다.
    """
    cache_key = (language_code, ssml_gender)
    if cache_key in _CHIRP_VOICE_CACHE:
        return _CHIRP_VOICE_CACHE[cache_key]

    try:
        voices = client.list_voices(language_code=language_code).voices
    except Exception as e:
        raise TTSError("voice_list", f"list_voices 실패: {e}")

    # 1) Chirp 계열만 필터 (Chirp 3: HD voices 포함)
    chirp = [v for v in voices if "Chirp" in (v.name or "")]
    if not chirp:
        raise TTSError("voice_pick", f"{language_code}에서 Chirp 보이스를 찾지 못했습니다.")

    # 2) gender까지 맞는 후보 우선
    gender_matched = [v for v in chirp if v.ssml_gender == ssml_gender]
    chosen = (gender_matched[0] if gender_matched else chirp[0])

    _CHIRP_VOICE_CACHE[cache_key] = chosen.name
    return chosen.name


def synthesize_pcm_bytes(
    text: str,
    accent: Optional[Literal["us", "uk", "au"]] = None,
    gender: Optional[Literal["f", "m"]] = None,
    sr: int = 16_000,
) -> Tuple[bytes, int]:
    if not text or not text.strip():
        raise TTSError("input", "TTS 입력 텍스트가 비어 있습니다.")

    client = _get_tts_client()

    accent = (accent or "us").lower()
    gender = (gender or "f").lower()

    # 언어코드 매핑
    if accent == "uk":
        language_code = "en-GB"
    elif accent == "au":
        language_code = "en-AU"
    else:
        language_code = "en-US"

    # 성별 매핑
    ssml_gender = (
        texttospeech.SsmlVoiceGender.MALE
        if gender == "m"
        else texttospeech.SsmlVoiceGender.FEMALE
    )

    voice_name = _pick_chirp_voice_name(client, language_code, ssml_gender)

    try:
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=sr,
        )
    except Exception as e:
        raise TTSError("request", f"TTS 요청 파라미터 구성 실패: {e}")

    logger.info(
        "Calling TTS(Chirp): lang=%s, gender=%s, sr=%d, voice=%s",
        language_code, gender, sr, voice_name
    )

    try:
        response = client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config,
        )
    except Exception as e:
        logger.error("TTS synthesize_speech() 실패: %s", e, exc_info=True)
        raise TTSError("api", f"TTS API 호출 실패: {e}")

    if not response.audio_content:
        raise TTSError("empty", "TTS 응답에 audio_content가 없습니다.")

    return response.audio_content, sr
