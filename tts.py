"""tts.py
Google Cloud Text-to-Speech for speaking ONLY the conversational English reply.
- Voice selection is hard-coded (no env).
- Exposes: speak_reply_en(text: str, sr: int = 16000, output_dir: str = "outputs") -> str
"""

import os
from google.cloud import texttospeech as tts
import numpy as np
import sounddevice as sd

# Hard-coded voice policy (Chirp 3: HD)
TTS_VOICE_EN: str = "en-US-Chirp-HD-F"

# Recommended sample rate for Chirp HD
_CHIRP_RECOMMENDED_SR = 24000

def _lang_from_voice(name: str, fallback: str) -> str:
    parts = name.split("-")
    return f"{parts[0]}-{parts[1]}" if len(parts) >= 2 and len(parts[0]) == 2 and len(parts[1]) == 2 else fallback

def _now_tag() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _play_pcm16(pcm: bytes, sr: int) -> None:
    sd.play(np.frombuffer(pcm, dtype=np.int16), samplerate=sr)
    sd.wait()

def _save_wav(pcm: bytes, path: str, sr: int) -> None:
    import wave
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(pcm)

def _ensure_sentence_boundary(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    if t[-1] not in ".!?":
        t += "."
    return t

def _append_silence_tail(pcm: bytes, sr: int, seconds: float = 0.25) -> bytes:
    # LINEAR16 mono → 2 bytes per sample
    tail_bytes = int(sr * seconds) * 2
    return pcm + (b"\x00" * tail_bytes)

def speak_reply_en(text: str, sr: int = 16000, output_dir: str = "outputs") -> str:
    voice_name = TTS_VOICE_EN
    lang_for_tts = _lang_from_voice(voice_name, "en-US")

    # For Chirp HD voices, prefer 24 kHz even if caller passed 16 kHz
    if "Chirp" in voice_name and sr < _CHIRP_RECOMMENDED_SR:
        sr = _CHIRP_RECOMMENDED_SR

    # Add an ending punctuation to help prosody close naturally
    text = _ensure_sentence_boundary(text)

    print(f"\n=== TTS via Google (voice={voice_name}, pcm@{sr}, lang={lang_for_tts}) ===")

    client = tts.TextToSpeechClient()
    synthesis_input = tts.SynthesisInput(text=text)

    voice_params = tts.VoiceSelectionParams(
        name=voice_name,
        language_code=lang_for_tts,
    )

    # Chirp 3: HD — keep it simple (no SSML/speaking_rate/pitch)
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        sample_rate_hertz=sr,
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    pcm = response.audio_content

    # Add short silence tail to prevent last word from sounding cut off
    pcm = _append_silence_tail(pcm, sr, seconds=0.25)

    _play_pcm16(pcm, sr)
    out = os.path.abspath(os.path.join(output_dir, f"reply_{_now_tag()}.wav"))
    _save_wav(pcm, out, sr)
    print(f"→ saved: {out}")
    return out
