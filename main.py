"""main_manual_toggle.py
Run the English conversation coach using split modules: stt.py, tts.py, llm.py
- Manual recording: ENTER to start, ENTER to stop (no VAD).
- Maintains chat history & topic.
- AI proposes/asks for topic first.
- Feedback reasons are in Korean; TTS speaks only the English reply.
"""

import os, sys, json, asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

# Windows event loop policy for sounddevice
if sys.platform.startswith("win"):
    import asyncio as _asyncio
    _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())

from stt import stt_transcribe, SAMPLE_RATE
from tts import speak_reply_en
from llm import gemini_analyze, refine_transcript

import sounddevice as sd
import threading

MAX_HISTORY_TURNS = 6
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")


class ManualRecorder:
    """Manual start/stop microphone recorder.
    Start with start(), stop with stop() → returns PCM16 mono bytes.
    """
    def __init__(self, sr: int = 16000, frame_ms: int = 20):
        self.sr = sr
        self.frame_ms = frame_ms
        self.frames_per_chunk = int(sr * frame_ms / 1000)
        self.buffers: List[bytes] = []
        self.stop_event = threading.Event()
        self.stream = sd.RawInputStream(
            samplerate=sr,
            blocksize=self.frames_per_chunk,
            dtype='int16',
            channels=1,
        )
        self.thread: Optional[threading.Thread] = None

    def _run(self):
        self.stream.start()
        while not self.stop_event.is_set():
            data, overflowed = self.stream.read(self.frames_per_chunk)
            if overflowed:
                pass
            self.buffers.append(data)
        try:
            self.stream.stop()
        except Exception:
            pass
        try:
            self.stream.close()
        except Exception:
            pass

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> bytes:
        self.stop_event.set()
        try:
            self.stream.abort()
        except Exception:
            pass
        if self.thread:
            self.thread.join()
        pcm = b"".join(self.buffers)
        print(f"- captured (manual) {len(pcm)} bytes")
        return pcm


def speak_initial_prompt() -> None:
    intro = (
        "Hi! I'm your English conversation coach. Let's pick a topic — "
        "we could talk about travel, food, or daily routine. "
        "Which sounds good? You can also suggest your own topic."
    )
    print("\n=== AI REPLY (spoken, initial prompt) ===")
    print(intro)
    speak_reply_en(intro, sr=SAMPLE_RATE, output_dir=OUTPUT_DIR)


async def main() -> None:
    print("English Coach Voice Loop. Manual mode: ENTER to start, ENTER to stop.\n")

    chat_history: List[Dict[str, str]] = []
    current_topic: Optional[str] = None

    # Proactive opening
    speak_initial_prompt()

    while True:
        # Manual toggle: ENTER to start, ENTER to stop (or 'q' to quit)
        try:
            choice = input("\nPress ENTER to START recording, or 'q' to quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if choice == 'q':
            print("Bye!")
            break

        rec = ManualRecorder(sr=SAMPLE_RATE, frame_ms=20)
        rec.start()
        try:
            input("Recording... Press ENTER to STOP: ")
        except (EOFError, KeyboardInterrupt):
            print("\nStopping recording...")
        pcm = rec.stop()

        # --- STT with N-best ---
        raw_transcript, alt_segments = await stt_transcribe(pcm, sr=SAMPLE_RATE)

        if not raw_transcript and not any(alt_segments):
            print("No transcript. Try again.\n")
            continue

        print("\n[Top-3 alternatives per segment]")
        for i, alts in enumerate(alt_segments):
            print(f"  seg {i+1}: {alts[:3]}")

        # --- LLM refinement (context-aware disambiguation) ---
        refined = refine_transcript(raw_transcript, alt_segments, chat_history, current_topic)
        print(f"\n[Refined transcript] {refined or raw_transcript}")

        # --- Tutor analysis ---
        result = gemini_analyze(refined or raw_transcript, chat_history, current_topic)
        corrected = result["corrected_en"]
        feedback = result["feedback"]
        reply_en = result["reply_en"]

        # Update topic if provided by the model
        try:
            meta_obj = json.loads(result.get("meta", "{}"))
            new_topic = meta_obj.get("topic") if isinstance(meta_obj, dict) else None
            if new_topic:
                current_topic = new_topic
        except Exception:
            pass

        # Keep short rolling history
        chat_history.append({"user": refined or raw_transcript, "corrected_en": corrected, "reply_en": reply_en})
        if len(chat_history) > MAX_HISTORY_TURNS:
            chat_history = chat_history[-MAX_HISTORY_TURNS:]

        print("\n=== FEEDBACK (text only) ===")
        if corrected:
            print("Corrected:", corrected)
        print(feedback or "(문법/자연스러움 이슈가 크게 없어요!)")

        print("\n=== AI REPLY (spoken) ===")
        print(reply_en)
        path = speak_reply_en(reply_en, sr=SAMPLE_RATE, output_dir=OUTPUT_DIR)
        print(f"(Saved) {path}")


if __name__ == "__main__":
    asyncio.run(main())
