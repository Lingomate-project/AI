"""main.py
분리된 모듈(stt.py, tts.py, llm.py)을 사용하는 영어 회화 코치 실행 스크립트

- 수동 녹음: ENTER를 누르면 녹음 시작, 다시 ENTER를 누르면 녹음 종료
- 대화 이력과 주제를 유지하여 맥락을 반영
- 주제가 없으면 AI가 먼저 주제를 제안하거나 사용자에게 선택을 요청
"""

import os, sys, json, asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

# sounddevice 사용 시 이벤트 루프 정책 설정
if sys.platform.startswith("win"):
    import asyncio as _asyncio
    _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())

from stt import stt_transcribe, SAMPLE_RATE
from tts import speak_reply_en
from llm import gemini_analyze, refine_transcript

import sounddevice as sd
import threading

MAX_HISTORY_TURNS = 6                     # 유지할 최근 대화 턴 수
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")  # 음성 파일 저장 경로


@dataclass
class SessionConfig:
    difficulty: str = "medium"    # "easy" | "medium" | "hard"
    mode: str = "casual"          # "casual" | "formal"
    accent: str = "us"            # "us" | "uk" | "au"
    gender: str = "f"             # "f" | "m"
    user_id: Optional[str] = None 


def select_session_config() -> SessionConfig:
    print("=== Session Settings ===")

    # 난이도
    diff_map = {"1": "easy", "2": "medium", "3": "hard"}
    diff_in = input("Difficulty [1=easy, 2=medium, 3=hard] (default=2): ").strip()
    difficulty = diff_map.get(diff_in, "medium")

    # 말투
    mode_map = {"1": "casual", "2": "formal"}
    mode_in = input("Style [1=casual, 2=formal] (default=1): ").strip()
    mode = mode_map.get(mode_in, "casual")

    # 억양
    accent_map = {"1": "us", "2": "uk", "3": "au"}
    acc_in = input("Accent [1=US, 2=UK, 3=AU] (default=1): ").strip()
    accent = accent_map.get(acc_in, "us")

    # 성별
    gender_map = {"1": "f", "2": "m"}
    gen_in = input("Voice gender [1=female, 2=male] (default=1): ").strip()
    gender = gender_map.get(gen_in, "f")

    cfg = SessionConfig(
        difficulty=difficulty,
        mode=mode,
        accent=accent,
        gender=gender,
        user_id=None,
    )
    print(
        f"\n→ difficulty={cfg.difficulty}, mode={cfg.mode}, "
        f"accent={cfg.accent.upper()}, gender={'female' if cfg.gender == 'f' else 'male'}"
    )
    return cfg


class ManualRecorder:
    """
    수동 시작/종료 방식 마이크 녹음기
    - start() 호출로 녹음 시작
    - stop() 호출로 녹음 종료
    """
    def __init__(self, sr: int = SAMPLE_RATE, frame_ms: int = 20):
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
        """녹음 스레드 시작"""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> bytes:
        """녹음 종료 후 누적된 PCM 바이트 반환"""
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


def speak_initial_prompt(cfg: SessionConfig) -> None:
    """세션 시작 시 안내 멘트를 영어로 말해주고, 동일 내용을 콘솔에도 출력"""
    intro = (
        "Hi! I'm your English conversation coach. You can suggest your own topic."
    )
    print("\n=== AI 응답 ===")
    print(intro)
    speak_reply_en(
        intro,
        sr=24000,
        output_dir=OUTPUT_DIR,
        accent=cfg.accent,
        gender=cfg.gender,
    )


async def main() -> None:
    """메인 루프: 수동 녹음 → STT → (필요 시) LLM 정제 → 튜터 분석 → 피드백/답변"""
    print("=========영어회화 대화 세션=========\n")

    user_id = os.getenv("USER_ID") or None

    # CLI에서 세션 설정 선택
    session_config = select_session_config()
    session_config.user_id = user_id

    chat_history: List[Dict[str, str]] = []   # 최근 대화 히스토리(간단 구조)
    current_topic: Optional[str] = None       # 현재 대화 주제

    # 시작 멘트(주제 제안)
    speak_initial_prompt(session_config)

    while True:
        # 입력 대기: ENTER로 녹음 시작, q로 종료
        try:
            choice = input("\nENTER 입력으로 녹음 시작, 'q'입력으로 세션 종료: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if choice == 'q':
            print("Bye!")
            break

        # 녹음 시작
        rec = ManualRecorder(sr=SAMPLE_RATE, frame_ms=20)
        rec.start()
        try:
            input("녹음중... ENTER 입력으로 종료: ")
        except (EOFError, KeyboardInterrupt):
            print("\n녹음 종료 중...")
        pcm = rec.stop()

        # 1) STT: N-best 후보 포함 인식
        raw_transcript, alt_segments = await stt_transcribe(pcm, sr=SAMPLE_RATE)

        if not raw_transcript and not any(alt_segments):
            print("No transcript. Try again.\n")
            continue

        # 2) LLM 후처리: 맥락/주제/N-best를 활용해 자연스러운 1문장으로 정제
        refined = refine_transcript(raw_transcript, alt_segments, chat_history, current_topic)
        print(f"\n[Refined transcript] {refined or raw_transcript}")

        # 3) 튜터 분석: 교정문(영어), 피드백(한국어 이유 설명), 후속 답변(영어) 생성
        result = gemini_analyze(
            refined or raw_transcript,
            chat_history,
            current_topic,
            difficulty=session_config.difficulty,
            register=session_config.mode,
        )
        corrected = result["corrected_en"]
        feedback = result["feedback"]
        reply_en = result["reply_en"]

        # meta에서 topic 업데이트
        try:
            meta_obj = json.loads(result.get("meta", "{}"))
            new_topic = meta_obj.get("topic") if isinstance(meta_obj, dict) else None
            if new_topic:
                current_topic = new_topic
        except Exception:
            pass

        # 최근 대화 히스토리 유지
        chat_history.append(
            {"user": refined or raw_transcript, "corrected_en": corrected, "reply_en": reply_en}
        )
        if len(chat_history) > MAX_HISTORY_TURNS:
            chat_history = chat_history[-MAX_HISTORY_TURNS:]

        # 텍스트 피드백 출력
        print("\n=== 사용자 응답에 대한 피드백 ===")
        if corrected:
            print("Corrected:", corrected)
        print(feedback or "(문법/자연스러움 이슈가 크게 없어요!)")

        # 영어 답변 음성 합성 및 재생
        print("\n=== AI 응답 ===")
        print(reply_en)
        path = speak_reply_en(
            reply_en,
            sr=24000,
            output_dir=OUTPUT_DIR,
            accent=session_config.accent,
            gender=session_config.gender,
        )
        print(f"(Saved) {path}")


if __name__ == "__main__":
    asyncio.run(main())
