"""
CLI voice loop: record from the mic, transcribe, answer, speak the answer.

Requires sounddevice for mic capture (the audio-recorder-streamlit widget only
exists inside app.py's browser UI). Mostly useful as an isolated way to sanity
check the STT -> answer -> TTS wiring outside Streamlit.

Run:
    python voice_chat.py                # 6s recording, one question
    python voice_chat.py --seconds 10
"""
import argparse
import io
import wave

import sounddevice as sd

import stt
import tts
from chat import answer


def record(seconds=6, samplerate=16000):
    print(f"Recording {seconds}s... speak now.")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate,
                   channels=1, dtype="int16")
    sd.wait()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)          # int16 -> 2 bytes/sample
        w.setframerate(samplerate)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description="CLI voice loop over the GIKI RAG")
    ap.add_argument("--seconds", type=int, default=6, help="recording length")
    args = ap.parse_args()

    wav_bytes = record(seconds=args.seconds)

    print("Transcribing...")
    query = stt.transcribe(wav_bytes)
    print(f"You said: {query!r}")
    if not query.strip():
        print("Nothing transcribed — try again, closer to the mic.")
        return

    text = answer(query, stream_to_stdout=True)

    print("\nSynthesizing spoken answer...")
    audio_path = tts.synthesize(text)
    print(f"Answer audio -> {audio_path}")


if __name__ == "__main__":
    main()
