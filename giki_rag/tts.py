"""
Unified text-to-speech interface. Three interchangeable backends, selected by
config.TTS_BACKEND (mirrors chat.py's LLM_BACKEND pattern):
  "edge"    -> edge-tts (cloud, best quality, needs internet)
  "piper"   -> piper-tts (local ONNX, CPU-fast, auto-downloads its voice model)
  "pyttsx3" -> pyttsx3 (offline SAPI5, instant, robotic)

edge-tts writes real mp3-codec bytes regardless of filename, so its output uses
a .mp3 extension; piper/pyttsx3 write real .wav files. Both faster-whisper and
st.audio()/browsers handle either format fine, so no transcoding is needed.
"""
import asyncio
import functools
import re
import uuid
import wave
from pathlib import Path

import config

_CITATION_RE = re.compile(r"\[source:\s*[^\]]*\]", re.IGNORECASE)


def strip_citations(text):
    """Remove '[source: <url>]' markup before speaking; on-screen text keeps it."""
    cleaned = _CITATION_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def synthesize(text, backend=None, out_path=None):
    """
    Synthesize `text` to an audio file and return its Path.
    backend defaults to config.TTS_BACKEND; out_path defaults to a fresh file
    under config.AUDIO_CACHE_DIR.
    """
    backend = backend or config.TTS_BACKEND
    config.AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = strip_citations(text)
    if not cleaned:
        cleaned = "There is nothing to say."

    if out_path is None:
        ext = ".mp3" if backend == "edge" else ".wav"
        out_path = config.AUDIO_CACHE_DIR / f"{backend}_{uuid.uuid4().hex}{ext}"
    out_path = Path(out_path)

    if backend == "edge":
        _synthesize_edge(cleaned, out_path)
    elif backend == "piper":
        _synthesize_piper(cleaned, out_path)
    elif backend == "pyttsx3":
        _synthesize_pyttsx3(cleaned, out_path)
    else:
        raise ValueError(f"Unknown TTS_BACKEND: {backend}")
    return out_path


# --------------------------------------------------------------------------
# Backend: edge-tts (cloud)
# --------------------------------------------------------------------------
def _synthesize_edge(text, out_path):
    import edge_tts

    async def _run():
        communicate = edge_tts.Communicate(text, voice=config.EDGE_TTS_VOICE,
                                            rate=config.EDGE_TTS_RATE)
        await communicate.save(str(out_path))

    asyncio.run(_run())


# --------------------------------------------------------------------------
# Backend: piper-tts (local ONNX)
# --------------------------------------------------------------------------
def _ensure_piper_voice_downloaded():
    if config.PIPER_MODEL_PATH.exists() and config.PIPER_CONFIG_PATH.exists():
        return
    from piper.download_voices import download_voice
    config.PIPER_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[tts] downloading piper voice '{config.PIPER_VOICE_NAME}' "
          f"(~63MB, one-time)...")
    download_voice(config.PIPER_VOICE_NAME, config.PIPER_VOICE_DIR)


@functools.lru_cache(maxsize=1)
def _piper_voice():
    _ensure_piper_voice_downloaded()
    from piper import PiperVoice
    return PiperVoice.load(str(config.PIPER_MODEL_PATH),
                            config_path=str(config.PIPER_CONFIG_PATH),
                            use_cuda=config.PIPER_USE_CUDA)


def _synthesize_piper(text, out_path):
    voice = _piper_voice()
    with wave.open(str(out_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)


# --------------------------------------------------------------------------
# Backend: pyttsx3 (offline SAPI5)
# --------------------------------------------------------------------------
def _synthesize_pyttsx3(text, out_path):
    # A cached/reused engine hangs on the second runAndWait() call on this
    # machine's Windows SAPI5/pywin32 combo (confirmed by testing) — a known
    # pyttsx3-on-Windows issue. A fresh engine per call avoids it entirely;
    # pyttsx3.init() is cheap (no model weights to load).
    import pyttsx3
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", config.PYTTSX3_RATE)
        if config.PYTTSX3_VOICE_ID:
            engine.setProperty("voice", config.PYTTSX3_VOICE_ID)
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
    finally:
        engine.stop()


if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else config.TTS_BACKEND
    text = " ".join(sys.argv[2:]) or "Hello from the GIKI multimodal RAG system."
    path = synthesize(text, backend=backend)
    print(f"[{backend}] wrote {path} ({path.stat().st_size} bytes)")
