"""
Speech-to-text via faster-whisper (local, fixed model — not a config switch).
Used both for live voice input (app.py / voice_chat.py) and for TTS round-trip
evaluation (evaluate_tts.py).
"""
import functools
import tempfile
from pathlib import Path

import config


@functools.lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel
    return WhisperModel(config.STT_MODEL_SIZE, device=config.STT_DEVICE,
                         compute_type=config.STT_COMPUTE_TYPE)


def transcribe(audio):
    """
    audio: a Path/str to an audio file, OR raw bytes (e.g. from
    audio_recorder_streamlit, which returns audio/wav bytes).
    Returns the transcribed text (concatenated segments, stripped).
    """
    model = _model()
    if isinstance(audio, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            tmp_path = f.name
        try:
            segments, _info = model.transcribe(tmp_path, beam_size=5,
                                               language=config.STT_LANGUAGE,
                                               initial_prompt=config.STT_INITIAL_PROMPT)
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        segments, _info = model.transcribe(str(audio), beam_size=5,
                                           language=config.STT_LANGUAGE,
                                           initial_prompt=config.STT_INITIAL_PROMPT)
        return " ".join(seg.text.strip() for seg in segments).strip()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python stt.py <audio_file>")
    print(transcribe(sys.argv[1]))
