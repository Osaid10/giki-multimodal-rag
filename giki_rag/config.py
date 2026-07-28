"""
Configuration for the GIKI multimodal RAG pipeline.

Pipeline:  scraped JSON  ->  ingest (chunk + embed)  ->  ChromaDB
           query  ->  retrieve (text + images)  ->  Claude  ->  cited answer
"""
import os
from pathlib import Path

# --- Where the scraped data lives (output of the scraper) ------------------
SCRAPE_DIR = Path(__file__).parent.parent / "giki_scraper" / "giki_scrape"
TEXT_JSON_DIR = SCRAPE_DIR / "text"        # per-page <slug>.json files
IMAGE_DIR = SCRAPE_DIR                       # image local_paths are relative to here

# --- Vector store ----------------------------------------------------------
CHROMA_DIR = Path(__file__).parent / "chroma_db"
TEXT_COLLECTION = "giki_text"
IMAGE_COLLECTION = "giki_images"
DISTANCE = "cosine"

# --- Embedding models (run locally on GPU if available) --------------------
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"       # 384-dim, fast, strong general retriever
IMAGE_EMBED_MODEL = "clip-ViT-B-32"         # shared image+text space (cross-modal)

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = 900        # characters per text chunk
CHUNK_OVERLAP = 150     # overlap so sentences aren't split mid-idea

# --- Retrieval -------------------------------------------------------------
TOP_K_TEXT = 3          # text chunks to retrieve per query
TOP_K_IMAGES = 1        # images to retrieve per query
MIN_IMAGE_SCORE = 0.20  # skip weakly-matched images (cosine similarity)

# --- Answer generation -----------------------------------------------------
# Which LLM writes the final answer:
#   "ollama"    -> a local, free, offline vision model (default)
#   "anthropic" -> Claude via API (needs ANTHROPIC_API_KEY + credits)
# Both read the retrieved IMAGES as well as the text (multimodal).
LLM_BACKEND = "ollama"

# -- Local backend (Ollama) --
# Qwen2.5-VL 3B: vision-capable and ~3.2GB, so it fits a 6GB GPU comfortably.
OLLAMA_MODEL = "qwen2.5vl:3b"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 300          # local generation can be slow on a laptop GPU
# Keep the model loaded in VRAM between questions so we don't pay a slow cold
# reload each time. "30m" = stay 30 min after last use; "-1" = never unload.
OLLAMA_KEEP_ALIVE = "30m"

# -- Cloud backend (Claude), used only if LLM_BACKEND == "anthropic" --
ANSWER_MODEL = "claude-opus-4-8"
MAX_TOKENS = 16000
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Only send images to Claude that are reasonably relevant AND exist on disk.
MAX_IMAGES_TO_LLM = 3

# --- Voice I/O ---------------------------------------------------------
# Where synthesized speech + recorded audio are cached (git-ignored, regenerable).
AUDIO_CACHE_DIR = Path(__file__).parent / "audio_cache"

# Which TTS engine speaks answers aloud. Switchable like LLM_BACKEND:
#   "edge"    -> Microsoft Edge neural TTS (cloud, needs internet, best quality)
#   "piper"   -> local neural TTS (ONNX, CPU-fast, auto-downloads its voice model)
#   "pyttsx3" -> offline Windows SAPI5 voices (instant, lower quality, zero download)
# Winner of evaluate_tts.py's round-trip evaluation (lowest WER), see
# giki_rag_tts_evaluation.csv: edge WER=0.255, piper WER=0.302, pyttsx3 WER=0.273.
TTS_BACKEND = "edge"

# -- edge-tts settings --
EDGE_TTS_VOICE = "en-US-EmmaMultilingualNeural"
EDGE_TTS_RATE = "+0%"

# -- piper-tts settings --
PIPER_VOICE_NAME = "en_US-lessac-medium"     # rhasspy/piper-voices model id
PIPER_VOICE_DIR = Path(__file__).parent / "piper_voices"
PIPER_MODEL_PATH = PIPER_VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx"
PIPER_CONFIG_PATH = PIPER_VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx.json"
PIPER_USE_CUDA = False        # keep the shared 6GB GPU free for Ollama

# -- pyttsx3 settings --
PYTTSX3_RATE = 175            # words/min (pyttsx3 default is ~200)
PYTTSX3_VOICE_ID = None       # None -> engine default SAPI5 voice

# -- STT (faster-whisper) — fixed component, not a config switch.
# Used both for live voice input and for TTS round-trip evaluation.
STT_MODEL_SIZE = "small"      # tiny/base/small/medium; base flubbed "GIKI" (a
                               # proper noun in nearly every query) — small is
                               # meaningfully more accurate at a modest CPU cost
STT_DEVICE = "cpu"            # keep GPU free for Ollama's qwen2.5vl:3b
STT_COMPUTE_TYPE = "int8"     # fastest CTranslate2 compute type on CPU
# Force English: on short clips, Whisper's language auto-detect is unreliable
# (observed guessing Urdu at 34% confidence on a clear English question) and
# silently transcribes in the wrong script instead of erroring — GIKI RAG
# content and queries are English, so there's no reason to auto-detect at all.
STT_LANGUAGE = "en"
# Biases recognition toward domain vocabulary Whisper wouldn't otherwise know
# (GIKI is not a common word and gets misheard as "Jeekey"/"Chikib" otherwise).
STT_INITIAL_PROMPT = ("GIKI, Ghulam Ishaq Khan Institute, admissions, "
                      "undergraduate, hostel, faculty, scholarships.")
