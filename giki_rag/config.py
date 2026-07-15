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
TOP_K_TEXT = 6          # text chunks to retrieve per query
TOP_K_IMAGES = 3        # images to retrieve per query
MIN_IMAGE_SCORE = 0.20  # skip weakly-matched images (cosine similarity)

# --- Answer generation (Claude) --------------------------------------------
# Per Anthropic defaults; change to "claude-sonnet-4-6" / "claude-haiku-4-5"
# to trade quality for cost/speed.
ANSWER_MODEL = "claude-opus-4-8"
MAX_TOKENS = 16000
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Only send images to Claude that are reasonably relevant AND exist on disk.
MAX_IMAGES_TO_LLM = 3
