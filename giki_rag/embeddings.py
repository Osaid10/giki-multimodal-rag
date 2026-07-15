"""
Local embedding models: text (MiniLM) and images (CLIP).

Both are loaded lazily and cached, so importing this module is cheap and the
(slow) model load only happens the first time you actually embed something.
CLIP encodes BOTH images and text into the same vector space, which is what
lets a text query retrieve relevant images.
"""
import functools

from PIL import Image
from sentence_transformers import SentenceTransformer

import config


@functools.lru_cache(maxsize=1)
def _text_model():
    return SentenceTransformer(config.TEXT_EMBED_MODEL)


@functools.lru_cache(maxsize=1)
def _clip_model():
    return SentenceTransformer(config.IMAGE_EMBED_MODEL)


def embed_texts(texts):
    """Embed a list of strings for the TEXT collection. Returns list[list float]."""
    model = _text_model()
    vecs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=64)
    return vecs.tolist()


def embed_query_text(text):
    """Embed a single query string for text retrieval."""
    return embed_texts([text])[0]


def embed_images(paths):
    """
    Embed image files with CLIP for the IMAGE collection.
    Returns (embeddings, ok_paths) — images that failed to open are skipped.
    """
    model = _clip_model()
    imgs, ok_paths = [], []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
            ok_paths.append(p)
        except Exception:
            continue
    if not imgs:
        return [], []
    vecs = model.encode(imgs, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=32)
    return vecs.tolist(), ok_paths


def embed_query_for_images(text):
    """Embed a text query into CLIP space to search images cross-modally."""
    model = _clip_model()
    vec = model.encode([text], normalize_embeddings=True,
                       show_progress_bar=False)
    return vec[0].tolist()
