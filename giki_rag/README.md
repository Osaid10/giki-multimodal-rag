# GIKI Multimodal RAG

Turns the scraped **giki.edu.pk** archive into a question-answering system that
retrieves both **text** and **images**, then answers with **Claude** — citing
the source pages.

## Architecture
```
scraped JSON (text + images + captions)
        │
        ▼   ingest.py
  ┌─────────────────────┐
  │  ChromaDB           │
  │  • giki_text  (MiniLM embeddings)     │
  │  • giki_images (CLIP embeddings)      │
  └─────────────────────┘
        │   retrieve.py  (text + cross-modal image search)
        ▼
  Claude Opus 4.8  ── multimodal: reads retrieved text AND images
        │
        ▼
  cited answer
```

## Defaults (all configurable in `config.py`)
| Component | Choice |
|-----------|--------|
| Vector DB | ChromaDB (local, persistent) |
| Text embeddings | `all-MiniLM-L6-v2` |
| Image embeddings | CLIP `clip-ViT-B-32` (shared text+image space → text queries can find images) |
| Answer LLM | Claude `claude-opus-4-8` (multimodal) |

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # Windows: setx ANTHROPIC_API_KEY "sk-ant-..."
```
The embedding models download automatically on first run (MiniLM ~80 MB, CLIP
~600 MB) and use your GPU if available.

## Run
```bash
python ingest.py --reset          # build the vector store from scraped data
python ingest.py --limit 50       # quick test on 50 pages first

python retrieve.py "hostel facilities"     # inspect retrieval (no LLM, no API key)
python chat.py "What undergraduate programs does GIKI offer?"
python chat.py                             # interactive Q&A loop
```

## Files
| File | Role |
|------|------|
| `config.py` | Models, paths, chunk sizes, top-k — all settings |
| `embeddings.py` | Local text (MiniLM) + image (CLIP) embedders |
| `ingest.py` | Chunk + embed scraped pages/images → ChromaDB |
| `retrieve.py` | Query → relevant text chunks + images |
| `chat.py` | Retrieve → Claude (multimodal) → cited answer |

## Notes
- **Grounded**: the system prompt forces Claude to answer only from retrieved
  context and to say so when the answer isn't present — reduces hallucination.
- **Re-ingest** whenever the scrape grows: `python ingest.py` upserts (safe to
  re-run; unchanged items are overwritten in place).
