# GIKI Multimodal RAG

Turns the scraped **giki.edu.pk** archive into a question-answering system that
retrieves both **text** and **images**, then answers with a **vision-capable LLM**
— citing the source pages. Runs fully **offline on a local model by default**, or
with **Claude** via API.

## Architecture
```
scraped JSON (text + images + captions)
        │
        ▼   ingest.py
  ┌───────────────────────────────────────┐
  │  ChromaDB                             │
  │  • giki_text   (MiniLM embeddings)   │
  │  • giki_images (CLIP embeddings)     │
  └───────────────────────────────────────┘
        │   retrieve.py  (text + cross-modal image search)
        ▼
  Vision LLM  ── multimodal: reads retrieved text AND images
   • Qwen2.5-VL 3B via Ollama  (local, free, offline)  [default]
   • Claude Opus 4.8 via API   (optional)
        │
        ▼
  cited answer   ──►  CLI  ·  Streamlit chat UI  ·  evaluation harness
```

## Defaults (all configurable in `config.py`)
| Component | Choice |
|-----------|--------|
| Vector DB | ChromaDB (local, persistent) |
| Text embeddings | `all-MiniLM-L6-v2` |
| Image embeddings | CLIP `clip-ViT-B-32` (shared text+image space → text queries can find images) |
| Answer LLM | **Ollama `qwen2.5vl:3b`** (local, default) — or Claude `claude-opus-4-8` |
| Retrieval top-k | 3 text chunks + 1 image per query (tuned for the local model) |

Switch backends with `LLM_BACKEND` in `config.py` (`"ollama"` or `"anthropic"`).

## Setup
```bash
pip install -r requirements.txt
```
The embedding models download automatically on first run (MiniLM ~80 MB, CLIP
~600 MB) and use your GPU if available.

**Local backend (default)** — install [Ollama](https://ollama.com), then:
```bash
ollama serve                 # start the server
ollama pull qwen2.5vl:3b     # ~3.2 GB vision model, fits a 6 GB GPU
```

**Cloud backend (optional)** — set `LLM_BACKEND = "anthropic"` in `config.py` and:
```bash
export ANTHROPIC_API_KEY=sk-ant-...      # Windows: setx ANTHROPIC_API_KEY "sk-ant-..."
```

## Run
```bash
python ingest.py --reset          # build the vector store from scraped data
python ingest.py --limit 50       # quick test on 50 pages first

python retrieve.py "hostel facilities"     # inspect retrieval (no LLM needed)
python chat.py "What undergraduate programs does GIKI offer?"
python chat.py                             # interactive Q&A loop

streamlit run app.py              # web chat UI (sources + images inline)
python generate_eval_csv.py       # run the 20-question evaluation → CSV
```

## Files
| File | Role |
|------|------|
| `config.py` | Models, paths, chunk sizes, top-k, backend — all settings |
| `embeddings.py` | Local text (MiniLM) + image (CLIP) embedders |
| `ingest.py` | Chunk + embed scraped pages/images → ChromaDB |
| `retrieve.py` | Query → relevant text chunks + images |
| `chat.py` | Retrieve → vision LLM (Ollama/Claude) → cited answer |
| `app.py` | Streamlit chat UI with retrieved sources + images |
| `generate_eval_csv.py` | 20-question benchmark → `giki_rag_evaluation.csv` |

## Evaluation
`generate_eval_csv.py` runs a fixed 20-question benchmark across the main GIKI
topics through the full pipeline and records **automatically-computed** metrics
per query — retrieval relevance (top/avg cosine similarity, chunks & images
retrieved, source URLs), the generated answer, whether it was answered from
context, answer length, and end-to-end latency. Because the scores are computed
(not hand-graded), the evaluation is objective and reproducible: re-run it after
any change to chunking, top-k, or the model and the numbers are directly
comparable. Results are written to `giki_rag_evaluation.csv`.

## Notes
- **Grounded**: the system prompt forces the model to answer only from retrieved
  context and to say so when the answer isn't present — reduces hallucination.
- **Re-ingest** whenever the scrape grows: `python ingest.py` upserts (safe to
  re-run; unchanged items are overwritten in place).
