# GIKI Multimodal RAG

Turns the scraped **giki.edu.pk** archive into a question-answering system that
retrieves both **text** and **images**, then answers with a **vision-capable LLM**
— citing the source pages. Runs fully **offline on a local model by default**, or
with **Claude** via API.

## Architecture
```
                 🎙 spoken question
                       │  stt.py (faster-whisper)
                       ▼
scraped JSON (text + images + captions)      transcribed text
        │                                          │
        ▼   ingest.py                              │
  ┌───────────────────────────────────────┐        │
  │  ChromaDB                             │◄───────┘
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
        │
        ▼  tts.py (edge-tts / piper / pyttsx3)
    🔊 spoken answer
```

## Defaults (all configurable in `config.py`)
| Component | Choice |
|-----------|--------|
| Vector DB | ChromaDB (local, persistent) |
| Text embeddings | `all-MiniLM-L6-v2` |
| Image embeddings | CLIP `clip-ViT-B-32` (shared text+image space → text queries can find images) |
| Answer LLM | **Ollama `qwen2.5vl:3b`** (local, default) — or Claude `claude-opus-4-8` |
| Retrieval top-k | 3 text chunks + 1 image per query (tuned for the local model) |
| Speech-to-text (input) | `faster-whisper` (`small`, CPU, int8, language forced to English) — fixed, not switchable |
| Text-to-speech (output) | **`edge-tts`** (WER=0.255 in `evaluate_tts.py`) — switchable to `piper`/`pyttsx3` |

Switch backends with `LLM_BACKEND` in `config.py` (`"ollama"` or `"anthropic"`),
and `TTS_BACKEND` (`"edge"` / `"piper"` / `"pyttsx3"`).

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

**Voice I/O** — no extra setup for most of it:
- `piper` downloads its voice model (`en_US-lessac-medium`, ~63MB) automatically
  on its first synthesis call — needs internet once, then works fully offline.
- `edge-tts` needs internet on every call (it's a cloud service, just a free one).
- `pyttsx3` and `faster-whisper` (STT) are fully offline, no download needed.

## Run
```bash
python ingest.py --reset          # build the vector store from scraped data
python ingest.py --limit 50       # quick test on 50 pages first

python retrieve.py "hostel facilities"     # inspect retrieval (no LLM needed)
python chat.py "What undergraduate programs does GIKI offer?"
python chat.py                             # interactive Q&A loop

streamlit run app.py              # web chat UI — mic input + spoken answers
python voice_chat.py              # CLI voice loop: record -> transcribe -> answer -> speak
python generate_eval_csv.py       # run the 20-question RAG evaluation → CSV
python evaluate_tts.py            # compare TTS engines → giki_rag_tts_evaluation.csv
```

## Files
| File | Role |
|------|------|
| `config.py` | Models, paths, chunk sizes, top-k, backend — all settings |
| `embeddings.py` | Local text (MiniLM) + image (CLIP) embedders |
| `ingest.py` | Chunk + embed scraped pages/images → ChromaDB |
| `retrieve.py` | Query → relevant text chunks + images |
| `chat.py` | Retrieve → vision LLM (Ollama/Claude) → cited answer |
| `tts.py` | Text → speech: `edge-tts` / `piper` / `pyttsx3`, switchable via `TTS_BACKEND` |
| `stt.py` | Speech → text via `faster-whisper` (fixed, used for input + eval) |
| `voice_chat.py` | CLI voice loop: record mic → transcribe → answer → speak |
| `app.py` | Streamlit chat UI — mic input, spoken answers, sources + images |
| `generate_eval_csv.py` | 20-question RAG benchmark → `giki_rag_evaluation.csv` |
| `evaluate_tts.py` | Round-trip TTS engine comparison → `giki_rag_tts_evaluation.csv` |

## Voice / Audio
The RAG supports a full voice loop, layered on top of the existing text pipeline
without changing it:
- **Input**: `app.py`'s mic recorder (or `voice_chat.py`'s CLI recorder) captures
  audio → `stt.py` transcribes it with `faster-whisper` → the transcript is fed
  into the *same* `answer()` call a typed question would use.
- **Output**: the generated answer (citation markup stripped) is spoken aloud via
  `tts.py`, using whichever engine `TTS_BACKEND` points to.
- Three interchangeable TTS engines, picked by `evaluate_tts.py` (see below):
  `edge-tts` (cloud, best quality, needs internet), `piper` (local ONNX, CPU-fast,
  auto-downloads its voice model once), `pyttsx3` (offline Windows SAPI5, instant,
  lower quality).
- **STT tuning** (in `config.py`): the language is forced to English
  (`STT_LANGUAGE`) because Whisper's auto-detect is unreliable on short clips and
  will silently transcribe English speech in the wrong script; an
  `STT_INITIAL_PROMPT` biases recognition toward GIKI-specific vocabulary
  (otherwise "GIKI" is misheard). The `small` model is used rather than `base`
  for noticeably better accuracy on proper nouns.

## Evaluation
`generate_eval_csv.py` runs a fixed 20-question benchmark across the main GIKI
topics through the full pipeline and records **automatically-computed** metrics
per query — retrieval relevance (top/avg cosine similarity, chunks & images
retrieved, source URLs), the generated answer, whether it was answered from
context, answer length, and end-to-end latency. Because the scores are computed
(not hand-graded), the evaluation is objective and reproducible: re-run it after
any change to chunking, top-k, or the model and the numbers are directly
comparable. Results are written to `giki_rag_evaluation.csv`.

`evaluate_tts.py` compares the three TTS engines with a **round-trip**
methodology: WER, ROUGE-L, and BERTScore all compare text against text, and
can't score synthesized audio directly. So for each engine, real GIKI RAG
answers (not generic sentences) are synthesized to speech, transcribed back with
the fixed `faster-whisper` model, and the round-trip transcript is compared
against the original text — approximating intelligibility with the requested
metrics. Synthesis time and real-time factor are recorded too. The engine with
the lowest average WER is reported as the winner; results go to
`giki_rag_tts_evaluation.csv`.

Results from the last run (8 real RAG answers x 3 engines, 24 round-trips):

| Engine | WER ↓ | ROUGE-L | BERTScore | Synth time (avg) | Needs internet? |
|--------|-------|---------|-----------|-------------------|------------------|
| **edge-tts** (winner) | **0.255** | 0.931 | 0.852 | 16.5s | Yes, every call |
| pyttsx3 | 0.273 | 0.916 | 0.844 | 0.27s | No |
| piper | 0.302 | 0.923 | 0.842 | 2.9s | No (after first voice download) |

`edge-tts` wins on WER and is set as the default, but the margin over `pyttsx3`
is small (0.255 vs 0.273) while `pyttsx3` is ~60x faster and fully offline — a
reasonable choice to switch to via `TTS_BACKEND` if offline operation or low
latency matters more than the last few points of accuracy.

## Notes
- **Grounded**: the system prompt forces the model to answer only from retrieved
  context and to say so when the answer isn't present — reduces hallucination.
- **Re-ingest** whenever the scrape grows: `python ingest.py` upserts (safe to
  re-run; unchanged items are overwritten in place).
