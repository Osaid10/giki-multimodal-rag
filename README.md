# GIKI Scrape + Multimodal RAG (Internship Project)

<!-- ============================================================= -->
<!-- RESUME POINT — a Claude on another PC should READ THIS FIRST.  -->
<!-- Keep this block updated as tasks complete.                    -->
<!-- ============================================================= -->
## 📍 PROJECT STATUS — resume here

**Last updated:** 2026-07-28 · **Phase:** 2 of 2 (scrape DONE, RAG + voice done)
**Deadline:** Monday 2026-07-20 demo (met); voice I/O added afterward.

### Where we are
- [x] Scraper built & tested (`giki_scraper/`)
- [x] Completeness audit tool built (`giki_scraper/verify.py`)
- [x] URL list collected — **3,371 pages** (`collect_urls.py` done)
- [x] **SCRAPE COMPLETE — 3,368/3,368 reachable pages (100%)**
      - 26,254 image refs / 7,704 files downloaded; ~2.3 GB archive
      - 3 URLs unreachable and NOT our bug: 2 are stale sitemap entries
        (HTTP 404 at source), 1 (`/jobs/`) redirects to an offline separate
        server (119.159.235.56:8081)
- [x] Multimodal RAG scaffold built (`giki_rag/`)
- [x] Local free LLM chosen: **Ollama + Qwen2.5-VL 3B** (vision, ~3.2GB, fits 6GB VRAM)
      - `chat.py` supports both backends via `LLM_BACKEND` ("ollama" | "anthropic")
- [x] Ollama model pulled; full `ingest.py` run complete (13,933 text chunks, 4,253 images)
- [x] End-to-end RAG demo (`chat.py`, `app.py` Streamlit UI, `generate_eval_csv.py`)
- [x] **Voice I/O added** (2026-07-28): `stt.py` (faster-whisper) for spoken
      questions, `tts.py` (edge-tts / piper / pyttsx3, switchable via
      `TTS_BACKEND`) for spoken answers. Compared via round-trip WER/ROUGE-L/
      BERTScore in `evaluate_tts.py` (`giki_rag/giki_rag_tts_evaluation.csv`) —
      TTS models are scored by synthesizing text, transcribing it back with a
      fixed STT model, and comparing to the original text, since WER/ROUGE/
      BERTScore compare text-to-text and can't score audio directly.
      **Winner: edge-tts** (WER 0.255 vs piper 0.302 / pyttsx3 0.273; pyttsx3 is
      ~60x faster and offline if that tradeoff is preferred instead). Wired
      into `app.py` (mic input + autoplay spoken answers) and `voice_chat.py`
      (CLI voice loop). Bugs found & fixed during live testing:
      - cached pyttsx3 engine hung on repeated calls (Windows SAPI5) → fresh
        engine per call;
      - bert-score's default model needed an unreachable HF LFS download →
        switched to the already-cached `all-MiniLM-L6-v2`;
      - STT transcribed English speech as Urdu script (Whisper auto-detect
        unreliable on short clips) → forced `STT_LANGUAGE="en"`, added an
        `STT_INITIAL_PROMPT` for GIKI vocabulary, bumped model `base`→`small`;
      - small LLM echoed raw retrieved-context scaffolding (`[2]`, `Title:`,
        inline `(source:)`) into answers → tightened `SYSTEM_PROMPT` to forbid
        it and prefer clean bulleted answers with sources at the end.
      - NOTE: if answers ever come back as `@@@@` garbage, the Ollama server
        has gotten into a bad state (seen after heavy eval querying) — restart
        it (kill `ollama.exe`; the tray app relaunches the server).
- [ ] `RAG_CONCEPTS.md` study guide (deep concepts for Monday)

### How to resume (on any PC)
The scraped data/vector store are NOT in git (large + regenerable). On a fresh
machine you must rebuild them:
```bash
cd giki_scraper && pip install -r requirements.txt
python collect_urls.py && python scrape.py   # resumable; re-run until verify.py is clean
python verify.py                             # confirm 100% coverage
cd ../giki_rag && pip install -r requirements.txt
setx ANTHROPIC_API_KEY "sk-ant-..."          # then open a new shell
python ingest.py --reset                     # build vector store
python chat.py "What programs does GIKI offer?"
```
If the data folders were copied over manually (via cloud storage), skip the
scrape and go straight to `ingest.py`.

### Known notes / gotchas
- giki.edu.pk has a broken SSL cert chain → `VERIFY_SSL=False` in config (intentional).
- Dead hosts `beta1.` / `www.` rewritten to live domain automatically.
- ~34 images + ~5 files are hard 404s at the source (2015-era, deleted) — unrecoverable, not a bug.
- Defaults: ChromaDB · text=all-MiniLM-L6-v2 · images=CLIP clip-ViT-B-32 · LLM=claude-opus-4-8.
<!-- ============================================================= -->

Two-stage project:
1. **`giki_scraper/`** — archives the entire [giki.edu.pk](https://giki.edu.pk)
   website: page text, images (with captions/alt), and files (PDFs/docs).
2. **`giki_rag/`** — a **multimodal RAG** over that archive: ask a question,
   retrieve relevant text **and** images, and get a cited answer from Claude.

Each folder has its own detailed `README.md`.

## Resuming on another PC
The scraped data and vector store are **not** in this repo (they're several GB
and regenerable). To continue elsewhere:

```bash
git clone <this-repo-url>
cd "Osaid Internship"

# 1. Scraper
cd giki_scraper
pip install -r requirements.txt
python collect_urls.py      # rebuild the URL list (~1 min)
python scrape.py            # (re)build the archive — resumable
python verify.py            # audit completeness

# 2. RAG
cd ../giki_rag
pip install -r requirements.txt
setx ANTHROPIC_API_KEY "sk-ant-..."   # Windows (new shell after)
python ingest.py --reset    # build the vector store from the archive
python chat.py "What programs does GIKI offer?"
```

> The scraped archive is regenerated by re-running the scraper. If you want the
> *exact* data moved between machines instead of re-scraping, copy the
> `giki_scraper/giki_scrape/` and `giki_rag/chroma_db/` folders directly (e.g.
> via cloud storage) — they're deliberately excluded from git.

## What's tracked in git
Only the **code** (scripts, configs, requirements, docs). The generated data
(`giki_scrape/`, `chroma_db/`, caches, logs) is git-ignored on purpose.
