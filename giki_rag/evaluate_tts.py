"""
Round-trip evaluation of TTS engines.

WER / ROUGE / BERTScore compare text against text — they can't score audio
directly. So each candidate TTS engine is scored by round-trip: synthesize
speech from a known reference text, transcribe that speech back with
faster-whisper (stt.py, fixed across all backends so it isn't a confound),
then compare the round-trip transcript against the original reference. This
approximates intelligibility using the exact metrics requested, plus raw
synthesis time and real-time factor (synthesis_time / audio_duration).

Reference texts are real GIKI RAG answers (not generic sentences), reusing
generate_eval_csv.py's benchmark questions — that's literally the text this
system would actually speak aloud in production.

Output: giki_rag_tts_evaluation.csv (+ printed summary + explicit winner).

Run:
    python evaluate_tts.py
"""
import csv
import time
import wave

import jiwer
from rouge_score import rouge_scorer

import config
import tts
import stt
from generate_eval_csv import QUESTIONS
from chat import answer

BACKENDS = ["edge", "piper", "pyttsx3"]
N_REFERENCE_TEXTS = 8   # subset of the 20 benchmark questions' generated answers


def _wav_duration_seconds(path):
    """Duration for wav files (piper/pyttsx3). None for mp3 (edge) — RTF is a
    secondary metric, not worth a new dependency (e.g. mutagen) just for this."""
    if str(path).lower().endswith(".mp3"):
        return None
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _build_reference_texts():
    """Run a subset of the real 20-question benchmark and clean the answers."""
    texts = []
    for query, category in QUESTIONS[:N_REFERENCE_TEXTS]:
        print(f"[refs] generating reference answer for: {query}")
        result = answer(query, stream_to_stdout=False, return_retrieval=False)
        cleaned = tts.strip_citations(result)
        if cleaned:
            texts.append((query, category, cleaned))
    return texts


def _bert_scorer():
    # Lazy import: bert-score pulls in a transformer model on first call.
    from bert_score import score as bert_score_fn
    return bert_score_fn


def evaluate():
    reference_texts = _build_reference_texts()
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bert_score_fn = _bert_scorer()

    rows = []
    per_backend = {b: {"wer": [], "rouge": [], "bert_f1": [], "synth_s": [], "rtf": []}
                    for b in BACKENDS}

    for backend in BACKENDS:
        print(f"\n=== Backend: {backend} ===")
        for i, (query, category, ref_text) in enumerate(reference_texts, 1):
            print(f"[{backend} {i}/{len(reference_texts)}] {query[:60]}")
            row = {"Backend": backend, "No.": i, "Query": query, "Category": category,
                   "Reference Text": ref_text}
            try:
                start = time.perf_counter()
                audio_path = tts.synthesize(ref_text, backend=backend)
                synth_s = time.perf_counter() - start
            except Exception as e:
                # edge-tts needs internet; don't let one backend's failure kill the run.
                print(f"  SKIPPED ({backend}): {e}")
                row["Status"] = f"skipped: {e}"
                rows.append(row)
                continue

            duration_s = _wav_duration_seconds(audio_path)
            rtf = (synth_s / duration_s) if duration_s else None

            transcript = stt.transcribe(audio_path)
            wer = jiwer.wer(ref_text, transcript)
            rouge_l = rouge.score(ref_text, transcript)["rougeL"].fmeasure
            _, _, bert_f1 = bert_score_fn([transcript], [ref_text], lang="en",
                                          model_type="sentence-transformers/all-MiniLM-L6-v2",
                                          num_layers=6, verbose=False)
            bert_f1 = float(bert_f1.mean())

            row.update({
                "Status": "ok",
                "Round-trip Transcript": transcript,
                "WER": round(wer, 4),
                "ROUGE-L F1": round(rouge_l, 4),
                "BERTScore F1": round(bert_f1, 4),
                "Synthesis Time (s)": round(synth_s, 3),
                "Audio Duration (s)": round(duration_s, 2) if duration_s else "N/A (mp3)",
                "Real-Time Factor": round(rtf, 3) if rtf else "N/A (mp3)",
            })
            rows.append(row)

            per_backend[backend]["wer"].append(wer)
            per_backend[backend]["rouge"].append(rouge_l)
            per_backend[backend]["bert_f1"].append(bert_f1)
            per_backend[backend]["synth_s"].append(synth_s)
            if rtf:
                per_backend[backend]["rtf"].append(rtf)

    return rows, per_backend


def _avg(xs):
    return sum(xs) / len(xs) if xs else None


def main():
    print(f"Running TTS round-trip evaluation on {N_REFERENCE_TEXTS} reference "
          f"texts x {len(BACKENDS)} backends (STT: faster-whisper "
          f"'{config.STT_MODEL_SIZE}')\n")

    rows, per_backend = evaluate()

    fieldnames = ["Backend", "No.", "Query", "Category", "Status",
                  "Reference Text", "Round-trip Transcript", "WER", "ROUGE-L F1",
                  "BERTScore F1", "Synthesis Time (s)", "Audio Duration (s)",
                  "Real-Time Factor"]
    out_path = "giki_rag_tts_evaluation.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("  TTS EVALUATION SUMMARY (lower WER = better)")
    print("=" * 60)
    summary = {}
    for backend in BACKENDS:
        m = per_backend[backend]
        avg_wer = _avg(m["wer"])
        summary[backend] = avg_wer
        if avg_wer is None:
            print(f"  {backend:10s}: no successful runs (skipped/failed)")
            continue
        avg_rtf = _avg(m["rtf"])
        rtf_str = f"{avg_rtf:.2f}" if avg_rtf else "N/A"
        print(f"  {backend:10s}: WER={avg_wer:.3f}  ROUGE-L={_avg(m['rouge']):.3f}  "
              f"BERTScore={_avg(m['bert_f1']):.3f}  "
              f"synth={_avg(m['synth_s']):.2f}s  RTF={rtf_str}")

    valid = {b: w for b, w in summary.items() if w is not None}
    if valid:
        winner = min(valid, key=valid.get)
        print(f"\n  WINNER (lowest WER): {winner}")
        print(f'  -> set TTS_BACKEND = "{winner}" in config.py')
    print("=" * 60)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
