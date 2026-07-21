"""
Evaluation harness for the GIKI multimodal RAG system.

Runs a fixed benchmark of 20 questions through the full pipeline
(retrieve -> generate) and records, for every query, automatically-computed
quality metrics for BOTH stages:

  Retrieval quality  - how relevant the fetched context is
      * text chunks / images retrieved
      * top & average cosine similarity of the retrieved text
      * top image similarity

  Generation quality - how good the produced answer is
      * the full generated answer
      * "Answered?"  - did the model answer, or fall back to "not in context"
      * answer length (words)

  Performance
      * end-to-end latency per query (seconds)

The point of computing these automatically (instead of a "Manual Review"
column) is that the evaluation is objective and reproducible: re-running it
after any change to chunking, top-k, or the model yields comparable numbers.

Output: giki_rag_evaluation.csv  (+ an aggregate summary printed to console)

Run:
    python generate_eval_csv.py
"""
import csv
import time

import config
from chat import answer

# 20 benchmark questions spanning the main information categories a
# prospective student / visitor would ask about GIKI.
QUESTIONS = [
    ("What undergraduate programs does GIKI offer?", "Academic Programs"),
    ("Tell me about the Artificial Intelligence degree program.", "AI Program"),
    ("What engineering disciplines can I study at GIKI?", "Engineering"),
    ("What postgraduate and PhD programs are available?", "Postgraduate"),
    ("How does the admission process work and what are the requirements?", "Admissions"),
    ("What scholarships or financial aid does GIKI offer?", "Scholarships"),
    ("Tell me about the hostel and accommodation facilities.", "Hostels"),
    ("What sports and recreational facilities are on campus?", "Sports"),
    ("What laboratories and research facilities are available?", "Facilities"),
    ("What student societies and clubs can students join?", "Student Life"),
    ("What is campus life like at GIKI?", "Campus Life"),
    ("What research areas and centers does GIKI focus on?", "Research"),
    ("Who is the Rector of GIKI and who leads the institute?", "Administration"),
    ("What is the history and background of GIKI?", "About"),
    ("Where is GIKI located and how do I get there?", "Location"),
    ("What industry partnerships or collaborations does GIKI have?", "Partnerships"),
    ("What career services or job fairs does GIKI organize?", "Careers"),
    ("What notable achievements or rankings does GIKI have?", "Achievements"),
    ("What faculty departments exist at GIKI?", "Departments"),
    ("How can I contact GIKI for admissions queries?", "Contact"),
]

# Phrases a well-behaved RAG model uses when the answer isn't in the context.
_REFUSAL_MARKERS = (
    "not contain", "does not contain", "not in the context", "no information",
    "cannot find", "could not find", "not mentioned", "not available in",
    "unable to find", "not provided",
)


def _round(x, n=3):
    return round(x, n) if x is not None else ""


def evaluate():
    rows = []
    # running totals for the aggregate summary
    tot_latency = 0.0
    n_answered = 0
    all_top_scores = []

    for i, (query, category) in enumerate(QUESTIONS, 1):
        print(f"[{i:2}/{len(QUESTIONS)}] {query}")

        start = time.perf_counter()
        result = answer(query, stream_to_stdout=False, return_retrieval=True)
        latency = time.perf_counter() - start

        text = result["answer"]
        retrieved = result["retrieved"]
        text_hits = retrieved["text"]
        image_hits = retrieved["images"]

        # --- retrieval metrics ---
        text_scores = [t["score"] for t in text_hits]
        img_scores = [im["score"] for im in image_hits]
        top_text = max(text_scores) if text_scores else None
        avg_text = sum(text_scores) / len(text_scores) if text_scores else None
        top_img = max(img_scores) if img_scores else None
        retrieved_urls = sorted({t["url"] for t in text_hits if t["url"]})

        # --- generation metrics ---
        low = text.lower()
        answered = not any(m in low for m in _REFUSAL_MARKERS)
        word_count = len(text.split())

        # --- accumulate for summary ---
        tot_latency += latency
        n_answered += int(answered)
        if top_text is not None:
            all_top_scores.append(top_text)

        rows.append({
            "No.": i,
            "Query": query,
            "Category": category,
            "Generated Answer": text.replace("\n", " ").strip(),
            "Answered?": "Yes" if answered else "No (not in context)",
            "Text Chunks Retrieved": len(text_hits),
            "Top Text Similarity": _round(top_text),
            "Avg Text Similarity": _round(avg_text),
            "Images Retrieved": len(image_hits),
            "Top Image Similarity": _round(top_img),
            "Retrieved Source URLs": " | ".join(retrieved_urls),
            "Answer Length (words)": word_count,
            "Latency (s)": _round(latency, 2),
        })

    return rows, {
        "questions": len(QUESTIONS),
        "answered": n_answered,
        "avg_latency": tot_latency / len(QUESTIONS),
        "avg_top_score": (sum(all_top_scores) / len(all_top_scores)
                          if all_top_scores else 0.0),
    }


def main():
    print(f"Running RAG evaluation on {len(QUESTIONS)} questions "
          f"(backend: {config.LLM_BACKEND}, model: {config.OLLAMA_MODEL})\n")

    rows, summary = evaluate()

    fieldnames = list(rows[0].keys())
    out_path = "giki_rag_evaluation.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate scorecard — the headline numbers for the supervisor.
    n = summary["questions"]
    print("\n" + "=" * 52)
    print("  EVALUATION SUMMARY")
    print("=" * 52)
    print(f"  Questions evaluated      : {n}")
    print(f"  Answered from context    : {summary['answered']}/{n} "
          f"({100*summary['answered']/n:.0f}%)")
    print(f"  Avg top text similarity  : {summary['avg_top_score']:.3f}")
    print(f"  Avg latency per query    : {summary['avg_latency']:.2f} s")
    print("=" * 52)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
