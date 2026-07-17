"""
Step 4 — Multimodal RAG answers.

Retrieves relevant text + images for a question, then asks a vision-capable LLM
to answer using ONLY that context, citing source page URLs. Retrieved images are
passed to the model as real images, so it reasons over visuals (photos, charts,
posters) — not just their captions.

Two interchangeable backends (set LLM_BACKEND in config.py):
  "ollama"    -> local, free, offline (Qwen2.5-VL)   [default]
  "anthropic" -> Claude via API (needs ANTHROPIC_API_KEY)

Run:
    python chat.py                       # interactive Q&A loop
    python chat.py "your question here"  # single question
"""
import base64
import json
import mimetypes
import sys

import requests

import config
from retrieve import retrieve

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the Ghulam Ishaq "
    "Khan Institute (GIKI), using ONLY the provided context retrieved from "
    "giki.edu.pk. Rules:\n"
    "- Base every claim on the supplied text and images. If the context does "
    "not contain the answer, say so plainly — do not invent facts.\n"
    "- Cite the source page URL(s) you used, in the form [source: <url>].\n"
    "- When an image is relevant, refer to it and use its caption.\n"
    "- Be concise and accurate."
)

# Image formats both backends accept.
OK_MEDIA = ("image/jpeg", "image/png", "image/gif", "image/webp")


# --------------------------------------------------------------------------
# Shared context assembly
# --------------------------------------------------------------------------
def _usable_images(retrieved):
    """Retrieved images that exist on disk and are a supported format."""
    out = []
    for im in retrieved["images"]:
        if len(out) >= config.MAX_IMAGES_TO_LLM:
            break
        path = config.IMAGE_DIR / im["local_path"]
        if not path.exists():
            continue
        media_type, _ = mimetypes.guess_type(str(path))
        if media_type not in OK_MEDIA:
            continue
        out.append((im, path, media_type))
    return out


def _text_context(retrieved):
    parts = ["# Retrieved text context\n"]
    for i, t in enumerate(retrieved["text"], 1):
        parts.append(f"[{i}] (source: {t['url']})\nTitle: {t['title']}\n"
                     f"{t['text']}\n")
    return "\n".join(parts)


def _caption_lines(images):
    if not images:
        return ""
    lines = ["\n# Attached images"]
    for i, (im, _, _) in enumerate(images, 1):
        caption = im["caption"] or im["alt"] or "(no caption)"
        lines.append(f"Image {i} (source: {im['url']}) — caption: {caption}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Backend: Ollama (local, free)
# --------------------------------------------------------------------------
def _answer_ollama(query, retrieved, stream_to_stdout):
    images = _usable_images(retrieved)
    prompt = (f"{_text_context(retrieved)}"
              f"{_caption_lines(images)}"
              f"\n\n# Question\n{query}")

    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt,
             # Ollama takes images as raw base64 strings on the message.
             "images": [base64.standard_b64encode(p.read_bytes()).decode()
                        for _, p, _ in images]},
        ],
        "stream": True,
    }

    try:
        resp = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload,
                             stream=True, timeout=config.OLLAMA_TIMEOUT)
    except requests.ConnectionError:
        raise SystemExit(
            f"Cannot reach Ollama at {config.OLLAMA_URL}.\n"
            "Start it with:  ollama serve\n"
            f"And make sure the model is pulled:  ollama pull {config.OLLAMA_MODEL}")
    resp.raise_for_status()

    chunks = []
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "error" in data:
            raise SystemExit(f"Ollama error: {data['error']}")
        piece = data.get("message", {}).get("content", "")
        chunks.append(piece)
        if stream_to_stdout and piece:
            print(piece, end="", flush=True)
        if data.get("done"):
            break
    if stream_to_stdout:
        print()
    return "".join(chunks)


# --------------------------------------------------------------------------
# Backend: Anthropic (Claude via API)
# --------------------------------------------------------------------------
def _answer_anthropic(query, retrieved, stream_to_stdout):
    import anthropic

    if not config.ANTHROPIC_API_KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY, or use LLM_BACKEND='ollama'.")

    images = _usable_images(retrieved)
    blocks = [{"type": "text", "text": _text_context(retrieved)}]
    for im, path, media_type in images:
        caption = im["caption"] or im["alt"] or "(no caption)"
        blocks.append({"type": "text",
                       "text": f"\nImage (source: {im['url']}) — caption: {caption}"})
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type,
                       "data": base64.standard_b64encode(path.read_bytes()).decode()},
        })
    blocks.append({"type": "text", "text": f"\n# Question\n{query}"})

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    with client.messages.stream(
        model=config.ANSWER_MODEL,
        max_tokens=config.MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": blocks}],
    ) as stream:
        if stream_to_stdout:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            print()
        final = stream.get_final_message()
    return "".join(b.text for b in final.content if b.type == "text")


# --------------------------------------------------------------------------
def answer(query, stream_to_stdout=True):
    """Retrieve context and generate an answer with the configured backend."""
    retrieved = retrieve(query)

    if stream_to_stdout:
        n_img = len(_usable_images(retrieved))
        print(f"[backend: {config.LLM_BACKEND} | "
              f"{len(retrieved['text'])} text chunks, {n_img} images retrieved]\n")

    if config.LLM_BACKEND == "ollama":
        text = _answer_ollama(query, retrieved, stream_to_stdout)
    elif config.LLM_BACKEND == "anthropic":
        text = _answer_anthropic(query, retrieved, stream_to_stdout)
    else:
        raise SystemExit(f"Unknown LLM_BACKEND: {config.LLM_BACKEND}")

    if stream_to_stdout:
        print("\n--- retrieved sources ---")
        for t in retrieved["text"]:
            print(f"  text  [{t['score']:.2f}] {t['url']}")
        for im, _, _ in _usable_images(retrieved):
            print(f"  image [{im['score']:.2f}] {im['local_path']}  ({im['url']})")
    return text


def main():
    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]))
        return
    print(f"GIKI RAG ({config.LLM_BACKEND}) — ask a question "
          f"(empty line or Ctrl+C to quit).\n")
    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        answer(q)
        print()


if __name__ == "__main__":
    main()
