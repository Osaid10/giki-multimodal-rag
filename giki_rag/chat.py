"""
Step 4 — Multimodal RAG answers.

Retrieve relevant text + images for a question, then ask Claude (Opus 4.8) to
answer using ONLY that context, with citations to source page URLs. Retrieved
images are passed to Claude as real image blocks, so it can reason over visuals
(photos, charts, posters), not just their captions.

Run:
    python chat.py                       # interactive Q&A loop
    python chat.py "your question here"  # single question

Requires:  set ANTHROPIC_API_KEY in your environment.
"""
import base64
import mimetypes
import sys

import anthropic

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


def _image_block(local_path):
    """Build an Anthropic image content block from a local file, or None."""
    path = config.IMAGE_DIR / local_path
    if not path.exists():
        return None
    media_type, _ = mimetypes.guess_type(str(path))
    if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        return None                      # Claude supports these image types
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def build_content(query, retrieved):
    """Assemble the multimodal user message: text context + image blocks + question."""
    blocks = []

    # --- Text context ---
    text_ctx = ["# Retrieved text context\n"]
    for i, t in enumerate(retrieved["text"], 1):
        text_ctx.append(f"[{i}] (source: {t['url']})\nTitle: {t['title']}\n"
                        f"{t['text']}\n")
    blocks.append({"type": "text", "text": "\n".join(text_ctx)})

    # --- Image context (real image blocks + a caption line each) ---
    sent = 0
    for im in retrieved["images"]:
        if sent >= config.MAX_IMAGES_TO_LLM:
            break
        block = _image_block(im["local_path"])
        if block is None:
            continue
        caption = im["caption"] or im["alt"] or "(no caption)"
        blocks.append({"type": "text",
                       "text": f"\nImage (source: {im['url']}) — caption: {caption}"})
        blocks.append(block)
        sent += 1

    blocks.append({"type": "text", "text": f"\n# Question\n{query}"})
    return blocks


def answer(query, stream_to_stdout=True):
    if not config.ANTHROPIC_API_KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY in your environment first.")

    retrieved = retrieve(query)
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    content = build_content(query, retrieved)

    with client.messages.stream(
        model=config.ANSWER_MODEL,
        max_tokens=config.MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        if stream_to_stdout:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            print()
        final = stream.get_final_message()

    # Show which sources fed the answer.
    used_imgs = [im for im in retrieved["images"]
                 if (config.IMAGE_DIR / im["local_path"]).exists()]
    if stream_to_stdout:
        print("\n--- retrieved sources ---")
        for t in retrieved["text"]:
            print(f"  text  [{t['score']:.2f}] {t['url']}")
        for im in used_imgs[:config.MAX_IMAGES_TO_LLM]:
            print(f"  image [{im['score']:.2f}] {im['local_path']}  ({im['url']})")
    return final


def main():
    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]))
        return
    print("GIKI RAG — ask a question (Ctrl+C or empty line to quit).\n")
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
