"""
Step 3 — Ingest scraped data into ChromaDB.

Reads every per-page JSON produced by the scraper, splits page text into
overlapping chunks, embeds text (MiniLM) and images (CLIP), and stores both in
two Chroma collections with rich metadata for citation.

Run:
    python ingest.py              # ingest everything
    python ingest.py --limit 50   # ingest first 50 pages (quick test)
    python ingest.py --reset      # wipe collections and re-ingest from scratch
"""
import argparse
import json

import chromadb

import config
import embeddings as emb


# --------------------------------------------------------------------------
def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Split text into overlapping character windows on paragraph boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = start + size
        # Try to break on a paragraph/sentence boundary near the window end.
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def get_collections(reset=False):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if reset:
        for name in (config.TEXT_COLLECTION, config.IMAGE_COLLECTION):
            try:
                client.delete_collection(name)
            except Exception:
                pass
    text_col = client.get_or_create_collection(
        config.TEXT_COLLECTION, metadata={"hnsw:space": config.DISTANCE})
    img_col = client.get_or_create_collection(
        config.IMAGE_COLLECTION, metadata={"hnsw:space": config.DISTANCE})
    return text_col, img_col


def load_pages(limit=None):
    files = sorted(config.TEXT_JSON_DIR.glob("*.json"))
    if limit:
        files = files[:limit]
    for f in files:
        try:
            yield json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue


# --------------------------------------------------------------------------
def ingest_text(pages, text_col):
    ids, docs, metas = [], [], []
    for page in pages:
        chunks = chunk_text(page.get("text", ""))
        for i, ch in enumerate(chunks):
            ids.append(f"{page['slug']}::chunk{i}")
            docs.append(ch)
            metas.append({
                "url": page["url"],
                "title": page.get("title", ""),
                "chunk_index": i,
            })
    if not ids:
        print("No text chunks to ingest.")
        return 0

    print(f"Embedding {len(docs)} text chunks...")
    # Embed and upsert in batches to keep memory bounded.
    B = 256
    for s in range(0, len(ids), B):
        vecs = emb.embed_texts(docs[s:s + B])
        text_col.upsert(ids=ids[s:s + B], embeddings=vecs,
                        documents=docs[s:s + B], metadatas=metas[s:s + B])
        print(f"  text {min(s + B, len(ids))}/{len(ids)}")
    return len(ids)


def ingest_images(pages, img_col):
    # Collect unique images (dedupe by local file path).
    records, seen = [], set()
    for page in pages:
        for im in page.get("images", []):
            local = im.get("local_path")
            if not local or local in seen:
                continue
            path = config.IMAGE_DIR / local
            if not path.exists():
                continue
            seen.add(local)
            records.append({
                "path": str(path),
                "url": page["url"],
                "page_title": page.get("title", ""),
                "source_url": im.get("source_url", ""),
                "caption": im.get("caption", ""),
                "alt": im.get("alt", ""),
                "local_path": local,
            })
    if not records:
        print("No images to ingest.")
        return 0

    print(f"Embedding {len(records)} images with CLIP...")
    B = 128
    total = 0
    for s in range(0, len(records), B):
        batch = records[s:s + B]
        vecs, ok_paths = emb.embed_images([r["path"] for r in batch])
        if not vecs:
            continue
        ok_set = set(ok_paths)
        batch_ok = [r for r in batch if r["path"] in ok_set]
        ids = [r["local_path"] for r in batch_ok]
        # A short document string helps text-only inspection / fallback.
        docs = [(r["caption"] or r["alt"] or r["page_title"]) for r in batch_ok]
        metas = [{k: r[k] for k in
                 ("url", "page_title", "source_url", "caption", "alt", "local_path")}
                 for r in batch_ok]
        img_col.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)
        total += len(ids)
        print(f"  images {min(s + B, len(records))}/{len(records)}")
    return total


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Ingest scraped GIKI data into ChromaDB")
    ap.add_argument("--limit", type=int, help="only ingest the first N pages")
    ap.add_argument("--reset", action="store_true", help="wipe collections first")
    args = ap.parse_args()

    pages = list(load_pages(args.limit))
    print(f"Loaded {len(pages)} pages from {config.TEXT_JSON_DIR}")
    if not pages:
        print("Nothing to ingest yet — run the scraper first.")
        return

    text_col, img_col = get_collections(reset=args.reset)
    n_text = ingest_text(pages, text_col)
    n_img = ingest_images(pages, img_col)
    print(f"\nDone. Indexed {n_text} text chunks and {n_img} images.")
    print(f"Vector store: {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
