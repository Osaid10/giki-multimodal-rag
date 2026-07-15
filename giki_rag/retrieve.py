"""
Retrieval: given a question, pull the most relevant text chunks and images
from ChromaDB. Text uses MiniLM; images are searched cross-modally with CLIP
(the text query is embedded into CLIP space and matched against image vectors).
"""
import chromadb

import config
import embeddings as emb


def _client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def retrieve_text(query, k=config.TOP_K_TEXT):
    col = _client().get_or_create_collection(config.TEXT_COLLECTION)
    res = col.query(query_embeddings=[emb.embed_query_text(query)], n_results=k)
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                               res["distances"][0]):
        out.append({
            "text": doc,
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "score": 1 - dist,          # cosine distance -> similarity
        })
    return out


def retrieve_images(query, k=config.TOP_K_IMAGES):
    col = _client().get_or_create_collection(config.IMAGE_COLLECTION)
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=[emb.embed_query_for_images(query)],
                    n_results=k)
    out = []
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        score = 1 - dist
        if score < config.MIN_IMAGE_SCORE:
            continue
        out.append({
            "local_path": meta.get("local_path", ""),
            "caption": meta.get("caption", ""),
            "alt": meta.get("alt", ""),
            "url": meta.get("url", ""),
            "page_title": meta.get("page_title", ""),
            "source_url": meta.get("source_url", ""),
            "score": score,
        })
    return out


def retrieve(query):
    """Return {'text': [...], 'images': [...]} for a query."""
    return {"text": retrieve_text(query), "images": retrieve_images(query)}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What programs does GIKI offer?"
    r = retrieve(q)
    print(f"\nQUERY: {q}\n")
    print("=== TEXT ===")
    for t in r["text"]:
        print(f"[{t['score']:.2f}] {t['title'][:60]}\n    {t['text'][:150]}...\n")
    print("=== IMAGES ===")
    for im in r["images"]:
        print(f"[{im['score']:.2f}] {im['local_path']}  caption={im['caption'][:50]}")
