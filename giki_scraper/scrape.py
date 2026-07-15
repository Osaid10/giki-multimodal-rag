"""
Step 2 — Scrape every page: text, images (+ captions), and downloadable files.

Run:
    python scrape.py                 # scrape everything in urls.txt (resumable)
    python scrape.py --limit 10      # only the first 10 pages (good for testing)
    python scrape.py --url <URL>     # scrape a single page

Outputs (under giki_scrape/):
    text/<slug>.txt      human-readable text of the page
    text/<slug>.json     structured data (title, text, images+captions, files)
    html/<slug>.html     raw HTML archive
    images/              downloaded image files
    files/               downloaded PDFs / documents
    manifest.json/.csv   index of every page
    images.csv           every image with its caption / alt / source page
    files.csv            every downloaded file with its source page
"""
import argparse
import csv
import json
import os
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

import config
from utils import (fetch, slugify, url_to_slug, unique_path, rewrite_url, log)

# Tags whose text is noise, not page content.
JUNK_TAGS = ["script", "style", "noscript", "nav", "header", "footer",
             "form", "svg", "iframe"]

# Attributes that may hold the real (lazy-loaded) image URL.
LAZY_ATTRS = ["data-src", "data-lazy-src", "data-original", "data-large_image", "src"]

# A global index so we never download the same asset twice across pages.
# { source_url : local_relative_path }
_asset_index = {}
ASSET_INDEX_FILE = config.OUTPUT_DIR / "asset_index.json"


# --------------------------------------------------------------------------
# Asset (image / file) downloading
# --------------------------------------------------------------------------
def load_asset_index():
    global _asset_index
    if ASSET_INDEX_FILE.exists():
        _asset_index = json.loads(ASSET_INDEX_FILE.read_text(encoding="utf-8"))


def save_asset_index():
    ASSET_INDEX_FILE.write_text(json.dumps(_asset_index, indent=2),
                                encoding="utf-8")


def _asset_filename(url):
    """A readable filename derived from the asset URL's path."""
    path = urlparse(url).path
    name = os.path.basename(path) or "asset"
    stem, ext = os.path.splitext(name)
    return slugify(stem, max_len=100) + ext.lower()


def download_asset(url, dest_dir):
    """
    Download an image or file to dest_dir (deduped by URL).
    Returns the path relative to OUTPUT_DIR, or None on failure.
    """
    if not url or url.startswith("data:"):
        return None
    url = rewrite_url(url)          # fix dead staging host, force https
    if url in _asset_index:
        existing = config.OUTPUT_DIR / _asset_index[url]
        if existing.exists():
            return _asset_index[url]

    resp = fetch(url, binary=True)
    if resp is None:
        return None

    dest = unique_path(dest_dir, _asset_filename(url))
    try:
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(8192):
                fh.write(chunk)
    except OSError as exc:
        log.warning(f"Could not save {url}: {exc}")
        return None

    rel = str(dest.relative_to(config.OUTPUT_DIR)).replace("\\", "/")
    _asset_index[url] = rel
    return rel


# --------------------------------------------------------------------------
# HTML extraction
# --------------------------------------------------------------------------
def extract_text(soup):
    """Return (title, meta_description, headings[], body_text)."""
    title = soup.title.get_text(strip=True) if soup.title else ""

    desc = ""
    md = soup.find("meta", attrs={"name": "description"}) or \
        soup.find("meta", attrs={"property": "og:description"})
    if md and md.get("content"):
        desc = md["content"].strip()

    # Work on a copy of the main content so we don't mangle image extraction.
    content = soup.find("article") or soup.body or soup
    content = BeautifulSoup(str(content), "lxml")
    for tag in content(JUNK_TAGS):
        tag.decompose()

    headings = [h.get_text(" ", strip=True)
                for h in content.find_all(["h1", "h2", "h3", "h4"])
                if h.get_text(strip=True)]

    # Join block-level text with newlines; collapse blank lines.
    raw = content.get_text("\n", strip=True)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    body_text = "\n".join(lines)
    return title, desc, headings, body_text


def find_caption(img):
    """
    Find the best caption for an <img>:
    1. <figcaption> inside the enclosing <figure>
    2. .wp-caption-text sibling/parent (classic WordPress captions)
    3. the image's own `title` attribute
    Returns "" if none found.
    """
    fig = img.find_parent("figure")
    if fig:
        cap = fig.find("figcaption")
        if cap and cap.get_text(strip=True):
            return cap.get_text(" ", strip=True)

    wp = img.find_parent(class_="wp-caption")
    if wp:
        cap = wp.find(class_="wp-caption-text")
        if cap and cap.get_text(strip=True):
            return cap.get_text(" ", strip=True)

    if img.get("title"):
        return img["title"].strip()
    return ""


def real_image_url(img, base_url):
    """Resolve the real image URL, preferring lazy-load attributes over src."""
    for attr in LAZY_ATTRS:
        val = img.get(attr, "").strip()
        if val and not val.startswith("data:"):
            return urljoin(base_url, val)
    # Fall back to the first entry of srcset.
    srcset = img.get("srcset") or img.get("data-srcset")
    if srcset:
        first = srcset.split(",")[0].strip().split(" ")[0]
        if first and not first.startswith("data:"):
            return urljoin(base_url, first)
    return None


def extract_images(soup, page_url):
    """Download every content image and return a list of records."""
    records = []
    seen = set()
    for img in soup.find_all("img"):
        src = real_image_url(img, page_url)
        if not src or src in seen:
            continue
        seen.add(src)
        local = download_asset(src, config.IMAGE_DIR)
        records.append({
            "source_url": src,
            "local_path": local,
            "caption": find_caption(img),
            "alt": (img.get("alt") or "").strip(),
            "title_attr": (img.get("title") or "").strip(),
        })
    return records


def extract_files(soup, page_url):
    """Download every linked document (PDF, docx, …) and return records."""
    records = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"].strip())
        ext = os.path.splitext(urlparse(href).path)[1].lower()
        if ext not in config.FILE_EXTENSIONS or href in seen:
            continue
        seen.add(href)
        local = download_asset(href, config.FILE_DIR)
        records.append({
            "source_url": href,
            "local_path": local,
            "link_text": a.get_text(" ", strip=True),
            "type": ext.lstrip("."),
        })
    return records


# --------------------------------------------------------------------------
# Per-page orchestration
# --------------------------------------------------------------------------
def scrape_page(url):
    """Scrape one page; return its record dict, or None if it failed."""
    slug = url_to_slug(url)
    json_path = config.TEXT_DIR / f"{slug}.json"
    if json_path.exists():                      # resume: already done
        log.info(f"skip (done): {url}")
        return json.loads(json_path.read_text(encoding="utf-8"))

    resp = fetch(url)
    if resp is None:
        return None

    html = resp.text
    soup = BeautifulSoup(html, "lxml")

    title, desc, headings, body_text = extract_text(soup)
    images = extract_images(soup, url)
    files = extract_files(soup, url)

    record = {
        "url": url,
        "slug": slug,
        "title": title,
        "description": desc,
        "headings": headings,
        "text": body_text,
        "images": images,
        "files": files,
        "num_images": len(images),
        "num_files": len(files),
    }

    # Write per-page outputs.
    config.TEXT_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    txt = [f"URL: {url}", f"TITLE: {title}"]
    if desc:
        txt.append(f"DESCRIPTION: {desc}")
    txt.append("\n" + "=" * 70 + "\n")
    txt.append(body_text)
    if images:
        txt.append("\n\n" + "-" * 30 + " IMAGES " + "-" * 30)
        for im in images:
            txt.append(f"\n[image] {im['source_url']}")
            if im["caption"]:
                txt.append(f"  caption: {im['caption']}")
            if im["alt"]:
                txt.append(f"  alt: {im['alt']}")
    (config.TEXT_DIR / f"{slug}.txt").write_text("\n".join(txt),
                                                 encoding="utf-8")

    if config.SAVE_RAW_HTML:
        config.HTML_DIR.mkdir(parents=True, exist_ok=True)
        (config.HTML_DIR / f"{slug}.html").write_text(html, encoding="utf-8")

    log.info(f"OK: {url}  ({len(images)} imgs, {len(files)} files)")
    return record


# --------------------------------------------------------------------------
# Manifest writing
# --------------------------------------------------------------------------
def write_manifests(records):
    records = [r for r in records if r]
    config.MANIFEST_JSON.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(config.MANIFEST_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["url", "title", "slug", "num_images", "num_files",
                    "description"])
        for r in records:
            w.writerow([r["url"], r["title"], r["slug"], r["num_images"],
                        r["num_files"], r["description"]])

    with open(config.IMAGES_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["page_url", "image_source_url", "local_path", "caption",
                    "alt", "title_attr"])
        for r in records:
            for im in r["images"]:
                w.writerow([r["url"], im["source_url"], im["local_path"],
                            im["caption"], im["alt"], im["title_attr"]])

    with open(config.FILES_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["page_url", "file_source_url", "local_path", "type",
                    "link_text"])
        for r in records:
            for f in r["files"]:
                w.writerow([r["url"], f["source_url"], f["local_path"],
                            f["type"], f["link_text"]])

    log.info(f"Wrote manifests for {len(records)} pages.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Scrape giki.edu.pk")
    ap.add_argument("--limit", type=int, help="only scrape the first N URLs")
    ap.add_argument("--url", help="scrape a single URL")
    args = ap.parse_args()

    load_asset_index()

    if args.url:
        urls = [args.url]
    else:
        if not config.URLS_FILE.exists():
            log.error("urls.txt not found — run collect_urls.py first.")
            return
        urls = [u.strip() for u in
                config.URLS_FILE.read_text(encoding="utf-8").splitlines()
                if u.strip()]
        if args.limit:
            urls = urls[:args.limit]

    log.info(f"Scraping {len(urls)} pages...")
    records = []
    try:
        for i, url in enumerate(urls, 1):
            log.info(f"[{i}/{len(urls)}]")
            records.append(scrape_page(url))
            if i % 25 == 0:                     # periodic checkpoint
                save_asset_index()
    except KeyboardInterrupt:
        log.warning("Interrupted — saving progress so far.")
    finally:
        save_asset_index()
        write_manifests(records)


if __name__ == "__main__":
    main()
