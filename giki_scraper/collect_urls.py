"""
Step 1 — Collect every page URL from the site's sitemap index.

Run:  python collect_urls.py
Output: giki_scrape/urls.txt  (one URL per line)
"""
import xml.etree.ElementTree as ET

import config
from utils import fetch, is_allowed, log

# WordPress sitemaps use this XML namespace.
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def parse_sitemap(url):
    """Return (child_sitemaps, page_urls) found in a single sitemap XML."""
    resp = fetch(url)
    if resp is None:
        return [], []
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        log.warning(f"Could not parse sitemap {url}: {exc}")
        return [], []

    tag = root.tag.split("}")[-1]           # strip namespace
    locs = [el.text.strip() for el in root.findall(".//sm:loc", NS) if el.text]

    if tag == "sitemapindex":
        return locs, []                     # these are child sitemaps
    return [], locs                         # this is a urlset (real pages)


def collect_all():
    """Walk the sitemap index recursively and return a sorted set of page URLs."""
    to_visit = [config.SITEMAP_INDEX]
    seen_sitemaps = set()
    pages = set()

    while to_visit:
        sm = to_visit.pop()
        if sm in seen_sitemaps:
            continue
        seen_sitemaps.add(sm)
        log.info(f"Reading sitemap: {sm}")
        children, page_urls = parse_sitemap(sm)
        to_visit.extend(c for c in children if c not in seen_sitemaps)
        for u in page_urls:
            if is_allowed(u):
                pages.add(u)

    return sorted(pages)


def main():
    pages = collect_all()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.URLS_FILE.write_text("\n".join(pages), encoding="utf-8")
    log.info(f"Collected {len(pages)} page URLs -> {config.URLS_FILE}")


if __name__ == "__main__":
    main()
