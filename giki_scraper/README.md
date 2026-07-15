# GIKI Website Scraper

Archives **giki.edu.pk** — every page's text, images (with captions/alt text),
and downloadable files (PDFs, docs) — into a clean, indexed folder structure.

## Tools used
| Tool | Purpose |
|------|---------|
| **requests** | Download HTML pages, images, and files (with retries + rate limiting) |
| **BeautifulSoup + lxml** | Parse HTML; extract text, images, captions, links |
| **Python stdlib** (`xml`, `csv`, `json`, `pathlib`, `argparse`, `logging`) | Sitemap parsing, output files, manifests, logging |

## How it works
1. **`collect_urls.py`** reads the site's `sitemap_index.xml` and walks every
   child sitemap (pages, posts, courses, faculty, events, departments, …) to
   build a complete list of URLs → `giki_scrape/urls.txt` (3,300+ pages).
2. **`scrape.py`** visits each URL and extracts:
   - **Text**: title, meta description, headings, and full body text
   - **Images**: real URLs (handles WordPress lazy-loading), downloaded to
     `images/`, each recorded with its **caption**, **alt text**, and title
   - **Files**: PDFs/docs linked on the page, downloaded to `files/`
   - **Raw HTML**: archived under `html/`

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python collect_urls.py           # Step 1: build the URL list (once)
python scrape.py                 # Step 2: scrape everything (resumable)

# Useful options:
python scrape.py --limit 20      # test on the first 20 pages
python scrape.py --url <URL>     # scrape a single page
```
The scraper is **resumable** — stop it anytime (Ctrl+C) and re-run; it skips
pages already done and never re-downloads an asset.

## Output (`giki_scrape/`)
```
text/<slug>.txt      human-readable page text
text/<slug>.json     structured data (title, text, images+captions, files)
html/<slug>.html     raw HTML archive
images/              all downloaded images (deduplicated)
files/               all downloaded PDFs / documents
manifest.json / .csv index of every page
images.csv           every image + caption + alt + source page
files.csv            every file + source page
scrape.log           run log
```

## Notes
- **Politeness**: 1 request/second, real User-Agent, respects `robots.txt`
  (only `/wp-admin/` is disallowed). Tune `REQUEST_DELAY` in `config.py`.
- **SSL**: the server ships an incomplete certificate chain, so verification is
  disabled for this read-only archive (see `config.py` → `VERIFY_SSL`).
- **Dead staging host**: old posts reference `beta1.giki.edu.pk`; those assets
  are auto-rewritten to the live domain (see `config.py` → `HOST_REWRITES`).
