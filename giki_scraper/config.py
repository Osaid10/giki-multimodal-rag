"""
Shared configuration for the GIKI website scraper.
Edit values here to change behaviour without touching the scraper logic.
"""
from pathlib import Path

# --- Target site -----------------------------------------------------------
BASE_URL = "https://giki.edu.pk"
SITEMAP_INDEX = f"{BASE_URL}/sitemap_index.xml"
ALLOWED_DOMAINS = {"giki.edu.pk", "www.giki.edu.pk"}

# --- Politeness / being a good citizen -------------------------------------
# Delay (seconds) between requests so we don't hammer the university server.
REQUEST_DELAY = 1.0
REQUEST_TIMEOUT = 30          # seconds before a request is considered failed
MAX_RETRIES = 3

# giki.edu.pk serves an INCOMPLETE certificate chain (missing intermediate
# cert) — a server-side misconfiguration. Browsers hide this via AIA fetching,
# but Python's requests cannot verify it. We only DOWNLOAD public data (never
# send credentials), so disabling verification for this read-only archive is
# safe. Flip back to True if the university fixes their cert chain.
VERIFY_SSL = False
USER_AGENT = (
    "GIKI-Archive-Bot/1.0 (Educational internship project; "
    "contact: 133695601+Osaid10@users.noreply.github.com)"
)

# robots.txt only disallows /wp-admin/ — we skip anything under these prefixes.
DISALLOWED_PREFIXES = ("/wp-admin/",)

# Old posts reference a now-dead staging subdomain; the same assets live on the
# production domain at the identical path. Rewrite these hosts before download.
HOST_REWRITES = {
    "beta1.giki.edu.pk": "giki.edu.pk",
    "www.giki.edu.pk": "giki.edu.pk",
}
FORCE_HTTPS_HOSTS = {"giki.edu.pk"}   # upgrade http -> https for these hosts

# --- File types we treat as downloadable "files" ---------------------------
FILE_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".csv", ".txt", ".rtf", ".odt",
}
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".tiff", ".ico",
}

# --- Output layout ---------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "giki_scrape"
TEXT_DIR = OUTPUT_DIR / "text"
IMAGE_DIR = OUTPUT_DIR / "images"
FILE_DIR = OUTPUT_DIR / "files"
HTML_DIR = OUTPUT_DIR / "html"

URLS_FILE = OUTPUT_DIR / "urls.txt"
MANIFEST_JSON = OUTPUT_DIR / "manifest.json"
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"
IMAGES_CSV = OUTPUT_DIR / "images.csv"
FILES_CSV = OUTPUT_DIR / "files.csv"
LOG_FILE = OUTPUT_DIR / "scrape.log"

# Set to True to also save the raw HTML of every page.
SAVE_RAW_HTML = True
