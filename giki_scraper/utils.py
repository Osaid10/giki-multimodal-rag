"""
Shared helpers: a polite HTTP session, filename sanitising, and logging.
"""
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
import urllib3

import config

# The target server has an incomplete cert chain (see config.VERIFY_SSL).
# Suppress the noisy warning we'd otherwise get on every request.
if not config.VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Logging ---------------------------------------------------------------
def get_logger():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("giki")
    if logger.handlers:                      # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%H:%M:%S")

    file_h = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)

    logger.addHandler(file_h)
    logger.addHandler(stream_h)
    return logger


log = get_logger()

# --- Polite HTTP session ---------------------------------------------------
_session = None


def get_session():
    """A single reused session with our User-Agent and connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": config.USER_AGENT})
    return _session


def fetch(url, *, binary=False):
    """
    Fetch a URL politely with retries and a delay. Returns the Response object,
    or None if it ultimately failed. Sleeps REQUEST_DELAY after each attempt.
    """
    session = get_session()
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=config.REQUEST_TIMEOUT,
                               stream=binary, verify=config.VERIFY_SSL)
            time.sleep(config.REQUEST_DELAY)          # be gentle
            if resp.status_code == 200:
                return resp
            log.warning(f"HTTP {resp.status_code} for {url}")
            if resp.status_code in (404, 403, 410):
                return None                            # no point retrying
        except requests.exceptions.SSLError as exc:
            log.warning(f"SSL error for {url}: {exc}")
            return None
        except requests.exceptions.ConnectionError as exc:
            # DNS / host-unreachable failures won't fix themselves on retry.
            log.warning(f"Cannot reach {url} (dead host?): "
                        f"{str(exc).splitlines()[0][:80]}")
            return None
        except requests.RequestException as exc:
            log.warning(f"Attempt {attempt}/{config.MAX_RETRIES} failed for "
                        f"{url}: {exc}")
            time.sleep(config.REQUEST_DELAY * attempt)  # back off
    log.error(f"Giving up on {url}")
    return None


# --- Filename / path helpers ----------------------------------------------
def rewrite_url(url):
    """Rewrite dead/staging hosts to production and upgrade http->https."""
    p = urlparse(url)
    host = p.netloc.lower()
    new_host = config.HOST_REWRITES.get(host, host)
    scheme = p.scheme
    if new_host in config.FORCE_HTTPS_HOSTS and scheme == "http":
        scheme = "https"
    if new_host == host and scheme == p.scheme:
        return url
    return p._replace(scheme=scheme, netloc=new_host).geturl()


def is_allowed(url):
    """True if the URL is on the target domain and not in a disallowed path."""
    p = urlparse(url)
    if p.netloc and p.netloc.lower() not in config.ALLOWED_DOMAINS:
        return False
    return not any(p.path.startswith(pre) for pre in config.DISALLOWED_PREFIXES)


def slugify(text, max_len=120):
    """Turn arbitrary text into a safe filename fragment."""
    text = unquote(text)
    text = re.sub(r"[^\w\-. ]", "_", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text.strip())
    text = text.strip("._-") or "index"
    return text[:max_len]


def url_to_slug(url):
    """Make a readable, unique filename slug from a page URL."""
    p = urlparse(url)
    path = p.path.strip("/")
    slug = slugify(path.replace("/", "__")) if path else "home"
    return slug


def unique_path(directory, filename):
    """Return a path inside `directory` for `filename`, avoiding overwrites."""
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / filename
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 1
    while True:
        candidate = directory / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1
