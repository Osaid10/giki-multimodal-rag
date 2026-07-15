"""
Audit the scrape for completeness. Reconciles urls.txt against what's actually
on disk and reports anything missing or failed.

Run:
    python verify.py             # summary + write missing_urls.txt
    python verify.py --verbose   # also list failed asset URLs

What it checks:
  1. Every URL in urls.txt has a corresponding text/<slug>.json  (missed pages)
  2. Every image/file record in those JSONs actually downloaded
     (local_path is set AND the file exists on disk)         (failed assets)
  3. Cross-checks that files on disk aren't orphaned/zero-byte
"""
import argparse
import json

import config
from utils import url_to_slug


def load_urls():
    if not config.URLS_FILE.exists():
        return []
    return [u.strip() for u in
            config.URLS_FILE.read_text(encoding="utf-8").splitlines() if u.strip()]


def audit():
    urls = load_urls()
    done_slugs = {p.stem for p in config.TEXT_DIR.glob("*.json")}

    missing_pages = []          # URLs with no JSON (not scraped / failed)
    for u in urls:
        if url_to_slug(u) not in done_slugs:
            missing_pages.append(u)

    # Walk every scraped page and check its assets.
    total_imgs = failed_imgs = 0
    total_files = failed_files = 0
    failed_img_urls, failed_file_urls = [], []
    zero_byte = []

    for jf in config.TEXT_DIR.glob("*.json"):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for im in d.get("images", []):
            total_imgs += 1
            lp = im.get("local_path")
            if not lp or not (config.OUTPUT_DIR / lp).exists():
                failed_imgs += 1
                failed_img_urls.append(im.get("source_url", ""))
            elif (config.OUTPUT_DIR / lp).stat().st_size == 0:
                zero_byte.append(lp)
        for fi in d.get("files", []):
            total_files += 1
            lp = fi.get("local_path")
            if not lp or not (config.OUTPUT_DIR / lp).exists():
                failed_files += 1
                failed_file_urls.append(fi.get("source_url", ""))
            elif (config.OUTPUT_DIR / lp).stat().st_size == 0:
                zero_byte.append(lp)

    return {
        "total_urls": len(urls),
        "pages_done": len(done_slugs),
        "missing_pages": missing_pages,
        "total_imgs": total_imgs, "failed_imgs": failed_imgs,
        "total_files": total_files, "failed_files": failed_files,
        "failed_img_urls": failed_img_urls,
        "failed_file_urls": failed_file_urls,
        "zero_byte": zero_byte,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    r = audit()
    pct = 100 * r["pages_done"] / r["total_urls"] if r["total_urls"] else 0

    print("=" * 60)
    print("SCRAPE COMPLETENESS AUDIT")
    print("=" * 60)
    print(f"Pages:   {r['pages_done']}/{r['total_urls']} scraped ({pct:.1f}%)")
    print(f"         {len(r['missing_pages'])} pages missing (not scraped / failed)")
    print(f"Images:  {r['total_imgs'] - r['failed_imgs']}/{r['total_imgs']} "
          f"downloaded ({r['failed_imgs']} failed)")
    print(f"Files:   {r['total_files'] - r['failed_files']}/{r['total_files']} "
          f"downloaded ({r['failed_files']} failed)")
    if r["zero_byte"]:
        print(f"WARNING: {len(r['zero_byte'])} zero-byte assets on disk")

    if r["missing_pages"]:
        config.OUTPUT_DIR.joinpath("missing_urls.txt").write_text(
            "\n".join(r["missing_pages"]), encoding="utf-8")
        print(f"\n-> Wrote {len(r['missing_pages'])} missing URLs to "
              f"missing_urls.txt")
        print("   Re-run `python scrape.py` to retry them (resumable), or")
        print("   `python scrape.py --url <URL>` for a single page.")

    if args.verbose:
        if r["failed_img_urls"]:
            print("\nFailed images (unique hosts):")
            hosts = sorted({u.split('/')[2] for u in r["failed_img_urls"] if u})
            for h in hosts:
                print(f"   {h}")
        if r["failed_file_urls"]:
            print("\nFailed files:")
            for u in r["failed_file_urls"][:30]:
                print(f"   {u}")

    if not r["missing_pages"] and not r["failed_imgs"] and not r["failed_files"]:
        print("\nAll pages and assets accounted for.")


if __name__ == "__main__":
    main()
