"""Download the Dominick's soft-drink files from the Kilts Center into ``data/raw/``.

Usage:  uv run python scripts/download_data.py [--force]

The data are provided by the James M. Kilts Center for Marketing, University of Chicago Booth
School of Business, for academic research use only. They are never committed to this repository.
If a download fails (URL change, terms page, etc.) this script stops and prints the manual steps
documented in data/README.md -- it does not try to scrape around the problem.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beverage_demand.config import path, settings  # noqa: E402

EXPECTED = {
    "wsdr.csv": 1_000_000_000,   # soft-drink movement file, ~1.2 GB unzipped
    "upcsdr.csv": 50_000,
    "demo.dta": 200_000,
}


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "text/html" in ctype:
            raise RuntimeError(f"{url} returned an HTML page (terms/login page?), not a data file")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args()

    cfg = settings()["download"]
    raw = path("raw_dir")
    for name, rel in cfg["files"].items():
        dest = raw / name
        url = rel if rel.startswith("http") else cfg["base_url"] + rel
        if dest.exists() and not args.force:
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {url}")
        try:
            download(url, dest)
        except Exception as exc:  # noqa: BLE001 -- any failure -> manual instructions
            print(f"\nDownload failed: {exc}\nFollow the manual steps in data/README.md.")
            return 1

    for z in raw.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            print(f"[unzip] {z.name}: {zf.namelist()}")
            zf.extractall(raw)

    missing = [f for f, size in EXPECTED.items()
               if not (raw / f).exists() or (raw / f).stat().st_size < size]
    if missing:
        print(f"Missing or truncated files: {missing}. See data/README.md.")
        return 1
    print("All raw files present in data/raw/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
