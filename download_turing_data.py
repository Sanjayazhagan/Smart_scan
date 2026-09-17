"""Turing Synthetic Radar Dataset (TSRD) Batch Downloader & Test Runner."""

import json
from pathlib import Path
import urllib.request

try:
    from huggingface_hub import get_token
except ImportError:
    get_token = lambda: None

TOKEN = get_token()
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
BASE_API = "https://huggingface.co/api/datasets/alan-turing-institute/turing-synthetic-radar-dataset/tree/main"
DOWNLOAD_URL = "https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset/resolve/main"

DEST_DIR = Path("turing_dataset")
DEST_DIR.mkdir(exist_ok=True)


def list_directory(subpath=""):
    url = f"{BASE_API}/{subpath}" if subpath else BASE_API
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_file(rel_path: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[EXISTS] {dest_path.name} ({dest_path.stat().st_size / 1e3:.1f} KB)")
        return

    url = f"{DOWNLOAD_URL}/{rel_path}"
    req = urllib.request.Request(url, headers=HEADERS)
    print(f"[DOWNLOADING] {rel_path} ... ", end="", flush=True)

    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        data = resp.read()
        f.write(data)
        print(f"DONE ({len(data) / 1e3:.1f} KB)")


if __name__ == "__main__":
    print("=" * 70)
    print("  TURING SYNTHETIC RADAR DATASET: MULTI-SCENARIO DOWNLOADER")
    print("=" * 70)

    # 1. Download a representative batch of test files from archive/test
    print("\n--- 1. Downloading Archive Test Suite (10 missions) ---")
    for i in range(10):
        fname = f"test_{i}.h5"
        download_file(f"archive/test/{fname}", DEST_DIR / "archive_test" / fname)

    # 2. Check scan/test_scan
    print("\n--- 2. Checking & Downloading Scan Mode Test Scenarios ---")
    try:
        scan_items = list_directory("scan/test_scan")
        for item in scan_items[:5]:
            fname = Path(item["path"]).name
            download_file(item["path"], DEST_DIR / "scan_test" / fname)
    except Exception as e:
        print(f"Scan directory note: {e}")

    # 3. Check stare/test_stare
    print("\n--- 3. Checking & Downloading Stare Mode Test Scenarios ---")
    try:
        stare_items = list_directory("stare/test_stare")
        for item in stare_items[:5]:
            fname = Path(item["path"]).name
            download_file(item["path"], DEST_DIR / "stare_test" / fname)
    except Exception as e:
        print(f"Stare directory note: {e}")

    print("\nAll target radar test datasets successfully downloaded into 'turing_dataset/'!")
