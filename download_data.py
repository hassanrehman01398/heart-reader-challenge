"""
Download PTB-XL v1.0.3 from PhysioNet.
Uses the official ZIP archive for reliability.
"""
import os
import sys
import zipfile
from pathlib import Path
import urllib.request

DATASET_URL = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)
DL_DIR   = Path("./data")
ZIP_PATH = DL_DIR / "ptbxl.zip"
OUT_DIR  = DL_DIR / "ptbxl"


def _reporthook(count, block_size, total_size):
    pct = min(100, int(count * block_size * 100 / max(total_size, 1)))
    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
    sys.stdout.write(f"\r  [{bar}] {pct}%  ({count * block_size / 1e6:.1f} MB)")
    sys.stdout.flush()


def download():
    DL_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"ZIP already present at {ZIP_PATH}, skipping download.")
    else:
        print(f"Downloading PTB-XL (~1.7 GB) …")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH, reporthook=_reporthook)
        print()

    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        print(f"Data already extracted at {OUT_DIR}, skipping extraction.")
        return

    print(f"Extracting to {OUT_DIR} …")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        members = z.namelist()
        n = len(members)
        for i, member in enumerate(members, 1):
            # Strip the top-level folder from the zip path
            parts = member.split("/", 1)
            dest = parts[1] if len(parts) > 1 else parts[0]
            if not dest:
                continue
            dest_path = OUT_DIR / dest
            if member.endswith("/"):
                dest_path.mkdir(parents=True, exist_ok=True)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
            if i % 2000 == 0:
                print(f"  {i}/{n} files …")

    print("Extraction complete.")
    print(f"Dataset ready at: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    download()
