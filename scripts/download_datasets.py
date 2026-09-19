import os
import sys
import time
import urllib.request
from pathlib import Path

DATASET_ROOT = Path("dataset")
RAW_DIR = DATASET_ROOT / "raw"

DATASETS = {
    "daigt": {
        "dir": RAW_DIR / "daigt",
        "files": {
            "daigt_v3_drcat.csv": "https://huggingface.co/datasets/ramensoft/daigt_v3/resolve/main/train_v3_drcat_01.csv",
        }
    },
    "hc3": {
        "dir": RAW_DIR / "hc3",
        "files": {
            "medicine.jsonl": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/medicine.jsonl",
            "wiki_csai.jsonl": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/wiki_csai.jsonl",
            "finance.jsonl": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/finance.jsonl",
            "open_qa.jsonl": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/open_qa.jsonl",
        }
    }
}


def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"  [Already exists] {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return

    print(f"  [Downloading] {url} -> {dest_path}...")
    start_time = time.time()

    def report_progress(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            pct = min(100.0, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            if block_num % 1000 == 0 or downloaded >= total_size:
                sys.stdout.write(f"\r    {mb_downloaded:.1f}/{mb_total:.1f} MB ({pct:.1f}%)")
                sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    urllib.request.urlretrieve(url, str(dest_path), reporthook=report_progress)
    elapsed = time.time() - start_time
    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"\n    Completed {dest_path.name} ({size_mb:.2f} MB in {elapsed:.1f}s)")


def main():
    print("=" * 60)
    print("PaperLab V1: Downloading DAIGT & HC3 Datasets to dataset/")
    print("=" * 60)

    for group_name, info in DATASETS.items():
        print(f"\nProcessing group: {group_name.upper()}")
        for filename, url in info["files"].items():
            dest_path = info["dir"] / filename
            try:
                download_file(url, dest_path)
            except Exception as e:
                print(f"\n  [Error] Failed to download {filename}: {e}")

    print("\n" + "=" * 60)
    print("Verifying downloaded dataset files:")
    print("=" * 60)
    total_bytes = 0
    for root, _, files in os.walk(DATASET_ROOT):
        for f in files:
            p = Path(root) / f
            size = p.stat().st_size
            total_bytes += size
            print(f" - {p.relative_to(DATASET_ROOT)} ({size / (1024*1024):.2f} MB)")

    print(f"\nTotal size in {DATASET_ROOT}/: {total_bytes / (1024*1024):.2f} MB")
    print("Download finished successfully.")


if __name__ == "__main__":
    main()
