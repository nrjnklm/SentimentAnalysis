"""
Stage 1: Download and convert the official Stanford IMDb Large Movie Review Dataset.

Downloads aclImdb_v1.tar.gz from Stanford AI Lab, extracts it, and converts the
labeled train/test folders into local CSV files:
  - dataset/imdb_train.csv  (25,000 training reviews)
  - dataset/imdb_test.csv   (25,000 test reviews)

Each CSV has columns: review, sentiment

The official train/test split is preserved:
  - train/pos + train/neg -> imdb_train.csv
  - test/pos  + test/neg  -> imdb_test.csv

The unsup (unlabeled) folder is skipped.
"""

import csv
import os
import tarfile
import tempfile
import urllib.request

DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(DATASET_DIR, "imdb_train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "imdb_test.csv")

POS_LABEL = "positive"
NEG_LABEL = "negative"


def download_dataset(dest_path):
    """Download the IMDb tar.gz archive to dest_path."""
    print(f"Downloading IMDb dataset from: {DATASET_URL}")
    print(f"Destination: {dest_path}")
    urllib.request.urlretrieve(DATASET_URL, dest_path)
    print("Download complete.")


def extract_archive(archive_path, extract_dir):
    """Extract the tar.gz archive into extract_dir."""
    print(f"Extracting archive to: {extract_dir}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete.")


def load_folder_to_rows(folder_path, label):
    """Read all .txt files in folder_path and return [(text, label), ...]."""
    rows = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(folder_path, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        rows.append((text, label))
    return rows


def write_csv(rows, csv_path):
    """Write [(text, label)] rows to a CSV with columns review, sentiment."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review", "sentiment"])
        writer.writerows(rows)


def main():
    # Step 1: Download archive to a temp directory
    tmp_dir = tempfile.mkdtemp(prefix="imdb_download_")
    archive_path = os.path.join(tmp_dir, "aclImdb_v1.tar.gz")

    try:
        download_dataset(archive_path)

        # Step 2: Extract
        extract_dir = os.path.join(tmp_dir, "aclImdb_extracted")
        extract_archive(archive_path, extract_dir)

        # The archive contains a top-level aclImdb/ folder
        acl_dir = os.path.join(extract_dir, "aclImdb")
        if not os.path.isdir(acl_dir):
            # Fallback: maybe the archive extracted directly to extract_dir
            acl_dir = extract_dir

        train_dir = os.path.join(acl_dir, "train")
        test_dir = os.path.join(acl_dir, "test")

        # Step 3: Load training data (preserve official split)
        train_pos = load_folder_to_rows(os.path.join(train_dir, "pos"), POS_LABEL)
        train_neg = load_folder_to_rows(os.path.join(train_dir, "neg"), NEG_LABEL)
        train_rows = train_pos + train_neg

        # Step 4: Load test data (preserve official split)
        test_pos = load_folder_to_rows(os.path.join(test_dir, "pos"), POS_LABEL)
        test_neg = load_folder_to_rows(os.path.join(test_dir, "neg"), NEG_LABEL)
        test_rows = test_pos + test_neg

        # Step 5: Write CSVs
        write_csv(train_rows, TRAIN_CSV)
        write_csv(test_rows, TEST_CSV)

        # Step 6: Verification report
        print("\n" + "=" * 60)
        print("IMDB DATASET CONVERSION - VERIFICATION REPORT")
        print("=" * 60)

        # Row counts
        print(f"\nTrain CSV: {TRAIN_CSV}")
        print(f"  Total rows: {len(train_rows)}")
        print(f"  Positive:   {len(train_pos)}")
        print(f"  Negative:   {len(train_neg)}")

        print(f"\nTest CSV: {TEST_CSV}")
        print(f"  Total rows: {len(test_rows)}")
        print(f"  Positive:   {len(test_pos)}")
        print(f"  Negative:   {len(test_neg)}")

        # Empty review check
        train_empty = sum(1 for t, _ in train_rows if not t)
        test_empty = sum(1 for t, _ in test_rows if not t)
        print(f"\nEmpty reviews in train: {train_empty}")
        print(f"Empty reviews in test:  {test_empty}")

        # Train/test overlap check
        train_texts = set(t for t, _ in train_rows)
        test_texts = set(t for t, _ in test_rows)
        overlap = train_texts & test_texts
        print(f"\nTrain/test overlap (exact duplicate reviews): {len(overlap)}")

        # Sample reviews
        print("\nSample train positive reviews:")
        for text, _ in train_pos[:3]:
            print(f"  '{text[:120]}...'" if len(text) > 120 else f"  '{text}'")

        print("\nSample train negative reviews:")
        for text, _ in train_neg[:3]:
            print(f"  '{text[:120]}...'" if len(text) > 120 else f"  '{text}'")

        print("\nSample test positive reviews:")
        for text, _ in test_pos[:3]:
            print(f"  '{text[:120]}...'" if len(text) > 120 else f"  '{text}'")

        print("\nSample test negative reviews:")
        for text, _ in test_neg[:3]:
            print(f"  '{text[:120]}...'" if len(text) > 120 else f"  '{text}'")

        print("\n" + "=" * 60)
        print("Stage 1 complete. IMDb dataset ready in dataset/imdb_train.csv and dataset/imdb_test.csv")
        print("=" * 60)

    finally:
        # Clean up temp download directory
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary download directory: {tmp_dir}")


if __name__ == "__main__":
    main()
