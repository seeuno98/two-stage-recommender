"""Download the RetailRocket dataset from Kaggle."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path


DATASET_NAME = "retailrocket/ecommerce-dataset"
RAW_DATA_DIR = Path("data/raw/retailrocket")
EXPECTED_FILES = {
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
    "category_tree.csv",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the dataset even if the expected files already exist.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download the dataset but do not extract the zip archive.",
    )
    return parser.parse_args()


def expected_files_exist(data_dir: Path) -> bool:
    """Return whether all expected raw CSV files exist."""

    return all((data_dir / filename).exists() for filename in EXPECTED_FILES)


def ensure_kaggle_cli() -> str:
    """Return the Kaggle CLI path or raise a helpful error."""

    kaggle_path = shutil.which("kaggle")
    if kaggle_path is None:
        raise RuntimeError(
            "Kaggle CLI is not installed or not on PATH. Install it with "
            "`pip install kaggle` and ensure `kaggle` is available."
        )
    return kaggle_path


def validate_kaggle_credentials() -> None:
    """Ensure Kaggle credentials are configured locally."""

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if kaggle_json.exists() or (username and key):
        return
    raise RuntimeError(
        "Kaggle credentials were not found. Place `kaggle.json` under "
        "`~/.kaggle/kaggle.json` with `chmod 600`, or set "
        "`KAGGLE_USERNAME` and `KAGGLE_KEY`."
    )


def run_kaggle_download(data_dir: Path) -> Path:
    """Run the Kaggle CLI download command and return the downloaded zip path."""

    ensure_kaggle_cli()
    validate_kaggle_credentials()
    data_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_NAME,
        "-p",
        str(data_dir),
    ]
    print(f"[download] downloading {DATASET_NAME} into {data_dir}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"Kaggle download failed: {stderr}") from exc

    zip_files = sorted(data_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(
            f"Download completed but no zip archive was found in {data_dir}."
        )
    return zip_files[0]


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a downloaded zip archive."""

    print(f"[download] extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def main() -> None:
    """Download and extract the RetailRocket dataset."""

    args = parse_args()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if expected_files_exist(RAW_DATA_DIR) and not args.force:
        print(f"[download] dataset already present at {RAW_DATA_DIR}")
        return

    if args.force:
        for filename in EXPECTED_FILES:
            file_path = RAW_DATA_DIR / filename
            if file_path.exists():
                file_path.unlink()

    zip_path = run_kaggle_download(RAW_DATA_DIR)
    print(f"[download] archive ready at {zip_path}")

    if args.skip_extract:
        print("[download] skipping extraction as requested")
        return

    extract_zip(zip_path, RAW_DATA_DIR)
    missing_files = [name for name in EXPECTED_FILES if not (RAW_DATA_DIR / name).exists()]
    if missing_files:
        missing = ", ".join(sorted(missing_files))
        raise FileNotFoundError(
            f"Extraction completed but expected files are missing: {missing}"
        )
    print(f"[download] extracted RetailRocket files into {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
