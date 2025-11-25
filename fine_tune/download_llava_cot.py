"""
download_llava_cot.py

Custom download script for LLaVA-CoT-100k dataset that handles split zip files.
This provides download functionality without modifying the core cobra download module.
"""
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import requests
from rich.progress import BarColumn, DownloadColumn, MofNCompleteColumn, Progress, TextColumn, TransferSpeedColumn
from tqdm import tqdm

from cobra.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def download_with_progress(url: str, download_dir: Path, chunk_size_bytes: int = 1024) -> Path:
    """Utility function for downloading files from the internet, with a handy Rich-based progress bar."""
    overwatch.info(f"Downloading {(dest_path := download_dir / Path(url).name)} from `{url}`", ctx_level=1)
    if dest_path.exists():
        return dest_path

    # Otherwise --> fire an HTTP Request, with `stream = True`
    response = requests.get(url, stream=True)

    # Download w/ Transfer-Aware Progress
    with Progress(
        TextColumn("[bold]{task.description} - {task.fields[fname]}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        transient=True,
    ) as dl_progress:
        dl_tid = dl_progress.add_task(
            "Downloading", fname=dest_path.name, total=int(response.headers.get("content-length", "None"))
        )
        with open(dest_path, "wb") as f:
            for data in response.iter_content(chunk_size=chunk_size_bytes):
                dl_progress.advance(dl_tid, f.write(data))

    return dest_path


def extract_with_progress(archive_path: Path, download_dir: Path, extract_type: str = "directory", cleanup: bool = False) -> Path:
    """Utility function for extracting compressed archives, with a handy Rich-based progress bar."""
    assert archive_path.suffix == ".zip", "Only `.zip` compressed archives are supported for now!"
    overwatch.info(f"Extracting {archive_path.name} to `{download_dir}`", ctx_level=1)

    # Extract w/ Progress
    with Progress(
        TextColumn("[bold]{task.description} - {task.fields[aname]}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        MofNCompleteColumn(),
        transient=True,
    ) as ext_progress:
        with ZipFile(archive_path) as zf:
            ext_tid = ext_progress.add_task("Extracting", aname=archive_path.name, total=len(members := zf.infolist()))
            extract_path = Path(zf.extract(members[0], download_dir))
            if extract_type == "file":
                assert len(members) == 1, f"Archive `{archive_path}` with extract type `{extract_type} has > 1 member!"
            elif extract_type == "directory":
                for member in members[1:]:
                    zf.extract(member, download_dir)
                    ext_progress.advance(ext_tid)
            else:
                raise ValueError(f"Extract type `{extract_type}` for archive `{archive_path}` is not defined!")

    # Cleanup (if specified)
    if cleanup:
        archive_path.unlink()

    return extract_path


def merge_split_zip_parts(parts_dir: Path, part_files: list, merged_name: str) -> Path:
    """
    Merge split zip file parts into a single zip file.
    
    :param parts_dir: Directory containing the split parts
    :param part_files: List of part file suffixes (e.g., ["aa", "ab", "ac"])
    :param merged_name: Name of the merged zip file
    :return: Path to the merged zip file
    """
    merged_path = parts_dir / merged_name
    if merged_path.exists():
        overwatch.info(f"Merged zip file already exists: {merged_path}")
        return merged_path
    
    overwatch.info(f"Merging {len(part_files)} zip parts into {merged_name}...")
    
    # Merge parts in order
    with open(merged_path, "wb") as merged_file:
        for part_suffix in tqdm(part_files, desc="Merging parts"):
            part_path = parts_dir / f"{merged_name}.part-{part_suffix}"
            if not part_path.exists():
                raise FileNotFoundError(f"Part file not found: {part_path}")
            
            with open(part_path, "rb") as part_file:
                shutil.copyfileobj(part_file, merged_file)
    
    overwatch.info(f"Successfully merged {len(part_files)} parts into {merged_path}")
    return merged_path


def download_llava_cot_dataset(root_dir: Path = Path("data")) -> None:
    """
    Download LLaVA-CoT-100k dataset including JSONL file and split image zip files.
    
    :param root_dir: Root directory for datasets
    """
    download_dir = root_dir / "download" / "llava-cot-100k"
    os.makedirs(download_dir, exist_ok=True)
    
    # Dataset URLs
    base_url = "https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k/resolve/main"
    train_jsonl_url = f"{base_url}/train.jsonl"
    image_zip_base = f"{base_url}/image.zip"
    
    # Part suffixes: aa, ab, ac, ..., ap (16 parts)
    part_suffixes = ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak", "al", "am", "an", "ao", "ap"]
    
    # Step 1: Download train.jsonl
    overwatch.info("Downloading train.jsonl...")
    train_jsonl_path = download_dir / "train.jsonl"
    if not train_jsonl_path.exists():
        download_with_progress(train_jsonl_url, download_dir)
        # Rename if needed
        downloaded_name = Path(train_jsonl_url).name
        if (download_dir / downloaded_name).exists() and downloaded_name != "train.jsonl":
            shutil.move(download_dir / downloaded_name, train_jsonl_path)
    else:
        overwatch.info("train.jsonl already exists, skipping download")
    
    # Step 2: Download all image zip parts
    overwatch.info(f"Downloading {len(part_suffixes)} image zip parts...")
    for part_suffix in part_suffixes:
        part_name = f"image.zip.part-{part_suffix}"
        part_path = download_dir / part_name
        if not part_path.exists():
            part_url = f"{image_zip_base}.part-{part_suffix}"
            overwatch.info(f"Downloading {part_name}...")
            download_with_progress(part_url, download_dir)
            # Rename if needed
            downloaded_name = Path(part_url).name
            if (download_dir / downloaded_name).exists() and downloaded_name != part_name:
                shutil.move(download_dir / downloaded_name, part_path)
        else:
            overwatch.info(f"{part_name} already exists, skipping")
    
    # Step 3: Merge zip parts
    overwatch.info("Merging image zip parts...")
    merged_zip = merge_split_zip_parts(download_dir, part_suffixes, "image.zip")
    
    # Step 4: Extract merged zip
    overwatch.info("Extracting image zip...")
    extract_with_progress(merged_zip, download_dir, extract_type="directory", cleanup=True)
    
    overwatch.info("✓ LLaVA-CoT-100k dataset download complete!")
    overwatch.info(f"Dataset location: {download_dir}")
    overwatch.info(f"  - train.jsonl: {train_jsonl_path}")
    overwatch.info(f"  - Images: {download_dir / 'images'}")


if __name__ == "__main__":
    download_llava_cot_dataset()

