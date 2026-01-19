#!/usr/bin/env python3
"""
Download open biomedical knowledge graphs for drug repurposing research.

This script downloads and prepares the following knowledge graphs:
- DRKG (Drug Repurposing Knowledge Graph)
- Hetionet
- PrimeKG

Usage:
    python scripts/download_graphs.py [--all] [--drkg] [--hetionet] [--primekg]
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Knowledge graph sources
KNOWLEDGE_GRAPHS = {
    "drkg": {
        "name": "Drug Repurposing Knowledge Graph",
        "url": "https://dgl-data.s3-accelerate.amazonaws.com/dataset/DRKG/drkg.tar.gz",
        "description": "97K entities, 5.8M edges from 6 databases including DrugBank",
        "license": "Apache 2.0",
        "citation": "https://github.com/gnn4dr/DRKG",
    },
    "hetionet": {
        "name": "Hetionet",
        "url": "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2",
        "description": "47K nodes, 2.2M edges integrating 29 public resources",
        "license": "CC0 1.0",
        "citation": "https://het.io/",
    },
    "primekg": {
        "name": "PrimeKG",
        "repo": "https://github.com/mims-harvard/PrimeKG",
        "url": "https://dataverse.harvard.edu/api/access/datafile/6180620",
        "filename": "kg.csv",
        "description": "129K nodes, 8M+ edges for precision medicine",
        "license": "MIT",
        "citation": "https://github.com/mims-harvard/PrimeKG",
    },
}


def download_file(url: str, dest_path: Path, desc: Optional[str] = None) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        desc = desc or dest_path.name
        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract tar.gz or zip archive."""
    try:
        if archive_path.suffix == ".gz" or str(archive_path).endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(dest_dir)
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
        elif archive_path.suffix == ".bz2":
            import bz2
            output_path = dest_dir / archive_path.stem
            with bz2.open(archive_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    f_out.write(f_in.read())
        else:
            logger.warning(f"Unknown archive format: {archive_path}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False


def download_drkg() -> bool:
    """Download DRKG (Drug Repurposing Knowledge Graph)."""
    logger.info("Downloading DRKG...")

    kg_info = KNOWLEDGE_GRAPHS["drkg"]
    dest_dir = DATA_DIR / "drkg"
    dest_dir.mkdir(parents=True, exist_ok=True)

    archive_path = dest_dir / "drkg.tar.gz"

    if (dest_dir / "drkg.tsv").exists():
        logger.info("DRKG already downloaded, skipping...")
        return True

    if not download_file(kg_info["url"], archive_path, "DRKG"):
        return False

    logger.info("Extracting DRKG...")
    if not extract_archive(archive_path, dest_dir):
        return False

    # Clean up archive
    archive_path.unlink(missing_ok=True)

    logger.success(f"DRKG downloaded to {dest_dir}")
    return True


def download_hetionet() -> bool:
    """Download Hetionet."""
    logger.info("Downloading Hetionet...")

    kg_info = KNOWLEDGE_GRAPHS["hetionet"]
    dest_dir = DATA_DIR / "hetionet"
    dest_dir.mkdir(parents=True, exist_ok=True)

    archive_path = dest_dir / "hetionet-v1.0.json.bz2"
    json_path = dest_dir / "hetionet-v1.0.json"

    if json_path.exists():
        logger.info("Hetionet already downloaded, skipping...")
        return True

    if not download_file(kg_info["url"], archive_path, "Hetionet"):
        return False

    logger.info("Extracting Hetionet...")
    if not extract_archive(archive_path, dest_dir):
        return False

    # Clean up archive
    archive_path.unlink(missing_ok=True)

    logger.success(f"Hetionet downloaded to {dest_dir}")
    return True


def download_primekg() -> bool:
    """Download PrimeKG."""
    logger.info("Downloading PrimeKG...")

    kg_info = KNOWLEDGE_GRAPHS["primekg"]
    dest_dir = DATA_DIR / "primekg"
    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dest_dir / "kg.csv"

    if csv_path.exists():
        logger.info("PrimeKG already downloaded, skipping...")
        return True

    if not download_file(kg_info["url"], csv_path, "PrimeKG"):
        return False

    logger.success(f"PrimeKG downloaded to {dest_dir}")
    return True


def print_summary():
    """Print summary of available knowledge graphs."""
    logger.info("\n" + "=" * 60)
    logger.info("AVAILABLE KNOWLEDGE GRAPHS FOR DRUG REPURPOSING")
    logger.info("=" * 60)

    for kg_id, kg_info in KNOWLEDGE_GRAPHS.items():
        dest_dir = DATA_DIR / kg_id
        status = "Downloaded" if dest_dir.exists() and any(dest_dir.iterdir()) else "Not downloaded"

        logger.info(f"\n{kg_info['name']} ({kg_id})")
        logger.info(f"  Status: {status}")
        logger.info(f"  Description: {kg_info['description']}")
        logger.info(f"  License: {kg_info['license']}")
        logger.info(f"  Citation: {kg_info['citation']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download biomedical knowledge graphs for drug repurposing research"
    )
    parser.add_argument("--all", action="store_true", help="Download all knowledge graphs")
    parser.add_argument("--drkg", action="store_true", help="Download DRKG")
    parser.add_argument("--hetionet", action="store_true", help="Download Hetionet")
    parser.add_argument("--primekg", action="store_true", help="Download PrimeKG")
    parser.add_argument("--list", action="store_true", help="List available knowledge graphs")

    args = parser.parse_args()

    if args.list:
        print_summary()
        return

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which graphs to download
    download_all = args.all or not (args.drkg or args.hetionet or args.primekg)

    results = {}

    if download_all or args.drkg:
        results["DRKG"] = download_drkg()

    if download_all or args.hetionet:
        results["Hetionet"] = download_hetionet()

    if download_all or args.primekg:
        results["PrimeKG"] = download_primekg()

    # Print results
    logger.info("\n" + "=" * 40)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 40)

    for name, success in results.items():
        status = "Success" if success else "Failed"
        logger.info(f"  {name}: {status}")

    # Print next steps
    logger.info("\nNext steps:")
    logger.info("  1. Run: python src/ingest/build_unified_graph.py")
    logger.info("  2. Explore: jupyter lab notebooks/01_explore_graphs.ipynb")


if __name__ == "__main__":
    main()
