#!/usr/bin/env python3
"""
Data Download Script for LDPC Steganography System
Downloads sample datasets and pre-trained models
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import hashlib
import json
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data sources configuration
DATA_SOURCES = {
    "sample_images": {
        "url": "https://github.com/your-username/ldpc-steganography-data/releases/download/v1.0/sample_images.zip",
        "filename": "sample_images.zip",
        "extract_to": "data/",
        "description": "Sample images for testing (100 images, ~50MB)",
        "size_mb": 50,
        "md5": "abc123def456ghi789jkl012mno345pq"
    },
    "coco_subset": {
        "url": "https://github.com/your-username/ldpc-steganography-data/releases/download/v1.0/coco_subset.tar.gz",
        "filename": "coco_subset.tar.gz", 
        "extract_to": "data/",
        "description": "COCO validation subset (5000 images, ~2GB)",
        "size_mb": 2048,
        "md5": "def456ghi789jkl012mno345pqr678stu"
    },
    "celeba_subset": {
        "url": "https://github.com/your-username/ldpc-steganography-data/releases/download/v1.0/celeba_subset.tar.gz",
        "filename": "celeba_subset.tar.gz",
        "extract_to": "data/",
        "description": "CelebA subset (10000 images, ~1.5GB)",
        "size_mb": 1536,
        "md5": "ghi789jkl012mno345pqr678stu901vwx"
    },
    "pretrained_models": {
        "url": "https://github.com/your-username/ldpc-steganography-data/releases/download/v1.0/pretrained_models.zip",
        "filename": "pretrained_models.zip",
        "extract_to": "results/models/",
        "description": "Pre-trained LDPC steganography models (~500MB)",
        "size_mb": 512,
        "md5": "jkl012mno345pqr678stu901vwx234yz0"
    }
}

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def calculate_md5(filepath):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, filepath, expected_md5=None, description="file"):
    """Download a file with progress bar and verification"""
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {description}...")
    logger.info(f"URL: {url}")
    logger.info(f"Destination: {filepath}")
    
    # Check if file already exists and is valid
    if filepath.exists() and expected_md5:
        logger.info("File exists, verifying...")
        if calculate_md5(filepath) == expected_md5:
            logger.info("✅ File already exists and is valid")
            return True
        else:
            logger.warning("❌ File exists but MD5 mismatch, re-downloading...")
            filepath.unlink()
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=description) as t:
            urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
        
        # Verify download
        if expected_md5:
            logger.info("Verifying download...")
            actual_md5 = calculate_md5(filepath)
            if actual_md5 == expected_md5:
                logger.info("✅ Download verified successfully")
            else:
                logger.error(f"❌ MD5 mismatch! Expected: {expected_md5}, Got: {actual_md5}")
                filepath.unlink()
                return False
        
        logger.info(f"✅ Successfully downloaded {description}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        if filepath.exists():
            filepath.unlink()
        return False

def extract_archive(archive_path, extract_to, description="archive"):
    """Extract archive file"""
    
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting {description}...")
    
    try:
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix.lower() in ['.tar', '.gz'] or archive_path.name.endswith('.tar.gz'):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.error(f"❌ Unsupported archive format: {archive_path.suffix}")
            return False
        
        logger.info(f"✅ Successfully extracted {description}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        return False

def download_and_extract(data_key, keep_archive=False):
    """Download and extract a dataset"""
    
    if data_key not in DATA_SOURCES:
        logger.error(f"❌ Unknown data source: {data_key}")
        return False
    
    source = DATA_SOURCES[data_key]
    
    # Download
    archive_path = Path("downloads") / source["filename"]
    success = download_file(
        url=source["url"],
        filepath=archive_path,
        expected_md5=source.get("md5"),
        description=source["description"]
    )
    
    if not success:
        return False
    
    # Extract
    success = extract_archive(
        archive_path=archive_path,
        extract_to=source["extract_to"],
        description=source["description"]
    )
    
    if not success:
        return False
    
    # Clean up archive if requested
    if not keep_archive:
        archive_path.unlink()
        logger.info(f"🗑️ Removed archive: {archive_path}")
    
    return True

def list_available_datasets():
    """List all available datasets"""
    
    print("📦 Available Datasets:\n")
    
    for key, source in DATA_SOURCES.items():
        print(f"🔹 {key}")
        print(f"   Description: {source['description']}")
        print(f"   Size: ~{source['size_mb']} MB")
        print(f"   Extract to: {source['extract_to']}")
        print()

def create_data_structure():
    """Create standard data directory structure"""
    
    logger.info("📁 Creating data directory structure...")
    
    directories = [
        "data/train",
        "data/val", 
        "data/test",
        "data/raw",
        "downloads",
        "results/models",
        "results/runs",
        "results/logs",
        "results/figures"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep file
        gitkeep = Path(directory) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    
    logger.info("✅ Data directory structure created")

def check_disk_space(required_mb):
    """Check available disk space"""
    
    import shutil
    free_space_bytes = shutil.disk_usage(".").free
    free_space_mb = free_space_bytes / (1024 * 1024)
    
    if free_space_mb < required_mb:
        logger.error(f"❌ Insufficient disk space! Required: {required_mb}MB, Available: {free_space_mb:.0f}MB")
        return False
    
    logger.info(f"✅ Sufficient disk space available: {free_space_mb:.0f}MB")
    return True

def download_sample_data():
    """Download minimal sample data for testing"""
    
    logger.info("🎯 Downloading sample data for quick start...")
    
    # Check disk space
    if not check_disk_space(100):  # 100MB buffer
        return False
    
    # Download sample images
    success = download_and_extract("sample_images", keep_archive=False)
    
    if success:
        logger.info("🎉 Sample data downloaded successfully!")
        logger.info("You can now run basic tests and examples.")
    else:
        logger.error("❌ Failed to download sample data")
    
    return success

def download_full_datasets():
    """Download all available datasets"""
    
    logger.info("📥 Downloading all datasets...")
    
    # Calculate total size
    total_size = sum(source["size_mb"] for source in DATA_SOURCES.values())
    
    if not check_disk_space(total_size + 1000):  # 1GB buffer
        return False
    
    success_count = 0
    
    for key in DATA_SOURCES.keys():
        logger.info(f"\n--- Downloading {key} ---")
        
        if download_and_extract(key, keep_archive=False):
            success_count += 1
            logger.info(f"✅ {key} completed")
        else:
            logger.error(f"❌ {key} failed")
    
    logger.info(f"\n📊 Download Summary: {success_count}/{len(DATA_SOURCES)} successful")
    
    if success_count == len(DATA_SOURCES):
        logger.info("🎉 All datasets downloaded successfully!")
        return True
    else:
        logger.warning("⚠️ Some downloads failed")
        return False

def verify_data_integrity():
    """Verify integrity of downloaded data"""
    
    logger.info("🔍 Verifying data integrity...")
    
    # Check if expected directories exist
    expected_dirs = [
        "data/train",
        "data/val",
        "data/test"
    ]
    
    for directory in expected_dirs:
        path = Path(directory)
        if path.exists():
            file_count = len(list(path.glob("*")))
            logger.info(f"✅ {directory}: {file_count} files")
        else:
            logger.warning(f"⚠️ {directory}: Not found")
    
    # Check models
    models_dir = Path("results/models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        logger.info(f"✅ Pre-trained models: {len(model_files)} files")
    else:
        logger.warning("⚠️ No pre-trained models found")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="Download datasets and models for LDPC Steganography System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py --sample-only          # Download only sample data
  python download_data.py --all                  # Download all datasets
  python download_data.py --datasets sample_images coco_subset
  python download_data.py --list                 # List available datasets
  python download_data.py --verify               # Verify downloaded data
        """
    )
    
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Download only sample data for testing"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Download all available datasets"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATA_SOURCES.keys()),
        help="Download specific datasets"
    )
    
    parser.add_argument(
        "--list",
        action="store_true", 
        help="List available datasets"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify integrity of downloaded data"
    )
    
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archive files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Change to output directory
    if args.output_dir != ".":
        os.chdir(args.output_dir)
    
    # Create data structure
    create_data_structure()
    
    # Handle different modes
    if args.list:
        list_available_datasets()
        return
    
    if args.verify:
        verify_data_integrity()
        return
    
    if args.sample_only:
        success = download_sample_data()
        sys.exit(0 if success else 1)
    
    if args.all:
        success = download_full_datasets()
        sys.exit(0 if success else 1)
    
    if args.datasets:
        success_count = 0
        for dataset in args.datasets:
            logger.info(f"\n--- Downloading {dataset} ---")
            if download_and_extract(dataset, keep_archive=args.keep_archives):
                success_count += 1
        
        success = (success_count == len(args.datasets))
        logger.info(f"\n📊 Downloaded {success_count}/{len(args.datasets)} datasets")
        sys.exit(0 if success else 1)
    
    # Default behavior
    logger.info("No specific action requested. Use --help for options.")
    list_available_datasets()

if __name__ == "__main__":
    main()