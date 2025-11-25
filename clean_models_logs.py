#!/usr/bin/env python3
"""
Script to clean up training logs and model directories.
Deletes: models/, ppo_model_logs/, ppo_tensorboard/, sac_model_logs/, sac_tensorboard/
"""

import os
import shutil
from pathlib import Path

# Directories to delete
DIRECTORIES_TO_DELETE = [
    "ppo_model_logs",
    "ppo_tensorboard",
    "sac_model_logs",
    "sac_tensorboard",
    "sample_img"
]


def delete_directory(dir_path):
    """
    Delete a directory if it exists.

    Args:
        dir_path: Path to the directory to delete
    """
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"✓ Deleted: {dir_path}")
            return True
        except Exception as e:
            print(f"✗ Error deleting {dir_path}: {e}")
            return False
    else:
        print(f"⊘ Not found: {dir_path}")
        return False


def main():
    """Main function to clean up directories."""
    # Get the script's directory (project root)
    script_dir = Path(__file__).resolve().parent

    print("Cleaning up training logs and model directories...")
    print(f"Working directory: {script_dir}\n")

    deleted_count = 0
    for dir_name in DIRECTORIES_TO_DELETE:
        dir_path = script_dir / dir_name
        if delete_directory(dir_path):
            deleted_count += 1

    print(f"\n{'='*60}")
    print(
        f"Cleanup complete! Deleted {deleted_count} out of {len(DIRECTORIES_TO_DELETE)} directories.")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Ask for confirmation
    print("This script will delete the following directories:")
    for dir_name in DIRECTORIES_TO_DELETE:
        print(f"  - {dir_name}/")

    response = input("\nAre you sure you want to proceed? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        main()
    else:
        print("Cleanup cancelled.")
