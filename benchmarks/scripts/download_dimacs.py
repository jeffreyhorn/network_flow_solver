#!/usr/bin/env python3
"""Download DIMACS benchmark instances from the official archive.

This script downloads minimum cost flow problem instances from the DIMACS
Implementation Challenge archive and organizes them in the local benchmarks
directory structure.

Usage:
    python benchmarks/scripts/download_dimacs.py [--all | --small | --medium | --large]
    python benchmarks/scripts/download_dimacs.py --list  # Show available instances

DIMACS Archive:
    http://archive.dimacs.rutgers.edu/pub/netflow/

License:
    DIMACS benchmark instances are in the public domain for academic use.
    Always cite: Johnson & McGeoch (1993), DIMACS Series Volume 12.

Example:
    # Download small instances only
    python benchmarks/scripts/download_dimacs.py --small

    # Download all instances
    python benchmarks/scripts/download_dimacs.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# DIMACS archive base URL
DIMACS_BASE_URL = "http://archive.dimacs.rutgers.edu/pub/netflow"

# Problem families and their URLs
DIMACS_INSTANCES = {
    "netgen_small": {
        "name": "NETGEN Small Instances",
        "description": "Small NETGEN-generated problems for testing",
        "size_category": "small",
        "url_base": f"{DIMACS_BASE_URL}/netgen/small",
        "files": [
            # Format: (filename, expected_size_approx_kb, description)
            # Note: These are example entries - actual files may differ
            # Update with real DIMACS archive contents
        ],
        "local_dir": "benchmarks/problems/dimacs/netgen/small",
    },
    "netgen_medium": {
        "name": "NETGEN Medium Instances",
        "description": "Medium NETGEN-generated problems",
        "size_category": "medium",
        "url_base": f"{DIMACS_BASE_URL}/netgen/medium",
        "files": [],
        "local_dir": "benchmarks/problems/dimacs/netgen/medium",
    },
    "netgen_large": {
        "name": "NETGEN Large Instances",
        "description": "Large NETGEN-generated problems",
        "size_category": "large",
        "url_base": f"{DIMACS_BASE_URL}/netgen/large",
        "files": [],
        "local_dir": "benchmarks/problems/dimacs/netgen/large",
    },
}


def download_file(url: str, dest_path: Path, show_progress: bool = True) -> bool:
    """Download a file from a URL to a local path.

    Args:
        url: URL to download from.
        dest_path: Local path to save the file.
        show_progress: Whether to show download progress.

    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        print(f"Downloading {url}...", end=" " if not show_progress else "\n")
        sys.stdout.flush()

        with urlopen(url, timeout=30) as response:
            content = response.read()

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(content)

        file_size_kb = len(content) / 1024
        print(f"✓ ({file_size_kb:.1f} KB)")
        return True

    except HTTPError as e:
        print(f"✗ HTTP Error {e.code}: {e.reason}")
        return False
    except URLError as e:
        print(f"✗ URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def verify_checksum(file_path: Path, expected_md5: str | None) -> bool:
    """Verify MD5 checksum of a downloaded file.

    Args:
        file_path: Path to the file to verify.
        expected_md5: Expected MD5 hash (None to skip verification).

    Returns:
        True if checksum matches or verification skipped, False otherwise.
    """
    if expected_md5 is None:
        return True

    md5_hash = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)

    actual_md5 = md5_hash.hexdigest()
    if actual_md5 != expected_md5:
        print(f"  ✗ Checksum mismatch: expected {expected_md5}, got {actual_md5}")
        return False

    print(f"  ✓ Checksum verified: {actual_md5}")
    return True


def download_instance_family(
    family_key: str,
    family_info: dict[str, Any],
    force: bool = False,
) -> tuple[int, int]:
    """Download all instances in a problem family.

    Args:
        family_key: Family identifier (e.g., 'netgen_small').
        family_info: Dictionary with family metadata and file list.
        force: If True, re-download even if file exists.

    Returns:
        Tuple of (successful_downloads, failed_downloads).
    """
    print(f"\n{family_info['name']}:")
    print(f"  {family_info['description']}")

    if not family_info["files"]:
        print("  ⚠ No instances configured for this family yet")
        print("  Note: Update DIMACS_INSTANCES in this script with actual file list")
        return (0, 0)

    local_dir = Path(family_info["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for file_info in family_info["files"]:
        filename = file_info[0]
        dest_path = local_dir / filename
        url = f"{family_info['url_base']}/{filename}"

        # Skip if file exists and not forcing re-download
        if dest_path.exists() and not force:
            print(f"  ⊙ {filename} (already exists, skipping)")
            successful += 1
            continue

        if download_file(url, dest_path, show_progress=False):
            successful += 1
        else:
            failed += 1

    return (successful, failed)


def list_instances() -> None:
    """List all available DIMACS instances."""
    print("Available DIMACS Instance Families:\n")

    for family_key, family_info in DIMACS_INSTANCES.items():
        print(f"  {family_key}:")
        print(f"    Name: {family_info['name']}")
        print(f"    Description: {family_info['description']}")
        print(f"    Size: {family_info['size_category']}")
        print(f"    Files: {len(family_info['files'])} instances")
        print(f"    Local directory: {family_info['local_dir']}")
        print()


def update_problem_catalog(downloaded_families: list[str]) -> None:
    """Update problem_catalog.json with downloaded instances.

    Args:
        downloaded_families: List of family keys that were downloaded.
    """
    catalog_path = Path("benchmarks/metadata/problem_catalog.json")
    if not catalog_path.exists():
        print("\n⚠ Warning: problem_catalog.json not found, skipping catalog update")
        return

    print("\n📝 Updating problem_catalog.json...")
    # Implementation note: This would parse DIMACS files and add metadata
    # For Phase 3 MVP, we'll defer detailed catalog updates to later
    print("  ⚠ Detailed catalog update deferred to Phase 4")


def main() -> int:
    """Main entry point for DIMACS download script."""
    parser = argparse.ArgumentParser(
        description="Download DIMACS benchmark instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available instances",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Download small instances only",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Download medium instances only",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Download large instances only",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available instances without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        list_instances()
        return 0

    # Determine which families to download
    families_to_download = []

    if args.all:
        families_to_download = list(DIMACS_INSTANCES.keys())
    else:
        if args.small:
            families_to_download.extend(
                [k for k, v in DIMACS_INSTANCES.items() if v["size_category"] == "small"]
            )
        if args.medium:
            families_to_download.extend(
                [k for k, v in DIMACS_INSTANCES.items() if v["size_category"] == "medium"]
            )
        if args.large:
            families_to_download.extend(
                [k for k, v in DIMACS_INSTANCES.items() if v["size_category"] == "large"]
            )

    # If no options specified, show help
    if not families_to_download:
        parser.print_help()
        return 1

    print("=" * 70)
    print("DIMACS Benchmark Instance Downloader")
    print("=" * 70)
    print("\nCitation:")
    print("  Johnson, D.S. and McGeoch, C.C. (Eds.).")
    print("  Network Flows and Matching: First DIMACS Implementation Challenge.")
    print("  DIMACS Series in Discrete Mathematics and Theoretical Computer Science,")
    print("  Volume 12, American Mathematical Society, Providence, RI, 1993.")
    print()

    # Download instances
    total_successful = 0
    total_failed = 0

    for family_key in families_to_download:
        family_info = DIMACS_INSTANCES[family_key]
        successful, failed = download_instance_family(family_key, family_info, force=args.force)
        total_successful += successful
        total_failed += failed

    # Summary
    print("\n" + "=" * 70)
    print("Download Summary:")
    print(f"  ✓ Successful: {total_successful}")
    print(f"  ✗ Failed: {total_failed}")
    print("=" * 70)

    # Update catalog
    if total_successful > 0:
        update_problem_catalog(families_to_download)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
