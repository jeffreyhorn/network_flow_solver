#!/usr/bin/env python3
"""Download DIMACS benchmark instances from the LEMON benchmark suite.

This script downloads minimum cost flow problem instances from the LEMON
project's benchmark suite. Files are automatically downloaded, decompressed
from gzip format, and saved as DIMACS .min files ready for parsing.

Usage:
    python benchmarks/scripts/download_dimacs.py --list           # Show available instances
    python benchmarks/scripts/download_dimacs.py --small          # Download small instances (11 files, ~700 KB)
    python benchmarks/scripts/download_dimacs.py --all            # Download all configured instances
    python benchmarks/scripts/download_dimacs.py --small --max-size 20  # Download only files ≤20KB (compressed)

LEMON Benchmark Data:
    https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData
    http://lime.cs.elte.hu/~kpeter/data/mcf/

License:
    LEMON library: Boost Software License 1.0 (very permissive)
    Generated instances (NETGEN, GRIDGEN, GOTO): Public Domain
    Citation: Péter Kovács, Optimization Methods and Software, 30:94-127, 2015

Features:
    - Automatic gzip decompression (.min.gz → .min)
    - Progress reporting with file sizes
    - Skip already downloaded files (use --force to re-download)
    - Downloaded files ready to parse with DIMACS parser

Example:
    # Download small benchmark instances
    python benchmarks/scripts/download_dimacs.py --small

    # Parse a downloaded instance
    from benchmarks.parsers.dimacs import parse_dimacs_file
    problem = parse_dimacs_file('benchmarks/problems/lemon/netgen/netgen_8_08a.min')
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# LEMON benchmark data base URL (well-maintained, accessible)
# See: https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData
LEMON_BASE_URL = "http://lime.cs.elte.hu/~kpeter/data/mcf"

# Problem families and their URLs
# Based on LEMON benchmark suite which is well-documented and accessible
DIMACS_INSTANCES = {
    "netgen_small": {
        "name": "NETGEN Small Instances",
        "description": "Small NETGEN-8 minimum cost flow problems (gzip compressed)",
        "size_category": "small",
        "url_base": f"{LEMON_BASE_URL}/netgen",
        "files": [
            # Format: (filename, description)
            # NETGEN-8 family: 256-512 nodes, varying density (08=sparse to 13=denser)
            # Files are gzip compressed (.min.gz), will be decompressed to .min
            ("netgen_8_08a.min.gz", "NETGEN-8 sparse network (256 nodes, variant a)"),
            ("netgen_8_08b.min.gz", "NETGEN-8 sparse network (256 nodes, variant b)"),
            ("netgen_8_09a.min.gz", "NETGEN-8 network (512 nodes, variant a)"),
            ("netgen_8_10a.min.gz", "NETGEN-8 network (512 nodes, variant a)"),
            ("netgen_8_11a.min.gz", "NETGEN-8 network (512 nodes, variant a)"),
        ],
        "local_dir": "benchmarks/problems/lemon/netgen",
        "license": "LEMON: Boost 1.0 (library), NETGEN instances: Public Domain",
    },
    "gridgen_small": {
        "name": "GRIDGEN Small Instances",
        "description": "Small GRIDGEN grid-based network problems (gzip compressed)",
        "size_category": "small",
        "url_base": f"{LEMON_BASE_URL}/gridgen",
        "files": [
            # GRIDGEN-8 family: Grid networks with 256-512 nodes
            ("gridgen_8_08a.min.gz", "GRIDGEN-8 grid network (256 nodes, variant a)"),
            ("gridgen_8_08b.min.gz", "GRIDGEN-8 grid network (256 nodes, variant b)"),
            ("gridgen_8_09a.min.gz", "GRIDGEN-8 grid network (512 nodes, variant a)"),
        ],
        "local_dir": "benchmarks/problems/lemon/gridgen",
        "license": "LEMON: Boost 1.0 (library), GRIDGEN instances: Public Domain",
    },
    "goto_small": {
        "name": "GOTO Small Instances",
        "description": "Small grid-on-torus network problems (gzip compressed)",
        "size_category": "small",
        "url_base": f"{LEMON_BASE_URL}/goto",
        "files": [
            # GOTO-8 family: Grid-on-torus networks with 256-512 nodes
            ("goto_8_08a.min.gz", "GOTO-8 grid-on-torus (256 nodes, variant a)"),
            ("goto_8_08b.min.gz", "GOTO-8 grid-on-torus (256 nodes, variant b)"),
            ("goto_8_09a.min.gz", "GOTO-8 grid-on-torus (512 nodes, variant a)"),
        ],
        "local_dir": "benchmarks/problems/lemon/goto",
        "license": "LEMON: Boost 1.0 (library), GOTO instances: Public Domain",
    },
}


def get_file_size(url: str) -> int | None:
    """Get the size of a file at a URL without downloading it.

    Args:
        url: URL to check.

    Returns:
        File size in bytes, or None if unable to determine.
    """
    try:
        request = Request(url, method="HEAD")
        with urlopen(request, timeout=10) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                return int(content_length)
    except Exception:
        pass
    return None


def download_file(url: str, dest_path: Path, show_progress: bool = True) -> bool:
    """Download a file from a URL to a local path.

    If the URL ends with .gz, the file will be automatically decompressed
    and saved without the .gz extension.

    Args:
        url: URL to download from.
        dest_path: Local path to save the file (without .gz for compressed files).
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

        # If the file is gzip compressed, decompress it
        if url.endswith(".gz"):
            try:
                decompressed_content = gzip.decompress(content)
                # Save decompressed file (dest_path should not have .gz extension)
                dest_path.write_bytes(decompressed_content)
                file_size_kb = len(decompressed_content) / 1024
                compressed_size_kb = len(content) / 1024
                print(
                    f"✓ ({compressed_size_kb:.1f} KB compressed → {file_size_kb:.1f} KB decompressed)"
                )
            except Exception as e:
                print(f"✗ Decompression error: {e}")
                return False
        else:
            # Save file as-is
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
    max_size_kb: float | None = None,
) -> tuple[int, int, int]:
    """Download all instances in a problem family.

    Args:
        family_key: Family identifier (e.g., 'netgen_small').
        family_info: Dictionary with family metadata and file list.
        force: If True, re-download even if file exists.
        max_size_kb: Maximum compressed file size in KB (None for no limit).

    Returns:
        Tuple of (successful_downloads, failed_downloads, skipped_due_to_size).
    """
    print(f"\n{family_info['name']}:")
    print(f"  {family_info['description']}")

    if not family_info["files"]:
        print("  ⚠ No instances configured for this family yet")
        print("  Note: Update DIMACS_INSTANCES in this script with actual file list")
        return (0, 0, 0)

    local_dir = Path(family_info["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    skipped_size = 0

    for file_info in family_info["files"]:
        filename = file_info[0]
        # If filename ends with .gz, strip it for the destination path
        # (download_file will decompress automatically)
        dest_filename = filename.replace(".gz", "") if filename.endswith(".gz") else filename
        dest_path = local_dir / dest_filename
        url = f"{family_info['url_base']}/{filename}"

        # Skip if file exists and not forcing re-download
        if dest_path.exists() and not force:
            print(f"  ⊙ {dest_filename} (already exists, skipping)")
            successful += 1
            continue

        # Check file size if max_size_kb is specified
        if max_size_kb is not None:
            file_size_bytes = get_file_size(url)
            if file_size_bytes is not None:
                file_size_kb = file_size_bytes / 1024
                if file_size_kb > max_size_kb:
                    print(
                        f"  ⊘ {dest_filename} (skipped: {file_size_kb:.1f} KB > {max_size_kb:.1f} KB limit)"
                    )
                    skipped_size += 1
                    continue

        if download_file(url, dest_path, show_progress=False):
            successful += 1
        else:
            failed += 1

    return (successful, failed, skipped_size)


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
    parser.add_argument(
        "--max-size",
        type=float,
        metavar="KB",
        help="Only download files smaller than KB (compressed size, e.g., --max-size 50 for 50KB limit)",
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
    print("\nSource: LEMON Benchmark Suite")
    print("  https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData")
    print()
    print("Citation:")
    print("  Péter Kovács. Minimum-cost flow algorithms: an experimental")
    print("  evaluation. Optimization Methods and Software, 30:94-127, 2015.")
    print()
    print("License:")
    print("  LEMON library: Boost Software License 1.0 (very permissive)")
    print("  Generated instances (NETGEN, GRIDGEN, GOTO): Public Domain")
    print()

    if args.max_size:
        print(f"File size limit: {args.max_size:.1f} KB (compressed)")
        print()

    # Download instances
    total_successful = 0
    total_failed = 0
    total_skipped_size = 0

    for family_key in families_to_download:
        family_info = DIMACS_INSTANCES[family_key]
        successful, failed, skipped_size = download_instance_family(
            family_key, family_info, force=args.force, max_size_kb=args.max_size
        )
        total_successful += successful
        total_failed += failed
        total_skipped_size += skipped_size

    # Summary
    print("\n" + "=" * 70)
    print("Download Summary:")
    print(f"  ✓ Successful: {total_successful}")
    print(f"  ✗ Failed: {total_failed}")
    if total_skipped_size > 0:
        print(f"  ⊘ Skipped (size limit): {total_skipped_size}")
    print("=" * 70)

    # Update catalog
    if total_successful > 0:
        update_problem_catalog(families_to_download)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
