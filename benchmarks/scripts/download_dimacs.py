#!/usr/bin/env python3
"""Download DIMACS benchmark instances from the LEMON benchmark suite.

This script provides a framework for downloading minimum cost flow problem
instances from the LEMON project's benchmark suite.

**Phase 4 MVP Status**: Framework is complete, but specific instance URLs need
verification. For now, use manual download from:
https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData

Usage:
    python benchmarks/scripts/download_dimacs.py --list  # Show configured instances
    python benchmarks/scripts/download_dimacs.py --small # Attempt download

LEMON Benchmark Data:
    https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData
    http://lime.cs.elte.hu/~kpeter/data/mcf/

License:
    LEMON library: Boost Software License 1.0 (very permissive)
    Generated instances (NETGEN, GRIDGEN, GOTO): Public Domain
    Citation: Péter Kovács, Optimization Methods and Software, 30:94-127, 2015

Manual Download (Recommended for MVP):
    1. Visit https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData
    2. Download instances from http://lime.cs.elte.hu/~kpeter/data/mcf/
    3. Save to benchmarks/problems/lemon/<family>/
    4. Parse with: from benchmarks.parsers.dimacs import parse_dimacs_file

Example:
    # List configured instances
    python benchmarks/scripts/download_dimacs.py --list

    # Try automated download (may need URL updates)
    python benchmarks/scripts/download_dimacs.py --small
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# LEMON benchmark data base URL (well-maintained, accessible)
# See: https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData
LEMON_BASE_URL = "http://lime.cs.elte.hu/~kpeter/data/mcf"

# Problem families and their URLs
# Based on LEMON benchmark suite which is well-documented and accessible
DIMACS_INSTANCES = {
    "netgen_small": {
        "name": "NETGEN Small Instances",
        "description": "Small NETGEN-generated minimum cost flow problems",
        "size_category": "small",
        "url_base": f"{LEMON_BASE_URL}/netgen",
        "files": [
            # Format: (filename, description, approx nodes, approx arcs)
            # NETGEN-8 family: 8,000 nodes, sparse to dense networks
            ("netgen-8-1.dmx", "NETGEN-8 instance 1 (~8K nodes, ~20K arcs)", 8000, 20000),
            ("netgen-8-2.dmx", "NETGEN-8 instance 2 (~8K nodes, ~20K arcs)", 8000, 20000),
            ("netgen-8-3.dmx", "NETGEN-8 instance 3 (~8K nodes, ~20K arcs)", 8000, 20000),
        ],
        "local_dir": "benchmarks/problems/lemon/netgen",
        "license": "LEMON: Boost 1.0 (library), NETGEN instances: Public Domain",
    },
    "gridgen_small": {
        "name": "GRIDGEN Small Instances",
        "description": "Small GRIDGEN grid-based network problems",
        "size_category": "small",
        "url_base": f"{LEMON_BASE_URL}/gridgen",
        "files": [
            # GRIDGEN-8 family: Grid networks with ~8,000 nodes
            ("gridgen-8-1.dmx", "GRIDGEN-8 instance 1 (grid network)", 8000, 30000),
            ("gridgen-8-2.dmx", "GRIDGEN-8 instance 2 (grid network)", 8000, 30000),
        ],
        "local_dir": "benchmarks/problems/lemon/gridgen",
        "license": "LEMON: Boost 1.0 (library), GRIDGEN instances: Public Domain",
    },
    "goto_small": {
        "name": "GOTO Small Instances",
        "description": "Small grid-on-torus network problems",
        "size_category": "small",
        "url_base": f"{LEMON_BASE_URL}/goto",
        "files": [
            # GOTO-8 family: Grid-on-torus networks with ~8,000 nodes
            ("goto-8-1.dmx", "GOTO-8 instance 1 (grid-on-torus)", 8000, 32000),
            ("goto-8-2.dmx", "GOTO-8 instance 2 (grid-on-torus)", 8000, 32000),
        ],
        "local_dir": "benchmarks/problems/lemon/goto",
        "license": "LEMON: Boost 1.0 (library), GOTO instances: Public Domain",
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
