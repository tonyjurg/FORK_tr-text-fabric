#!/usr/bin/env python3
"""
Script: p1_04_acquire_tr.py
Phase: 1 - Reconnaissance
Purpose: Acquire the public-domain Stephens 1550 TR source text

Input:  None (clones or refreshes the public-domain upstream repo)
Output: data/source/tr_source.csv

The source text is acquired from the public-domain repository configured at
`sources.tr.repo_url` and converted into a normalized word-level CSV.

Usage:
    python -m scripts.phase1.p1_04_acquire_tr
    python -m scripts.phase1.p1_04_acquire_tr --dry-run
    python -m scripts.phase1.p1_04_acquire_tr --fresh
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.config import load_config
from scripts.utils.logging import ScriptLogger, get_logger


def main(config: dict = None, dry_run: bool = False, fresh: bool = False) -> bool:
    """Main entry point.

    Args:
        config: Pipeline configuration dict
        dry_run: If True, don't actually download
        fresh: If True, refresh the local source checkout from upstream
    """
    if config is None:
        config = load_config()

    logger = get_logger(__name__)

    source_dir = Path(config["paths"]["data"]["source"])
    output_path = source_dir / "tr_source.csv"

    # Check if already exists (and not forcing refresh)
    if output_path.exists() and not fresh:
        logger.info(f"TR data already exists: {output_path}")
        logger.info("Use --fresh to refresh from the upstream public-domain repo")
        return True

    if dry_run:
        logger.info("[DRY RUN] Would acquire Stephens 1550 from the configured public-domain repo")
        logger.info(f"[DRY RUN] Would save to: {output_path}")
        return True

    logger.info("Acquiring Stephens 1550 TR from the configured public-domain repository...")
    logger.info(f"Repository: {config['sources']['tr']['repo_url']}")

    from scripts.download_stephens_tr import download_all

    result_path = download_all(fresh=fresh)

    if result_path and Path(result_path).exists():
        logger.info(f"TR data saved to: {result_path}")
        return True
    else:
        logger.error("Failed to download TR data")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without acquiring source data")
    parser.add_argument("--fresh", action="store_true",
                        help="Refresh the local source checkout from upstream")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    with ScriptLogger("p1_04_acquire_tr") as logger:
        config = load_config()
        success = main(config, dry_run=args.dry_run, fresh=args.fresh)
        sys.exit(0 if success else 1)
