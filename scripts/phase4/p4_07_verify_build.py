#!/usr/bin/env python3
"""
Script: p4_07_verify_build
Phase: 4 - Compilation
Purpose: Verify the canonical Text-Fabric build

Checks:
    - dataset loads from tf/<version>
    - slot 1 is Matthew 1:1
    - section navigation works
    - node counts match intermediate data
    - required features exist
    - parent edges stay within slot range
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.config import load_config
from scripts.utils.logging import ScriptLogger, get_logger
from scripts.utils.canonical import BOOK_NAME_MAP, NT_CANONICAL_BOOKS, get_tf_dataset_dir


REQUIRED_FILES = [
    "otext.tf",
    "otype.tf",
    "oslots.tf",
    "unicode.tf",
    "book.tf",
    "chapter.tf",
    "verse.tf",
    "parent.tf",
]


def verify_required_files(tf_dir: Path) -> bool:
    logger = get_logger(__name__)
    ok = True
    for name in REQUIRED_FILES:
        path = tf_dir / name
        if path.exists():
            logger.info(f"  {name}: {path.stat().st_size:,} bytes")
        else:
            logger.error(f"  {name}: missing")
            ok = False
    return ok


def load_expected_counts(config: dict) -> dict:
    import pandas as pd

    intermediate = Path(config["paths"]["data"]["intermediate"])
    complete = pd.read_parquet(intermediate / "tr_complete.parquet")
    containers = pd.read_parquet(intermediate / "tr_containers.parquet")
    structure = pd.read_parquet(intermediate / "tr_structure_nodes.parquet")

    return {
        "w": len(complete),
        "verse": int((containers["otype"] == "verse").sum()),
        "chapter": int((containers["otype"] == "chapter").sum()),
        "book": int((containers["otype"] == "book").sum()),
        "clause": int((structure["otype"] == "clause").sum()),
        "phrase": int((structure["otype"] == "phrase").sum()),
        "wg": int((structure["otype"] == "wg").sum()),
    }


def verify_loaded_dataset(tf_dir: Path, expected_counts: dict) -> bool:
    from tf.fabric import Fabric

    logger = get_logger(__name__)
    TF = Fabric(locations=str(tf_dir), silent="deep")
    api = TF.load(
        "book chapter verse unicode after lemma parent typ function rela clausetype rule "
        "structure_source structure_confidence",
        silent="deep",
    )
    if not api:
        logger.error("Failed to load Text-Fabric dataset")
        return False

    F = api.F
    T = api.T
    E = api.E

    checks_ok = True

    first_section = T.sectionFromNode(1)
    logger.info(f"  Slot 1 section: {first_section}")
    if first_section != ("Matthew", 1, 1):
        logger.error("Slot 1 is not Matthew 1:1")
        checks_ok = False

    book_nodes = list(F.otype.s("book"))
    book_names = [F.book.v(node) for node in book_nodes]
    expected_book_names = [BOOK_NAME_MAP[book] for book in NT_CANONICAL_BOOKS]
    logger.info(f"  First 10 books: {book_names[:10]}")
    if book_names != expected_book_names:
        logger.error("Book node order does not follow canonical NT order")
        checks_ok = False

    navigation_targets = [
        ("Matthew", 1, 1),
        ("Romans", 1, 1),
        ("Revelation", 22, 21),
    ]
    for target in navigation_targets:
        node = T.nodeFromSection(target)
        logger.info(f"  Section {target}: node {node}")
        if node is None:
            logger.error(f"Could not resolve section {target}")
            checks_ok = False

    actual_counts = {otype: len(list(F.otype.s(otype))) for otype in expected_counts}
    logger.info("  Node counts:")
    for otype in expected_counts:
        logger.info(f"    {otype}: {actual_counts[otype]:,}")
        if actual_counts[otype] != expected_counts[otype]:
            logger.error(
                f"Count mismatch for {otype}: expected {expected_counts[otype]:,}, "
                f"got {actual_counts[otype]:,}"
            )
            checks_ok = False

    for feature_name in ("unicode", "book", "chapter", "verse"):
        if not hasattr(F, feature_name):
            logger.error(f"Missing required feature: {feature_name}")
            checks_ok = False

    if hasattr(E, "parent"):
        slot_max = expected_counts["w"]
        bad_edges = 0
        for child in range(1, slot_max + 1):
            for parent in E.parent.t(child):
                if not (1 <= parent <= slot_max):
                    bad_edges += 1
        logger.info(f"  Parent edges out of range: {bad_edges}")
        if bad_edges:
            checks_ok = False
    else:
        logger.error("Missing edge feature: parent")
        checks_ok = False

    verse_samples = [
        ("Matthew", 1, 1),
        ("John", 1, 1),
        ("Acts", 8, 37),
        ("I_John", 5, 7),
        ("Revelation", 22, 21),
    ]
    for section in verse_samples:
        node = T.nodeFromSection(section)
        if node is None:
            logger.error(f"Missing representative verse {section}")
            checks_ok = False

    return checks_ok


def main(config: dict = None, dry_run: bool = False) -> bool:
    if config is None:
        config = load_config()

    logger = get_logger(__name__)
    tf_dir = get_tf_dataset_dir(config)

    if dry_run:
        logger.info("[DRY RUN] Would verify canonical TF dataset build")
        logger.info(f"[DRY RUN] Location: {tf_dir}")
        return True

    if not tf_dir.exists():
        logger.error(f"TF directory not found: {tf_dir}")
        return False

    logger.info(f"Verifying TF dataset at: {tf_dir}")
    files_ok = verify_required_files(tf_dir)
    expected_counts = load_expected_counts(config)
    dataset_ok = verify_loaded_dataset(tf_dir, expected_counts)

    if files_ok and dataset_ok:
        logger.info("VERIFICATION PASSED")
        return True

    logger.error("VERIFICATION FAILED")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    with ScriptLogger("p4_07_verify_build") as logger:
        config = load_config()
        success = main(config, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
