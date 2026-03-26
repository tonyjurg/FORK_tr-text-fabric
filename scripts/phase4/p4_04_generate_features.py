#!/usr/bin/env python3
"""
Script: p4_04_generate_features.py
Phase: 4 - Compilation
Purpose: Build the canonical Text-Fabric dataset with Text-Fabric's converter

Input:
    - data/intermediate/tr_complete.parquet
    - data/intermediate/tr_containers.parquet
    - data/intermediate/tr_structure_nodes.parquet
Output:
    - tf/<version>/ directory with canonical .tf files

Usage:
    python -m scripts.phase4.p4_04_generate_features
    python -m scripts.phase4.p4_04_generate_features --dry-run
"""

import argparse
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.config import load_config
from scripts.utils.logging import ScriptLogger, get_logger
from scripts.utils.canonical import (
    full_book_name,
    get_tf_dataset_dir,
    sort_canonically,
    validate_word_id_canonical_order,
)


WORD_FEATURE_MAP = (
    ("unicode", "word", "surface word form"),
    ("lemma", "lemma", "dictionary lemma"),
    ("strong", "strong", "Strong number"),
    ("morph", "morph", "morphology code"),
    ("strong_source", "strong_source", "provenance of the Strong number"),
    ("morph_source", "morph_source", "provenance of the morphology code"),
    ("strong_confidence", "strong_confidence", "confidence score for projected Strong number"),
    ("morph_confidence", "morph_confidence", "confidence score for projected morphology code"),
    ("sp", "sp", "part of speech"),
    ("function", "function", "syntactic function"),
    ("role", "role", "syntactic role"),
    ("case", "case", "grammatical case"),
    ("gender", "gn", "grammatical gender"),
    ("number", "nu", "grammatical number"),
    ("person", "ps", "grammatical person"),
    ("tense", "tense", "grammatical tense"),
    ("voice", "voice", "grammatical voice"),
    ("mood", "mood", "grammatical mood"),
    ("gloss", "gloss", "English gloss"),
    ("source", "source", "word annotation source"),
    ("translit", "translit", "word transliteration"),
    ("lemmatranslit", "lemmatranslit", "lemma transliteration"),
    ("unaccent", "unaccent", "word without accents"),
    ("after", "after", "trailing punctuation or spacing"),
    ("ln", "ln", "Louw-Nida domain"),
    ("bookshort", "bookshort", "book abbreviation"),
    ("text", "text", "surface text alias"),
    ("normalized", "normalized", "normalized Unicode form"),
    ("num", "num", "word position in verse"),
    ("ref", "ref", "reference string"),
    ("id", "id", "word identifier"),
    ("cls", "cls", "word class"),
    ("trans", "trans", "contextual translation"),
    ("domain", "domain", "semantic domain"),
    ("typems", "typems", "morphological subtype"),
)

WORD_INT_FEATURES = {"num", "person"}

STRUCTURE_FEATURE_MAP = (
    ("typ", "typ", "syntactic type"),
    ("function", "function", "syntactic function"),
    ("rela", "rela", "relation to context"),
    ("clausetype", "clausetype", "clause type"),
    ("rule", "rule", "word-group rule"),
    ("structure_source", "source", "structure provenance"),
    ("structure_confidence", "confidence", "structure confidence score"),
)

STRING_FEATURES = {"book", "unicode", "after", "id", "ref", "text", "source", "structure_source"}
INT_FEATURES = {"chapter", "verse", "num", "person"}


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _normalize_feature_value(feature_name: str, value):
    if _is_missing(value):
        return None
    if isinstance(value, str):
        return value
    if feature_name in INT_FEATURES:
        return int(value)
    if feature_name == "structure_confidence":
        # Text-Fabric stores scalar node features as ints or strings.
        return f"{float(value):.2f}"
    return str(value)


def _read_metadata_dict(config: dict, key: str) -> dict:
    """Read a metadata dictionary from tf_output config."""
    value = config.get("tf_output", {}).get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"tf_output.{key} must be a dictionary")
    return value


def _normalize_metadata_items(metadata: dict, context: str) -> dict:
    """Normalize metadata keys and values to strings, skipping null values."""
    normalized = {}
    for key, value in metadata.items():
        if value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized


def _sort_non_slots(df):
    if df.empty:
        return df
    return df.sort_values(["first_slot", "last_slot", "node_id"], kind="mergesort").reset_index(drop=True)


def load_build_inputs(config: dict):
    import pandas as pd

    intermediate_dir = Path(config["paths"]["data"]["intermediate"])
    complete_path = intermediate_dir / "tr_complete.parquet"
    containers_path = intermediate_dir / "tr_containers.parquet"
    structure_path = intermediate_dir / "tr_structure_nodes.parquet"

    for path in (complete_path, containers_path, structure_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    complete_df = pd.read_parquet(complete_path)
    containers_df = pd.read_parquet(containers_path)
    structure_df = pd.read_parquet(structure_path)

    complete_df = sort_canonically(complete_df)
    validate_word_id_canonical_order(complete_df)
    containers_df = _sort_non_slots(containers_df)
    structure_df = _sort_non_slots(structure_df)

    return complete_df, containers_df, structure_df


def build_feature_metadata(complete_df, structure_df, config: dict) -> tuple[dict, set[str]]:
    feature_meta = {
        "book": {"description": "book name (full)"},
        "chapter": {"description": "chapter number"},
        "verse": {"description": "verse number"},
        "parent": {"description": "parent (head) word in dependency tree"},
    }
    int_features = {"chapter", "verse"}

    for output_name, input_name, description in WORD_FEATURE_MAP:
        if input_name not in complete_df.columns:
            continue
        if complete_df[input_name].notna().sum() == 0:
            continue
        feature_meta[output_name] = {"description": description}
        if output_name in WORD_INT_FEATURES:
            int_features.add(output_name)

    for output_name, input_name, description in STRUCTURE_FEATURE_MAP:
        if input_name not in structure_df.columns:
            continue
        if structure_df[input_name].notna().sum() == 0:
            continue
        feature_meta[output_name] = {"description": description}

    configured_feature_meta = _read_metadata_dict(config, "feature_metadata")
    for feature_name, overrides in configured_feature_meta.items():
        if not isinstance(overrides, dict):
            raise ValueError(
                f"tf_output.feature_metadata.{feature_name} must be a dictionary"
            )
        feature_name = str(feature_name)
        if feature_name in feature_meta:
            feature_meta[feature_name].update(
                _normalize_metadata_items(overrides, f"feature_metadata.{feature_name}")
            )

    return feature_meta, int_features


def prepare_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    trailer_path = output_dir / "trailer.tf"
    if trailer_path.exists():
        trailer_path.unlink()
    cache_dir = output_dir / ".tf"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def build_dataset(complete_df, containers_df, structure_df, output_dir: Path, config: dict) -> bool:
    from tf.fabric import Fabric
    from tf.convert.walker import CV

    logger = get_logger(__name__)

    feature_meta, int_features = build_feature_metadata(complete_df, structure_df, config)
    generic_metadata = {
        "name": config["tf_output"]["dataset_name"],
        "version": str(config["tf_output"]["version"]),
        "language": config["tf_output"]["language"],
        "description": config["project"]["description"],
        "source": "TR via graft-and-patch from N1904",
    }
    generic_metadata.update(
        _normalize_metadata_items(
            _read_metadata_dict(config, "global_feature_metadata"),
            "global_feature_metadata",
        )
    )
    otext = {
        "fmt:text-orig-full": "{unicode}{after}",
        "sectionTypes": "book,chapter,verse",
        "sectionFeatures": "book,chapter,verse",
    }

    slot_rows = list(complete_df.to_dict("records"))
    structure_rows = {
        otype: list(_sort_non_slots(structure_df[structure_df["otype"] == otype]).to_dict("records"))
        for otype in ("clause", "phrase", "wg")
    }

    def director(cv):
        slot_handles_by_word_id = {}
        current_book = None
        current_chapter = None
        current_verse = None
        current_book_key = None
        current_chapter_key = None
        current_verse_key = None

        for slot_number, row in enumerate(slot_rows, start=1):
            book_key = row["book"]
            chapter_key = int(row["chapter"])
            verse_key = int(row["verse"])

            if current_book_key != book_key:
                if current_verse is not None:
                    cv.terminate(current_verse)
                    current_verse = None
                if current_chapter is not None:
                    cv.terminate(current_chapter)
                    current_chapter = None
                if current_book is not None:
                    cv.terminate(current_book)
                current_book = cv.node("book")
                cv.feature(current_book, book=full_book_name(book_key))
                current_book_key = book_key
                current_chapter_key = None
                current_verse_key = None

            if current_chapter_key != chapter_key:
                if current_verse is not None:
                    cv.terminate(current_verse)
                    current_verse = None
                if current_chapter is not None:
                    cv.terminate(current_chapter)
                current_chapter = cv.node("chapter")
                cv.feature(
                    current_chapter,
                    book=full_book_name(book_key),
                    chapter=chapter_key,
                )
                current_chapter_key = chapter_key
                current_verse_key = None

            if current_verse_key != verse_key:
                if current_verse is not None:
                    cv.terminate(current_verse)
                current_verse = cv.node("verse")
                cv.feature(
                    current_verse,
                    book=full_book_name(book_key),
                    chapter=chapter_key,
                    verse=verse_key,
                )
                current_verse_key = verse_key

            handle = cv.slot()
            slot_handles_by_word_id[row["word_id"]] = handle

            cv.feature(
                handle,
                book=full_book_name(book_key),
                chapter=chapter_key,
                verse=verse_key,
            )

            for output_name, input_name, _description in WORD_FEATURE_MAP:
                if input_name not in row:
                    continue
                value = _normalize_feature_value(output_name, row.get(input_name))
                if value is not None:
                    cv.feature(handle, **{output_name: value})

        for row in slot_rows:
            parent_id = row.get("parent")
            if _is_missing(parent_id):
                continue
            try:
                parent_word_id = int(float(parent_id))
            except (TypeError, ValueError):
                continue

            child_handle = slot_handles_by_word_id.get(row["word_id"])
            parent_handle = slot_handles_by_word_id.get(parent_word_id)
            if child_handle is not None and parent_handle is not None:
                cv.edge(child_handle, parent_handle, parent=None)

        if current_verse is not None:
            cv.terminate(current_verse)
        if current_chapter is not None:
            cv.terminate(current_chapter)
        if current_book is not None:
            cv.terminate(current_book)

        def make_explicit_node(row, otype: str):
            slots = list(range(int(row["first_slot"]), int(row["last_slot"]) + 1))
            handle = cv.node(otype, slots=slots)

            if "book" in row and not _is_missing(row.get("book")):
                cv.feature(handle, book=full_book_name(row["book"]))
            if "chapter" in row and not _is_missing(row.get("chapter")):
                cv.feature(handle, chapter=int(row["chapter"]))
            if "verse" in row and not _is_missing(row.get("verse")):
                cv.feature(handle, verse=int(row["verse"]))

            if otype in {"clause", "phrase", "wg"}:
                for output_name, input_name, _description in STRUCTURE_FEATURE_MAP:
                    value = _normalize_feature_value(output_name, row.get(input_name))
                    if value is not None:
                        cv.feature(handle, **{output_name: value})

            return handle

        for otype in ("clause", "phrase", "wg"):
            for row in structure_rows[otype]:
                make_explicit_node(row, otype)

    logger.info("Building TF dataset with tf.convert.walker.CV.walk()")
    TF = Fabric(locations=str(output_dir), silent="deep")
    cv = CV(TF, silent="deep")
    return cv.walk(
        director,
        slotType="w",
        otext=otext,
        generic=generic_metadata,
        intFeatures=int_features,
        featureMeta=feature_meta,
        warn=True,
        force=False,
    )


def main(config: dict = None, dry_run: bool = False) -> bool:
    """Main entry point."""
    if config is None:
        config = load_config()

    logger = get_logger(__name__)
    output_dir = get_tf_dataset_dir(config)

    if dry_run:
        logger.info("[DRY RUN] Would build canonical TF dataset")
        logger.info(f"[DRY RUN] Output: {output_dir}")
        return True

    try:
        complete_df, containers_df, structure_df = load_build_inputs(config)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return False

    logger.info("Loaded canonical build inputs")
    logger.info(f"  Words: {len(complete_df):,}")
    logger.info(f"  Section containers: {len(containers_df):,}")
    logger.info(f"  Structure nodes: {len(structure_df):,}")

    prepare_output_dir(output_dir)
    success = build_dataset(complete_df, containers_df, structure_df, output_dir, config)

    if success:
        logger.info(f"\nTF dataset written to: {output_dir}")
    else:
        logger.error("TF dataset build failed")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    with ScriptLogger("p4_04_generate_features") as logger:
        config = load_config()
        success = main(config, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
