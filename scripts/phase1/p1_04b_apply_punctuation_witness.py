#!/usr/bin/env python3
"""
Script: p1_04b_apply_punctuation_witness.py
Phase: 1 - Reconnaissance
Purpose: Overlay accents/punctuation from a local witness onto the public-domain base source

Input:
    - data/source/tr_source.csv
    - Local punctuation witness specified in config.yaml

Output:
    - data/source/tr_source_prepared.csv

The witness must be a local text source derived from an accented/punctuated
edition, for example a manually extracted PDF transcription. The base token
sequence remains authoritative; this step only transfers reconstructed
accented words and trailing punctuation by verse-local alignment.

Supported witness formats:
    1. CSV with columns: book, chapter, verse, text
    2. Directory of per-book text files with lines like: 1:1 ...
    3. PDF witness using the Iglesia Reformada Stephens 1550 layout

Usage:
    python -m scripts.phase1.p1_04b_apply_punctuation_witness
    python -m scripts.phase1.p1_04b_apply_punctuation_witness --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.config import load_config
from scripts.utils.logging import ScriptLogger, get_logger
from scripts.utils.stephens_pdf import (
    align_witness_chapter,
    align_witness_to_greek_base,
    align_witness_verse,
    extract_pdf_chapter_texts,
    extract_pdf_verses,
    parse_stv_chapters,
    parse_stv_verses,
)


GREEK_TOKEN_RE = re.compile(r"([\u0370-\u03FF\u1F00-\u1FFF]+)([^\u0370-\u03FF\u1F00-\u1FFF]*)")
VERSE_LINE_RE = re.compile(r"^(\d+):(\d+)\s+(.*)$")

BOOK_CODE_MAP = {
    "MT": "MAT",
    "MR": "MAR",
    "LU": "LUK",
    "JOH": "JHN",
    "AC": "ACT",
    "RO": "ROM",
    "1CO": "1CO",
    "2CO": "2CO",
    "GA": "GAL",
    "EPH": "EPH",
    "PHP": "PHP",
    "COL": "COL",
    "1TH": "1TH",
    "2TH": "2TH",
    "1TI": "1TI",
    "2TI": "2TI",
    "TIT": "TIT",
    "PHM": "PHM",
    "HEB": "HEB",
    "JAS": "JAS",
    "1JO": "1JN",
    "2JO": "2JN",
    "3JO": "3JN",
    "1PE": "1PE",
    "2PE": "2PE",
    "JUDE": "JUD",
    "RE": "REV",
}


def normalize_greek(text: str) -> str:
    """Normalize Greek text for accent-insensitive comparison."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = unicodedata.normalize("NFC", text)
    return text.lower()


def tokenize_witness_verse(text: str) -> list[dict[str, str]]:
    """Tokenize witness verse text into word + after pairs."""
    tokens = []
    for match in GREEK_TOKEN_RE.finditer(text):
        word = match.group(1)
        after = match.group(2) or " "
        after = after.replace("\xa0", " ")
        after = re.sub(r"\s+", " ", after)
        tokens.append(
            {
                "word": word,
                "after": after if after else " ",
                "norm": normalize_greek(word),
            }
        )
    return tokens


def load_witness_from_csv(path: Path) -> pd.DataFrame:
    """Load a witness CSV with verse text."""
    df = pd.read_csv(path)
    required = {"book", "chapter", "verse", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Witness CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["book"] = df["book"].astype(str).map(lambda b: BOOK_CODE_MAP.get(b, b))
    df["chapter"] = df["chapter"].astype(int)
    df["verse"] = df["verse"].astype(int)
    df["text"] = df["text"].fillna("").astype(str)
    return df[list(["book", "chapter", "verse", "text"])]


def load_witness_from_directory(path: Path) -> pd.DataFrame:
    """Load a witness directory containing per-book verse text files."""
    rows = []

    for item in sorted(path.iterdir()):
        if item.suffix.lower() not in {".txt", ".text"}:
            continue

        raw_code = item.stem.upper()
        book = BOOK_CODE_MAP.get(raw_code, raw_code)

        for raw_line in item.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("["):
                continue

            match = VERSE_LINE_RE.match(line)
            if not match:
                continue

            chapter = int(match.group(1))
            verse = int(match.group(2))
            text = match.group(3).strip()
            rows.append(
                {
                    "book": book,
                    "chapter": chapter,
                    "verse": verse,
                    "text": text,
                }
            )

    return pd.DataFrame(rows, columns=["book", "chapter", "verse", "text"])


def load_witness(config: dict) -> pd.DataFrame | None:
    """Load the configured punctuation witness, or return None if absent."""
    witness_cfg = config.get("sources", {}).get("tr", {}).get("punctuation_witness", {})
    path_value = witness_cfg.get("path")
    fmt = str(witness_cfg.get("format", "csv")).lower()

    if not path_value:
        return None

    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Punctuation witness not found: {path}")

    if fmt == "csv":
        return load_witness_from_csv(path)
    if fmt == "directory":
        return load_witness_from_directory(path)
    if fmt == "pdf":
        return pd.DataFrame(
            [
                {"book": book, "chapter": chapter, "verse": verse, "text": text}
                for (book, chapter, verse), text in extract_pdf_verses(path).items()
            ],
            columns=["book", "chapter", "verse", "text"],
        )

    raise ValueError("sources.tr.punctuation_witness.format must be 'csv', 'directory', or 'pdf'")


def _apply_pdf_aligned_tokens(
    result_df: pd.DataFrame,
    row_indices: list[int],
    aligned_tokens: list,
) -> None:
    """Write aligned witness tokens back onto the target rows."""
    for row_idx, aligned_token in zip(row_indices, aligned_tokens):
        result_df.at[row_idx, "word"] = aligned_token.greek_word
        result_df.at[row_idx, "after"] = aligned_token.after
        result_df.at[row_idx, "witness_word"] = aligned_token.greek_word


def _build_pdf_candidate_texts(
    witness_lookup: dict[tuple[str, int, int], str],
    verse_key: tuple[str, int, int],
) -> list[tuple[str, str]]:
    """Build a small set of nearby witness candidates for stubborn verse boundaries."""
    book, chapter, verse = verse_key
    prev_key = (book, chapter, verse - 1)
    next_key = (book, chapter, verse + 1)

    raw_candidates = [
        ("self", witness_lookup.get(verse_key, "")),
        ("next", witness_lookup.get(next_key, "")),
        ("prev", witness_lookup.get(prev_key, "")),
        ("self+next", " ".join(part for part in [witness_lookup.get(verse_key, ""), witness_lookup.get(next_key, "")] if part).strip()),
        ("prev+self", " ".join(part for part in [witness_lookup.get(prev_key, ""), witness_lookup.get(verse_key, "")] if part).strip()),
    ]

    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for label, text in raw_candidates:
        if not text or text in seen:
            continue
        seen.add(text)
        candidates.append((label, text))
    return candidates


def _find_pdf_alignment(
    verse_key: tuple[str, int, int],
    verse_df: pd.DataFrame,
    witness_lookup: dict[tuple[str, int, int], str],
    base_stv_lookup: dict[tuple[str, int, int], list[str]] | None,
) -> tuple[list, int, bool, str] | None:
    """Find the best conservative PDF alignment candidate for one verse."""
    row_count = len(verse_df)
    base_tokens = base_stv_lookup.get(verse_key, []) if base_stv_lookup else []
    greek_base_tokens = verse_df["base_word"].fillna(verse_df["word"]).astype(str).tolist()
    partial_match: tuple[list, int, bool, str] | None = None

    for label, witness_text in _build_pdf_candidate_texts(witness_lookup, verse_key):
        if base_tokens:
            aligned_tokens = align_witness_verse(base_tokens, witness_text)
            if aligned_tokens is not None:
                if len(aligned_tokens) == row_count:
                    return aligned_tokens, row_count, True, label
                if label == "self" and 0 < len(aligned_tokens) < row_count and partial_match is None:
                    partial_match = (aligned_tokens, len(aligned_tokens), False, label)

        aligned_tokens = align_witness_to_greek_base(greek_base_tokens, witness_text)
        if aligned_tokens is not None and len(aligned_tokens) == row_count:
            return aligned_tokens, row_count, True, f"{label}:greek"

    return partial_match


def overlay_punctuation(
    base_df: pd.DataFrame,
    witness_df: pd.DataFrame,
    logger,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """Overlay witness accents/punctuation onto the base token sequence."""
    result_df = base_df.copy()
    if "after" in result_df.columns:
        result_df["after"] = result_df["after"].fillna(" ")
    else:
        result_df["after"] = " "
    if "witness_word" in result_df.columns:
        result_df["witness_word"] = result_df["witness_word"].fillna("")
    else:
        result_df["witness_word"] = ""
    if "base_word" not in result_df.columns:
        result_df["base_word"] = result_df["word"]

    witness_lookup = {
        (row.book, row.chapter, row.verse): row.text
        for row in witness_df.itertuples(index=False)
    }
    witness_fmt = str(config.get("sources", {}).get("tr", {}).get("punctuation_witness", {}).get("format", "csv")).lower()
    base_stv_lookup = None
    if witness_fmt == "pdf":
        repo_dir = Path(config["sources"]["tr"]["local_checkout"])
        base_stv_lookup = parse_stv_verses(repo_dir)
        base_stv_chapter_lookup = parse_stv_chapters(repo_dir)
        witness_cfg = config.get("sources", {}).get("tr", {}).get("punctuation_witness", {})
        chapter_witness_lookup = extract_pdf_chapter_texts(Path(witness_cfg.get("path")))

    stats = {
        "verses_with_witness": 0,
        "verses_missing_witness": 0,
        "verses_aligned": 0,
        "verses_partially_aligned": 0,
        "verses_failed_alignment": 0,
        "tokens_updated": 0,
        "tokens_left_unchanged": 0,
    }

    if witness_fmt == "pdf":
        chapter_grouped = result_df.groupby(["book", "chapter"], sort=False)
        for chapter_key, chapter_df in chapter_grouped:
            chapter_witness_text = chapter_witness_lookup.get(chapter_key, "")
            chapter_base_tokens = base_stv_chapter_lookup.get(chapter_key, []) if base_stv_chapter_lookup else []
            chapter_aligned = None
            if chapter_witness_text and chapter_base_tokens and len(chapter_base_tokens) == len(chapter_df):
                chapter_aligned = align_witness_chapter(chapter_base_tokens, chapter_witness_text)

            if chapter_aligned is not None and len(chapter_aligned) == len(chapter_df):
                for row_idx, aligned_token in zip(chapter_df.index.tolist(), chapter_aligned):
                    result_df.at[row_idx, "word"] = aligned_token.greek_word
                    result_df.at[row_idx, "after"] = aligned_token.after
                    result_df.at[row_idx, "witness_word"] = aligned_token.greek_word
                    stats["tokens_updated"] += 1
                verse_sizes = chapter_df.groupby(["book", "chapter", "verse"], sort=False).size()
                stats["verses_with_witness"] += len(verse_sizes)
                stats["verses_aligned"] += len(verse_sizes)
                continue

            for verse_key, verse_df in chapter_df.groupby(["book", "chapter", "verse"], sort=False):
                witness_text = witness_lookup.get(verse_key)
                if witness_text is None:
                    stats["verses_missing_witness"] += 1
                    stats["tokens_left_unchanged"] += len(verse_df)
                    continue

                stats["verses_with_witness"] += 1
                if not base_stv_lookup or not base_stv_lookup.get(verse_key):
                    stats["verses_failed_alignment"] += 1
                    stats["tokens_left_unchanged"] += len(verse_df)
                    continue

                alignment = _find_pdf_alignment(verse_key, verse_df, witness_lookup, base_stv_lookup)
                if alignment is None:
                    stats["verses_failed_alignment"] += 1
                    stats["tokens_left_unchanged"] += len(verse_df)
                    continue

                aligned_tokens, matched_rows, is_full_alignment, _source_label = alignment
                _apply_pdf_aligned_tokens(result_df, verse_df.index.tolist()[:matched_rows], aligned_tokens[:matched_rows])
                if is_full_alignment:
                    stats["verses_aligned"] += 1
                else:
                    stats["verses_partially_aligned"] += 1
                stats["tokens_updated"] += matched_rows
                stats["tokens_left_unchanged"] += len(verse_df) - matched_rows

        logger.info(f"Verses with witness text: {stats['verses_with_witness']:,}")
        logger.info(f"Verses missing witness text: {stats['verses_missing_witness']:,}")
        logger.info(f"Verses aligned successfully: {stats['verses_aligned']:,}")
        logger.info(f"Verses partially aligned: {stats['verses_partially_aligned']:,}")
        logger.info(f"Verses failed alignment: {stats['verses_failed_alignment']:,}")
        logger.info(f"Tokens updated from witness: {stats['tokens_updated']:,}")
        logger.info(f"Tokens left unchanged: {stats['tokens_left_unchanged']:,}")
        return result_df, stats

    grouped = result_df.groupby(["book", "chapter", "verse"], sort=False)

    for verse_key, verse_df in grouped:
        witness_text = witness_lookup.get(verse_key)
        if witness_text is None:
            stats["verses_missing_witness"] += 1
            stats["tokens_left_unchanged"] += len(verse_df)
            continue

        stats["verses_with_witness"] += 1
        if witness_fmt == "pdf":
            base_tokens = base_stv_lookup.get(verse_key) if base_stv_lookup else None
            if not base_tokens:
                stats["verses_failed_alignment"] += 1
                stats["tokens_left_unchanged"] += len(verse_df)
                continue

            aligned_tokens = align_witness_verse(base_tokens, witness_text)
            if aligned_tokens is None or len(aligned_tokens) != len(verse_df):
                stats["verses_failed_alignment"] += 1
                stats["tokens_left_unchanged"] += len(verse_df)
                continue

            stats["verses_aligned"] += 1
            for row_idx, aligned_token in zip(verse_df.index.tolist(), aligned_tokens):
                result_df.at[row_idx, "word"] = aligned_token.greek_word
                result_df.at[row_idx, "after"] = aligned_token.after
                result_df.at[row_idx, "witness_word"] = aligned_token.greek_word
                stats["tokens_updated"] += 1
            continue

        witness_tokens = tokenize_witness_verse(witness_text)
        if not witness_tokens:
            stats["verses_failed_alignment"] += 1
            stats["tokens_left_unchanged"] += len(verse_df)
            continue

        base_indices = verse_df.index.tolist()
        base_tokens = verse_df["word"].fillna("").astype(str).tolist()
        base_norm = [normalize_greek(token) for token in base_tokens]
        witness_norm = [token["norm"] for token in witness_tokens]
        matcher = SequenceMatcher(None, base_norm, witness_norm)
        aligned_base_positions = set()

        for block in matcher.get_matching_blocks():
            base_start, witness_start, size = block
            for offset in range(size):
                base_pos = base_start + offset
                witness_pos = witness_start + offset
                if base_pos >= len(base_indices) or witness_pos >= len(witness_tokens):
                    continue

                row_idx = base_indices[base_pos]
                witness_token = witness_tokens[witness_pos]
                result_df.at[row_idx, "after"] = witness_token["after"] or " "
                result_df.at[row_idx, "witness_word"] = witness_token["word"]
                aligned_base_positions.add(base_pos)
                stats["tokens_updated"] += 1

        if len(aligned_base_positions) == len(base_indices):
            stats["verses_aligned"] += 1
        else:
            stats["verses_failed_alignment"] += 1
        stats["tokens_left_unchanged"] += len(base_indices) - len(aligned_base_positions)

    logger.info(f"Verses with witness text: {stats['verses_with_witness']:,}")
    logger.info(f"Verses missing witness text: {stats['verses_missing_witness']:,}")
    logger.info(f"Verses aligned successfully: {stats['verses_aligned']:,}")
    logger.info(f"Verses failed alignment: {stats['verses_failed_alignment']:,}")
    logger.info(f"Tokens updated from witness: {stats['tokens_updated']:,}")
    logger.info(f"Tokens left unchanged: {stats['tokens_left_unchanged']:,}")

    return result_df, stats


def main(config: dict = None, dry_run: bool = False) -> bool:
    """Main entry point."""
    if config is None:
        config = load_config()

    logger = get_logger(__name__)

    source_dir = Path(config["paths"]["data"]["source"])
    base_path = source_dir / "tr_source.csv"
    output_path = source_dir / "tr_source_prepared.csv"

    if dry_run:
        logger.info("[DRY RUN] Would overlay punctuation witness onto tr_source.csv")
        logger.info(f"[DRY RUN] Input base: {base_path}")
        logger.info(f"[DRY RUN] Output: {output_path}")
        return True

    if not base_path.exists():
        logger.error(f"Base source not found: {base_path}")
        logger.error("Run p1_04_acquire_tr first")
        return False

    witness_df = load_witness(config)
    if witness_df is None:
        logger.info("No punctuation witness configured; skipping overlay step")
        return True

    base_df = pd.read_csv(base_path)
    overlaid_df, _stats = overlay_punctuation(base_df, witness_df, logger, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlaid_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved prepared source to: {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    with ScriptLogger("p1_04b_apply_punctuation_witness") as logger:
        config = load_config()
        success = main(config, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
