#!/usr/bin/env python3
"""
Script: p4_01d_project_strong_morph.py
Phase: 4 - Compilation
Purpose: Project Strong's and morphology onto NLP-only words conservatively

Input:
    - data/intermediate/tr_complete.parquet
    - data/intermediate/n1904_words.parquet
Output:
    - data/intermediate/tr_complete.parquet

Strategy:
    Fill missing `strong`/`morph` only when the original N1904 TF corpus has a
    unique (strong, morph) pair for the exact (word, lemma, sp) signature.

This intentionally favors precision over recall. Existing aligned values are
kept unchanged and annotated as directly aligned.
"""

import argparse
import sys
from numbers import Integral, Real
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.config import load_config
from scripts.utils.logging import ScriptLogger, get_logger


PROJECTION_SOURCE = "n1904_projected_word_lemma_sp"
ALIGNED_SOURCE = "n1904_aligned"
PROJECTION_CONFIDENCE = 0.92
ALIGNED_CONFIDENCE = 1.0
SIGNATURE_COLUMNS = ("word", "lemma", "sp")


def _norm(value) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and float(value).is_integer():
        return str(int(value))
    return str(value)


def build_unique_lookup(n1904_df):
    """Build a unique (strong, morph) lookup keyed by (word, lemma, sp)."""
    records = {}

    grouped = n1904_df.groupby(list(SIGNATURE_COLUMNS), dropna=False)
    for key, group in grouped:
        normalized_pairs = {
            (_norm(strong), _norm(morph))
            for strong, morph in zip(group["strong"], group["morph"])
            if _norm(strong) and _norm(morph)
        }
        if len(normalized_pairs) != 1:
            continue

        records[key if isinstance(key, tuple) else (key,)] = next(iter(normalized_pairs))

    return records


def annotate_existing_sources(complete_df):
    """Mark directly aligned lexical annotations with provenance/confidence."""
    for feature in ("strong", "morph"):
        source_col = f"{feature}_source"
        confidence_col = f"{feature}_confidence"

        if source_col not in complete_df.columns:
            complete_df[source_col] = None
        if confidence_col not in complete_df.columns:
            complete_df[confidence_col] = None

    aligned_mask = complete_df["source"].eq("n1904")

    for feature in ("strong", "morph"):
        source_col = f"{feature}_source"
        confidence_col = f"{feature}_confidence"
        has_value = complete_df[feature].notna()
        needs_source = complete_df[source_col].isna()
        needs_conf = complete_df[confidence_col].isna()

        complete_df.loc[aligned_mask & has_value & needs_source, source_col] = ALIGNED_SOURCE
        complete_df.loc[aligned_mask & has_value & needs_conf, confidence_col] = ALIGNED_CONFIDENCE

    return complete_df


def project_annotations(complete_df, n1904_df):
    """Project strong/morph onto rows with missing lexical annotations."""
    import pandas as pd

    logger = get_logger(__name__)

    complete_df = annotate_existing_sources(complete_df)
    lookup = build_unique_lookup(n1904_df)
    logger.info(f"Unique N1904 word+lemma+sp signatures: {len(lookup):,}")

    target_mask = complete_df["strong"].isna() | complete_df["morph"].isna()
    target_df = complete_df.loc[target_mask, list(SIGNATURE_COLUMNS)].copy()

    for col in SIGNATURE_COLUMNS:
        target_df[col] = target_df[col].map(_norm)

    projected = 0
    for idx, row in target_df.iterrows():
        key = tuple(row[col] for col in SIGNATURE_COLUMNS)
        pair = lookup.get(key)
        if pair is None:
            continue

        strong, morph = pair

        if pd.isna(complete_df.at[idx, "strong"]) and strong:
            complete_df.at[idx, "strong"] = strong
            complete_df.at[idx, "strong_source"] = PROJECTION_SOURCE
            complete_df.at[idx, "strong_confidence"] = PROJECTION_CONFIDENCE

        if pd.isna(complete_df.at[idx, "morph"]) and morph:
            complete_df.at[idx, "morph"] = morph
            complete_df.at[idx, "morph_source"] = PROJECTION_SOURCE
            complete_df.at[idx, "morph_confidence"] = PROJECTION_CONFIDENCE

        projected += 1

    logger.info(f"Projected strong/morph pairs onto {projected:,} words")
    return complete_df, projected


def main(config: dict = None, dry_run: bool = False) -> bool:
    """Main entry point."""
    if config is None:
        config = load_config()

    logger = get_logger(__name__)
    intermediate_dir = Path(config["paths"]["data"]["intermediate"])
    complete_path = intermediate_dir / "tr_complete.parquet"
    n1904_path = intermediate_dir / "n1904_words.parquet"

    if dry_run:
        logger.info("[DRY RUN] Would project strong/morph via word+lemma+sp")
        return True

    import pandas as pd

    if not complete_path.exists() or not n1904_path.exists():
        logger.error("Required input files are missing")
        return False

    complete_df = pd.read_parquet(complete_path)
    n1904_df = pd.read_parquet(n1904_path)

    before_strong = int(complete_df["strong"].notna().sum()) if "strong" in complete_df.columns else 0
    before_morph = int(complete_df["morph"].notna().sum()) if "morph" in complete_df.columns else 0

    complete_df, projected = project_annotations(complete_df, n1904_df)

    for feature in ("strong", "morph", "strong_source", "morph_source"):
        if feature in complete_df.columns:
            complete_df[feature] = complete_df[feature].map(
                lambda value: _norm(value) or None
            ).astype(object)

    for feature in ("strong_confidence", "morph_confidence"):
        if feature in complete_df.columns:
            complete_df[feature] = complete_df[feature].astype(float)

    complete_df.to_parquet(complete_path, index=False)

    after_strong = int(complete_df["strong"].notna().sum())
    after_morph = int(complete_df["morph"].notna().sum())

    logger.info(f"Saved to: {complete_path}")
    logger.info("")
    logger.info("Projection Summary:")
    logger.info("----------------------------------------")
    logger.info(f"Words projected by word+lemma+sp: {projected:,}")
    logger.info(f"Strong coverage: {before_strong:,} -> {after_strong:,}")
    logger.info(f"Morph coverage:  {before_morph:,} -> {after_morph:,}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with ScriptLogger("p4_01d_project_strong_morph"):
        success = main(dry_run=args.dry_run)
        sys.exit(0 if success else 1)
