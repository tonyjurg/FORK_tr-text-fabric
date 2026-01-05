#!/usr/bin/env python3
"""
Script: p4_01c_fix_nlp_errors.py
Phase: 4 - Compilation
Purpose: Fix systematic NLP errors in lemmatization and POS tagging

This script corrects known errors in Stanza NLP output for TR-only words.
It runs AFTER p4_01b_fill_glosses.py and BEFORE p4_02_generate_containers.py.

Error sources:
1. Stanza confuses genitive noun endings (-ου) with verb forms
2. Elided prepositions (μετ᾽, ἐφ᾽) get wrong lemmas
3. Some proper names are mislemmatized
4. POS tag inconsistencies (noun vs subs)

Input:  data/intermediate/tr_complete.parquet (from p4_01b)
Output: data/intermediate/tr_complete.parquet (updated with fixes)

Usage:
    python -m scripts.phase4.p4_01c_fix_nlp_errors
    python -m scripts.phase4.p4_01c_fix_nlp_errors --dry-run
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from scripts.utils.config import load_config
from scripts.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# MANUAL CORRECTIONS FOR TR-ONLY WORDS
# Format: (word, wrong_lemma) -> (correct_lemma, correct_sp)
# =============================================================================

TR_ONLY_CORRECTIONS = {
    # Elided prepositions - lemma should be the full form
    ("μετ᾽", "μεί"): ("μετά", "prep"),
    ("ἐφ᾽", "ἐάω"): ("ἐπί", "prep"),
    ("παρ᾽", "πάρειμι"): ("παρά", "prep"),
    ("μεθ᾽", "μεθίημι"): ("μετά", "prep"),
    ("ἀφ᾽", "ἀφίημι"): ("ἀπό", "prep"),
    ("ὑφ᾽", "ὑφίημι"): ("ὑπό", "prep"),
    ("καθ᾽", "καθίημι"): ("κατά", "prep"),
    ("ἀνθ᾽", "ἀντί"): ("ἀντί", "prep"),

    # More elided forms - fix lemma/SP
    ("ἀλλ᾽", "ἀλλ᾽"): ("ἀλλά", "conj"),
    ("δι᾽", "δι᾽"): ("διά", "prep"),
    ("ἐπ᾽", "ἐπ᾽"): ("ἐπί", "prep"),
    ("κατ᾽", "κατά"): ("κατά", "prep"),  # Fix SP from adv
    ("καθ᾽", "κατά"): ("κατά", "prep"),  # Fix SP from ptcl
    ("δ᾽", "δέ"): ("δέ", "conj"),  # Fix SP from ptcl

    # ἰδού - imperative of ὁράω, not οἶδα
    ("Ἰδού", "οἶδα"): ("ὁράω", "verb"),
    ("ἰδού", "οἶδα"): ("ὁράω", "verb"),

    # Names - proper nouns
    ("Δαβίδ", "τέβημα"): ("Δαυίδ", "subs"),
    ("Δαβὶδ", "Δαβίς"): ("Δαυίδ", "subs"),
    ("Ἠλίας", "ἅλίη"): ("Ἠλίας", "subs"),
    ("Ἠλίαν", "ἅλίω"): ("Ἠλίας", "subs"),
    ("Ἠλίου", "ἅλίω"): ("Ἠλίας", "subs"),
    ("Ἠλίᾳ", "ἅλίω"): ("Ἠλίας", "subs"),
    ("Ἰωάννης", "Ἰωάννη"): ("Ἰωάννης", "subs"),
    ("Ἰωάννην", "Ἰωάννή"): ("Ἰωάννης", "subs"),
    ("Ἰωάννου", "Ἰωάννός"): ("Ἰωάννης", "subs"),
    ("Πιλᾶτος", "Πιλάτος"): ("Πιλᾶτος", "subs"),
    ("Ἡρῴδης", "Ἡρῴδη"): ("Ἡρῴδης", "subs"),

    # Relative pronouns confused with other words
    ("ἡς", "εἶμι"): ("ὅς", "pron"),
    ("ἧς", "εἶμι"): ("ὅς", "pron"),

    # μήποτε - conjunction/adverb, not verb
    ("μήποτε", "μίπτω"): ("μήποτε", "conj"),
    ("μήποτε", "μήποτε"): ("μήποτε", "conj"),  # Fix SP if verb

    # Numerals
    ("τεσσαράκοντα", "τεσσαράκός"): ("τεσσαράκοντα", "adjv"),

    # Ῥαββί - Rabbi, not a form of ῥαβδίζω
    ("Ῥαββί", "Ῥάβομαι"): ("ῥαββί", "subs"),
    ("ῥαββί", "Ῥάβομαι"): ("ῥαββί", "subs"),

    # Miscellaneous noun/verb confusions
    ("βύσσου", "βοάω"): ("βύσσος", "subs"),  # fine linen
    ("τετράρχου", "τρέχω"): ("τετράρχης", "subs"),  # tetrarch
    ("παραδείσου", "παραδέίνομαι"): ("παράδεισος", "subs"),  # paradise
    ("φόρτου", "φόρόω"): ("φόρτος", "subs"),  # cargo
}


def build_n1904_reference(n1904_df):
    """Build a reference dictionary from N1904 for lemma/POS lookup."""
    ref = {}
    for _, row in n1904_df.iterrows():
        word = row['word']
        if word not in ref:
            ref[word] = defaultdict(int)
        ref[word][(row['lemma'], row['sp'])] += 1

    # Convert to best match
    best_ref = {}
    for word, counts in ref.items():
        best = max(counts.items(), key=lambda x: x[1])
        best_ref[word] = best[0]  # (lemma, sp)

    return best_ref


def apply_corrections(tr_df, n1904_ref):
    """Apply all corrections to TR data."""
    fixes = {
        'n1904_lemma': 0,
        'n1904_sp': 0,
        'manual_lemma': 0,
        'manual_sp': 0,
    }

    nlp_mask = tr_df['source'] == 'nlp'
    nlp_indices = tr_df[nlp_mask].index

    for idx in nlp_indices:
        word = tr_df.at[idx, 'word']
        current_lemma = tr_df.at[idx, 'lemma']
        current_sp = tr_df.at[idx, 'sp']

        # First check manual corrections (TR-only words)
        manual_key = (word, current_lemma)
        if manual_key in TR_ONLY_CORRECTIONS:
            correct_lemma, correct_sp = TR_ONLY_CORRECTIONS[manual_key]
            if current_lemma != correct_lemma:
                tr_df.at[idx, 'lemma'] = correct_lemma
                fixes['manual_lemma'] += 1
            if current_sp != correct_sp:
                tr_df.at[idx, 'sp'] = correct_sp
                fixes['manual_sp'] += 1
            continue

        # Then check N1904 reference
        if word in n1904_ref:
            correct_lemma, correct_sp = n1904_ref[word]
            if current_lemma != correct_lemma:
                tr_df.at[idx, 'lemma'] = correct_lemma
                fixes['n1904_lemma'] += 1
            if current_sp != correct_sp:
                tr_df.at[idx, 'sp'] = correct_sp
                fixes['n1904_sp'] += 1

    return tr_df, fixes


def main(config=None, dry_run=False):
    """Main entry point."""
    if config is None:
        config = load_config()

    logger.info("=" * 60)
    logger.info("PHASE 4.3: FIX NLP ERRORS")
    logger.info("=" * 60)

    intermediate_dir = Path(config["paths"]["data"]["intermediate"])

    # Load data
    logger.info("Loading data...")
    tr_path = intermediate_dir / "tr_complete.parquet"
    n1904_path = intermediate_dir / "n1904_words.parquet"

    if not tr_path.exists():
        logger.error(f"TR data not found: {tr_path}")
        return False

    tr_df = pd.read_parquet(tr_path)
    n1904_df = pd.read_parquet(n1904_path) if n1904_path.exists() else pd.DataFrame()

    nlp_count = (tr_df['source'] == 'nlp').sum()
    logger.info(f"  TR words: {len(tr_df):,}")
    logger.info(f"  NLP-sourced words: {nlp_count:,}")
    logger.info(f"  N1904 reference words: {len(n1904_df):,}")

    if dry_run:
        logger.info("[DRY RUN] Would apply corrections")
        return True

    # Build N1904 reference
    logger.info("Building N1904 reference...")
    n1904_ref = build_n1904_reference(n1904_df)
    logger.info(f"  Reference entries: {len(n1904_ref):,}")

    # Apply corrections
    logger.info("Applying corrections...")
    tr_df, fixes = apply_corrections(tr_df, n1904_ref)

    # Report
    total_fixes = sum(fixes.values())
    logger.info(f"  N1904-based lemma fixes: {fixes['n1904_lemma']:,}")
    logger.info(f"  N1904-based SP fixes: {fixes['n1904_sp']:,}")
    logger.info(f"  Manual lemma fixes: {fixes['manual_lemma']:,}")
    logger.info(f"  Manual SP fixes: {fixes['manual_sp']:,}")
    logger.info(f"  Total fixes: {total_fixes:,}")

    # Save
    tr_df.to_parquet(tr_path)
    logger.info(f"Saved: {tr_path}")

    # Summary
    logger.info("")
    logger.info("=" * 40)
    logger.info("NLP ERROR FIX SUMMARY")
    logger.info("=" * 40)
    logger.info(f"  NLP words processed: {nlp_count:,}")
    logger.info(f"  Total corrections: {total_fixes:,}")
    logger.info(f"  Correction rate: {total_fixes/nlp_count*100:.1f}%")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    success = main(dry_run=args.dry_run)
    sys.exit(0 if success else 1)
