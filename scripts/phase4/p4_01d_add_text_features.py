#!/usr/bin/env python3
"""
Script: p4_01d_add_text_features.py
Phase: 4 - Compilation
Purpose: Add text features to match N1904: translit, unaccent, after, ln

Input:  data/intermediate/tr_complete.parquet
Output: data/intermediate/tr_complete.parquet (updated in place)

Features added:
- translit: Greek-to-Latin transliteration of word
- lemmatranslit: Greek-to-Latin transliteration of lemma
- unaccent: Word with diacritics removed (still Greek letters)
- after: Punctuation/spacing after the word
- ln: Louw-Nida semantic domain codes

Usage:
    python -m scripts.phase4.p4_01d_add_text_features
    python -m scripts.phase4.p4_01d_add_text_features --dry-run
"""

import argparse
import sys
import unicodedata
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.config import load_config
from scripts.utils.logging import ScriptLogger, get_logger


# Greek to Latin transliteration mapping (scholarly standard)
GREEK_TRANSLIT = {
    # Lowercase
    'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e',
    'ζ': 'z', 'η': 'e', 'θ': 'th', 'ι': 'i', 'κ': 'k',
    'λ': 'l', 'μ': 'm', 'ν': 'n', 'ξ': 'x', 'ο': 'o',
    'π': 'p', 'ρ': 'r', 'σ': 's', 'ς': 's', 'τ': 't',
    'υ': 'u', 'φ': 'ph', 'χ': 'kh', 'ψ': 'ps', 'ω': 'o',
    # Uppercase
    'Α': 'A', 'Β': 'B', 'Γ': 'G', 'Δ': 'D', 'Ε': 'E',
    'Ζ': 'Z', 'Η': 'E', 'Θ': 'Th', 'Ι': 'I', 'Κ': 'K',
    'Λ': 'L', 'Μ': 'M', 'Ν': 'N', 'Ξ': 'X', 'Ο': 'O',
    'Π': 'P', 'Ρ': 'R', 'Σ': 'S', 'Τ': 'T', 'Υ': 'U',
    'Φ': 'Ph', 'Χ': 'Kh', 'Ψ': 'Ps', 'Ω': 'O',
    # Rough breathing mark (when preserved)
    'ʽ': 'h',
}


def strip_accents(text: str) -> str:
    """
    Remove diacritical marks from Greek text, keeping base letters.

    This normalizes to NFD (decomposed), removes combining marks,
    then recomposes to NFC.
    """
    if not text:
        return text

    # Normalize to decomposed form
    decomposed = unicodedata.normalize('NFD', text)

    # Remove combining diacritical marks (category M)
    stripped = ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn')

    # Recompose
    return unicodedata.normalize('NFC', stripped)


def transliterate_greek(text: str) -> str:
    """
    Transliterate Greek text to Latin characters.

    Uses a scholarly transliteration scheme similar to N1904.
    """
    if not text:
        return text

    # First strip accents to get base letters
    base = strip_accents(text)

    # Transliterate character by character
    result = []
    for char in base:
        if char in GREEK_TRANSLIT:
            result.append(GREEK_TRANSLIT[char])
        elif char.isascii():
            # Keep ASCII characters as-is (punctuation, etc.)
            result.append(char)
        else:
            # Unknown character - keep original
            result.append(char)

    return ''.join(result)


def build_ln_lookup(n1904_tf_path: str) -> dict:
    """
    Build a lookup table for Louw-Nida semantic domain codes.

    Returns dict mapping (lemma, strong) -> ln_code
    """
    from tf.fabric import Fabric

    logger = get_logger(__name__)
    logger.info(f"Loading N1904 TF data for ln lookup from: {n1904_tf_path}")

    TF = Fabric(locations=n1904_tf_path, silent='deep')
    api = TF.load('lemma ln strong', silent='deep')

    # Build lookup: (lemma, strong) -> ln
    ln_lookup = {}
    for w in api.F.otype.s('word'):
        lemma = api.F.lemma.v(w)
        ln = api.F.ln.v(w)
        strong = api.F.strong.v(w) if hasattr(api.F, 'strong') else None

        if lemma and ln:
            # Primary key: (lemma, strong)
            if strong:
                key = (lemma, strong)
                if key not in ln_lookup:
                    ln_lookup[key] = ln
            # Fallback key: just lemma
            if lemma not in ln_lookup:
                ln_lookup[lemma] = ln

    logger.info(f"Built ln lookup with {len(ln_lookup)} entries")
    return ln_lookup


def extract_punctuation(word: str) -> tuple:
    """
    Extract trailing punctuation from a word.

    Returns (clean_word, after_punctuation)
    """
    # Common Greek/Unicode punctuation
    punct_chars = '.,;·:!?()[]—\u0387'  # includes Greek ano teleia

    after = ''
    clean = word

    # Extract trailing punctuation
    while clean and clean[-1] in punct_chars:
        after = clean[-1] + after
        clean = clean[:-1]

    # Add space if there's trailing content or it's end of word
    if not after:
        after = ' '  # Default trailing space
    else:
        after = after + ' '  # Punctuation followed by space

    return clean, after


def add_text_features(df, ln_lookup: dict, logger):
    """
    Add text features to the dataframe.

    Features added:
    - translit: transliteration of word
    - lemmatranslit: transliteration of lemma
    - unaccent: word without diacritics
    - after: trailing punctuation/space
    - ln: Louw-Nida codes
    """
    import pandas as pd

    logger.info("Adding translit feature...")
    df['translit'] = df['word'].apply(transliterate_greek)

    logger.info("Adding lemmatranslit feature...")
    df['lemmatranslit'] = df['lemma'].apply(transliterate_greek)

    logger.info("Adding unaccent feature...")
    df['unaccent'] = df['word'].apply(strip_accents)

    logger.info("Adding after feature...")
    # For now, use simple space - we don't have raw punctuation data
    # TODO: Parse from source if we have access to raw verse text
    df['after'] = ' '

    logger.info("Adding ln feature...")

    def lookup_ln(row):
        lemma = row.get('lemma')
        strong = row.get('strong')

        if not lemma:
            return None

        # Try (lemma, strong) first
        if strong:
            key = (lemma, strong)
            if key in ln_lookup:
                return ln_lookup[key]

        # Fall back to just lemma
        if lemma in ln_lookup:
            return ln_lookup[lemma]

        return None

    df['ln'] = df.apply(lookup_ln, axis=1)

    # Report coverage
    ln_coverage = df['ln'].notna().sum() / len(df) * 100
    logger.info(f"ln coverage: {ln_coverage:.1f}%")

    return df


def main(config: dict = None, dry_run: bool = False) -> bool:
    """Main entry point."""
    if config is None:
        config = load_config()

    logger = get_logger(__name__)

    complete_path = Path(config["paths"]["data"]["intermediate"]) / "tr_complete.parquet"
    n1904_tf_path = "/home/michael/text-fabric-data/github/CenterBLC/N1904/tf/1.0.0"

    if dry_run:
        logger.info("[DRY RUN] Would add text features: translit, lemmatranslit, unaccent, after, ln")
        return True

    import pandas as pd

    # Check input
    if not complete_path.exists():
        logger.error(f"Input not found: {complete_path}")
        return False

    # Load data
    logger.info(f"Loading data from: {complete_path}")
    df = pd.read_parquet(complete_path)
    logger.info(f"Loaded {len(df)} words")

    # Check if features already exist
    new_features = ['translit', 'lemmatranslit', 'unaccent', 'after', 'ln']
    existing = [f for f in new_features if f in df.columns]
    if existing:
        logger.info(f"Features already exist (will be overwritten): {existing}")

    # Build ln lookup
    ln_lookup = build_ln_lookup(n1904_tf_path)

    # Add features
    df = add_text_features(df, ln_lookup, logger)

    # Save
    logger.info(f"Saving to: {complete_path}")
    df.to_parquet(complete_path, index=False)

    # Report
    logger.info("\nFeature summary:")
    for feat in new_features:
        non_null = df[feat].notna().sum()
        logger.info(f"  {feat}: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    with ScriptLogger("p4_01d_add_text_features") as logger:
        config = load_config()
        success = main(config, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
