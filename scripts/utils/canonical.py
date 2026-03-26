"""
Canonical ordering helpers for the TR Text-Fabric pipeline.

These helpers keep the New Testament book order consistent across
intermediate data generation, structure regeneration, and final TF export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


NT_CANONICAL_BOOKS: tuple[str, ...] = (
    "MAT",
    "MAR",
    "LUK",
    "JHN",
    "ACT",
    "ROM",
    "1CO",
    "2CO",
    "GAL",
    "EPH",
    "PHP",
    "COL",
    "1TH",
    "2TH",
    "1TI",
    "2TI",
    "TIT",
    "PHM",
    "HEB",
    "JAS",
    "1PE",
    "2PE",
    "1JN",
    "2JN",
    "3JN",
    "JUD",
    "REV",
)

BOOK_ORDER_MAP: dict[str, int] = {
    book: order for order, book in enumerate(NT_CANONICAL_BOOKS, start=1)
}

BOOK_NAME_MAP: dict[str, str] = {
    "MAT": "Matthew",
    "MAR": "Mark",
    "LUK": "Luke",
    "JHN": "John",
    "ACT": "Acts",
    "ROM": "Romans",
    "1CO": "I_Corinthians",
    "2CO": "II_Corinthians",
    "GAL": "Galatians",
    "EPH": "Ephesians",
    "PHP": "Philippians",
    "COL": "Colossians",
    "1TH": "I_Thessalonians",
    "2TH": "II_Thessalonians",
    "1TI": "I_Timothy",
    "2TI": "II_Timothy",
    "TIT": "Titus",
    "PHM": "Philemon",
    "HEB": "Hebrews",
    "JAS": "James",
    "1PE": "I_Peter",
    "2PE": "II_Peter",
    "1JN": "I_John",
    "2JN": "II_John",
    "3JN": "III_John",
    "JUD": "Jude",
    "REV": "Revelation",
}


def get_tf_dataset_dir(config: dict) -> Path:
    """Return the canonical TF dataset directory for the configured version."""
    return Path(config["paths"]["root"]) / "tf" / str(config["tf_output"]["version"])


def get_book_order(book: str) -> int:
    """Return the canonical NT order for a book abbreviation."""
    try:
        return BOOK_ORDER_MAP[str(book)]
    except KeyError as exc:
        raise ValueError(f"Unknown NT book abbreviation: {book}") from exc


def canonical_book_names(books: Iterable[str]) -> list[str]:
    """Return book abbreviations sorted in canonical NT order."""
    unique_books = {str(book) for book in books}
    return [book for book in NT_CANONICAL_BOOKS if book in unique_books]


def canonical_sort_columns(
    df,
    extra_columns: Sequence[str] | None = None,
    include_word_id: bool = True,
) -> list[str]:
    """Return a safe canonical sort column list for a TR dataframe."""
    columns = ["book_order", "chapter", "verse"]
    if "word_rank" in df.columns:
        columns.append("word_rank")
    if extra_columns:
        columns.extend(extra_columns)
    if include_word_id and "word_id" in df.columns and "word_id" not in columns:
        columns.append("word_id")
    return columns


def add_canonical_order(df):
    """Return a copy of a dataframe with a numeric canonical book order column."""
    ordered = df.copy()
    ordered["book_order"] = ordered["book"].map(get_book_order)
    if ordered["book_order"].isna().any():
        bad_books = sorted(ordered.loc[ordered["book_order"].isna(), "book"].astype(str).unique())
        raise ValueError(f"Unknown NT books encountered: {bad_books}")
    ordered["book_order"] = ordered["book_order"].astype(int)
    return ordered


def sort_canonically(df, extra_columns: Sequence[str] | None = None, include_word_id: bool = True):
    """Return a canonically sorted dataframe without the helper sort column."""
    ordered = add_canonical_order(df)
    sort_columns = canonical_sort_columns(
        ordered, extra_columns=extra_columns, include_word_id=include_word_id
    )
    ordered = ordered.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    return ordered.drop(columns=["book_order"])


def validate_word_id_canonical_order(df) -> None:
    """
    Validate that canonical sorting is identical to ascending word_id order.

    The current pipeline relies on word_id already representing Matthew-to-
    Revelation order. This check fails loudly if that invariant is broken.
    """
    if "word_id" not in df.columns:
        raise ValueError("Expected a word_id column for canonical order validation")

    by_word_id = df.sort_values("word_id", kind="mergesort").reset_index(drop=True)
    by_canonical = sort_canonically(df)

    left = by_word_id[["word_id", "book", "chapter", "verse"]]
    right = by_canonical[["word_id", "book", "chapter", "verse"]]
    if not left.equals(right):
        mismatch = left.compare(right).head(10)
        raise ValueError(
            "Canonical book/chapter/verse order does not match ascending word_id.\n"
            f"First mismatches:\n{mismatch}"
        )


def full_book_name(book: str) -> str:
    """Map a book abbreviation to the full N1904-compatible book name."""
    return BOOK_NAME_MAP.get(str(book), str(book))
