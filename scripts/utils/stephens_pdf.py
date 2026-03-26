#!/usr/bin/env python3
"""
Utilities for reconstructing accented/punctuated Stephens 1550 tokens from the
Iglesia Reformada PDF witness while keeping the byztxt plain-text source as the
authoritative token sequence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import beta_code
from pypdf import PdfReader


BOOK_NAME_TO_CODE = {
    "Matthew": "MAT",
    "Mark": "MAR",
    "Luke": "LUK",
    "John": "JHN",
    "Acts": "ACT",
    "Romans": "ROM",
    "I Corinthians": "1CO",
    "II Corinthians": "2CO",
    "Galatians": "GAL",
    "Ephesians": "EPH",
    "Philippians": "PHP",
    "Colossians": "COL",
    "I Thessalonians": "1TH",
    "II Thessalonians": "2TH",
    "I Timothy": "1TI",
    "II Timothy": "2TI",
    "Titus": "TIT",
    "Philemon": "PHM",
    "Hebrews": "HEB",
    "James": "JAS",
    "I Peter": "1PE",
    "II Peter": "2PE",
    "I John": "1JN",
    "II John": "2JN",
    "III John": "3JN",
    "Jude": "JUD",
    "Revelation": "REV",
}

BOOK_CODE_TO_STV = {
    "MAT": "MT",
    "MAR": "MR",
    "LUK": "LU",
    "JHN": "JOH",
    "ACT": "AC",
    "ROM": "RO",
    "1CO": "1CO",
    "2CO": "2CO",
    "GAL": "GA",
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
    "1PE": "1PE",
    "2PE": "2PE",
    "1JN": "1JO",
    "2JN": "2JO",
    "3JN": "3JO",
    "JUD": "JUDE",
    "REV": "RE",
}

CHAPTER_HEADING_RE = re.compile(
    r"^(Matthew|Mark|Luke|John|Acts|Romans|I Corinthians|II Corinthians|Galatians|"
    r"Ephesians|Philippians|Colossians|I Thessalonians|II Thessalonians|I Timothy|"
    r"II Timothy|Titus|Philemon|Hebrews|James|I Peter|II Peter|I John|II John|"
    r"III John|Jude|Revelation)\s+(\d+)$"
)
VERSE_START_RE = re.compile(r"(?<!\d)(\d{1,3})\s+")
TRANS_MARK_RE = re.compile(r"[A-Za-z\[\]#;'/\\=(){}<>|+`]")
SKIP_LINES = {
    "TEXTUS RECEPTUS STEPHANUS 1550",
    "THE COMPLETE NEW TESTAMENT",
    "TH#S KAINH#S DIAYH;KHS A=PANTA",
    "TH#S KAINH#S",
    "DIAYH;KHS",
    "A=PANTA",
}
BOOK_HEADER_LINES = {
    "MATTHEW",
    "MARK",
    "LUKE",
    "JOHN",
    "ACTS",
    "ROMANS",
    "GALATIANS",
    "EPHESIANS",
    "PHILIPPIANS",
    "COLOSSIANS",
    "TITUS",
    "PHILEMON",
    "HEBREWS",
    "JAMES",
    "JUDE",
    "REVELATION",
    "I CORINTHIANS",
    "II CORINTHIANS",
    "I THESSALONIANS",
    "II THESSALONIANS",
    "I TIMOTHY",
    "II TIMOTHY",
    "I PETER",
    "II PETER",
    "I JOHN",
    "II JOHN",
    "III JOHN",
}
SINGLE_CHAPTER_HEADER_TO_CODE = {
    "PHILEMON": "PHM",
    "II JOHN": "2JN",
    "III JOHN": "3JN",
    "JUDE": "JUD",
}
BETA_PUNCT_CHARS = ".,;:?!>·,"


@dataclass
class AlignedWitnessToken:
    beta_word: str
    greek_word: str
    after: str
    merged_source: str


GREEK_TO_NORM = str.maketrans(
    {
        "α": "a",
        "β": "b",
        "γ": "g",
        "δ": "d",
        "ε": "e",
        "ζ": "z",
        "η": "h",
        "θ": "q",
        "ι": "i",
        "κ": "k",
        "λ": "l",
        "μ": "m",
        "ν": "n",
        "ξ": "x",
        "ο": "o",
        "π": "p",
        "ρ": "r",
        "σ": "s",
        "ς": "s",
        "τ": "t",
        "υ": "u",
        "φ": "f",
        "χ": "x",
        "ψ": "y",
        "ω": "w",
        "[": "",
        "]": "",
    }
)


def normalize_base_token(token: str) -> str:
    """Normalize an accentless STV token for alignment."""
    text = token.lower()
    text = text.replace("v", "s")
    text = text.replace("c", "x")
    text = text.replace("y", "q")
    return re.sub(r"[^a-z0-9]", "", text)


def normalize_witness_fragment(text: str) -> str:
    """Normalize a PDF witness fragment for alignment."""
    text = text.lower()
    text = text.replace("v", "s")
    text = text.replace("c", "x")
    text = text.replace("y", "q")
    text = text.replace("j", "")
    return re.sub(r"[^a-z0-9]", "", text)


def normalize_greek_base_token(token: str) -> str:
    """Normalize a Greek base token to the same Latin comparison space."""
    text = unicodedata.normalize("NFD", token.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = unicodedata.normalize("NFC", text)
    text = text.translate(GREEK_TO_NORM)
    return re.sub(r"[^a-z0-9]", "", text)


def beta_to_unicode(beta_text: str) -> str:
    """Convert Stephens-style beta code to Unicode Greek."""
    clean = beta_text.strip().lower()
    prefix = ""
    while clean and clean[0] in "[]/=`\\{}":
        prefix += clean[0]
        clean = clean[1:]
    if prefix and clean:
        clean = clean[0] + prefix + clean[1:]

    clean = clean.replace("{", "<RCIRC>")
    clean = clean.replace("}", "<SCIRC>")
    clean = clean.replace("`", "<RGRAVE>")
    clean = clean.replace("\\", "<SGRAVE>")
    clean = clean.replace("[", "<ROUGH>")
    clean = clean.replace("]", "<SMOOTH>")
    clean = clean.replace("#", "<CIRC>")
    clean = clean.replace(";", "/")
    clean = clean.replace("'", "/")
    clean = clean.replace("=", "(/")
    clean = clean.replace("<RCIRC>", "(=")
    clean = clean.replace("<SCIRC>", ")=")
    clean = clean.replace("<RGRAVE>", "(\\")
    clean = clean.replace("<SGRAVE>", ")\\")
    clean = clean.replace("<ROUGH>", "(")
    clean = clean.replace("<SMOOTH>", ")")
    clean = clean.replace("<CIRC>", "=")
    clean = clean.replace("v", "s")
    clean = clean.replace("c", "<CHI>")
    clean = clean.replace("x", "c")
    clean = clean.replace("<CHI>", "x")
    clean = clean.replace("y", "<THETA>")
    clean = clean.replace("q", "y")
    clean = clean.replace("<THETA>", "q")
    greek = beta_code.beta_code_to_greek(clean)
    return greek


def match_score(target: str, candidate: str) -> float | None:
    """Score how plausibly a witness candidate matches a base token."""
    if not candidate:
        return None
    if candidate == target:
        return 1.0
    if (
        len(target) == len(candidate)
        and len(target) > 1
        and target[:-1] == candidate[:-1]
        and {target[-1], candidate[-1]} <= {"k", "x"}
    ):
        return 0.985

    target_variants = {
        target,
        target.rstrip("n"),
        target.rstrip("s"),
        target.rstrip("n").rstrip("s"),
    }
    candidate_variants = {
        candidate,
        candidate.rstrip("n"),
        candidate.rstrip("s"),
        candidate.rstrip("n").rstrip("s"),
    }
    if target_variants & candidate_variants:
        return 0.99

    ratio = SequenceMatcher(None, target, candidate).ratio()
    if ratio >= 0.84:
        return ratio
    return None


def cleanup_after(after: str) -> str:
    """Normalize trailing punctuation/spacing after a witness token."""
    text = after.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if not text:
        return " "
    return f"{text} "


def clean_witness_text(text: str) -> str:
    """Normalize a few recurring PDF extraction quirks before token alignment."""
    cleaned = re.sub(r"For\s+more\s+public.*$", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"Formorepublic.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"http://.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"b>$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bd\s+iati([;:,.>?]*)", r"dia ti\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(dia['`]?)ti([;:,.>?]*)", r"\1 ti\2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(i[\[\]]?na['`]?)ti([;:,.>?]*)", r"\1 ti\2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(mh[;:'`]?)tiv([;:,.>?]*)", r"\1 tiv\2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(dia)pa;?nto['`]?v\b", r"\1 pa;nto'v", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(a[\[\]]?na)me;?son\b", r"\1 me;son", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(toute;sti)\b", "tout e;sti", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"e\s+\]kk", "e]kk", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"a\]\s+delf", "a]delf", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_token_and_after(raw_merged: str) -> tuple[str, str]:
    """Split a merged witness token into beta-word and trailing punctuation."""
    trimmed = raw_merged.strip()
    match = re.match(r"^(.*?)([.,;:?!>·]+)?$", trimmed)
    if not match:
        return trimmed, " "
    beta_word = match.group(1) or trimmed
    trailing = match.group(2) or ""
    return beta_word, cleanup_after(trailing)


def parse_stv_verses(repo_dir: Path) -> dict[tuple[str, int, int], list[str]]:
    """Parse the repo's plain-text STV files into verse token lists."""
    verse_map: dict[tuple[str, int, int], list[str]] = {}
    stv_dir = repo_dir / "textonly"

    for book_code, stv_name in BOOK_CODE_TO_STV.items():
        path = stv_dir / f"{stv_name}.STV"
        if not path.exists():
            raise FileNotFoundError(f"Missing STV base text: {path}")

        current_key: tuple[str, int, int] | None = None
        current_parts: list[str] = []

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                continue

            match = re.match(r"^(\d+):(\d+)\s+(.*)$", stripped)
            if match:
                if current_key is not None:
                    verse_map[current_key] = " ".join(current_parts).split()
                current_key = (book_code, int(match.group(1)), int(match.group(2)))
                verse_text = re.sub(r"^\[[^\]]+\]\s*", "", match.group(3).strip())
                verse_text = re.sub(r"\[\d+:\d+\]", "", verse_text)
                current_parts = [verse_text] if verse_text else []
            elif current_key is not None:
                current_parts.append(re.sub(r"\[\d+:\d+\]", "", stripped))

        if current_key is not None:
            verse_map[current_key] = " ".join(current_parts).split()

    return verse_map


def parse_stv_chapters(repo_dir: Path) -> dict[tuple[str, int], list[str]]:
    """Flatten STV verse tokens into chapter token streams."""
    verse_map = parse_stv_verses(repo_dir)
    chapter_map: dict[tuple[str, int], list[str]] = {}
    for (book, chapter, verse), tokens in sorted(verse_map.items()):
        chapter_map.setdefault((book, chapter), []).extend(tokens)
    return chapter_map


def extract_pdf_chapters(pdf_path: Path) -> dict[tuple[str, int], list[str]]:
    """Extract PDF body lines grouped by book/chapter heading."""
    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        reader.decrypt("")

    chapters: dict[tuple[str, int], list[str]] = {}
    current_key: tuple[str, int] | None = None

    for page in reader.pages:
        text = page.extract_text() or ""
        for raw_line in text.splitlines():
            squashed = " ".join(raw_line.split())
            if not squashed:
                continue

            match = CHAPTER_HEADING_RE.match(squashed)
            if match:
                current_key = (BOOK_NAME_TO_CODE[match.group(1)], int(match.group(2)))
                chapters.setdefault(current_key, [])
                continue

            if squashed in SINGLE_CHAPTER_HEADER_TO_CODE:
                current_key = (SINGLE_CHAPTER_HEADER_TO_CODE[squashed], 1)
                chapters.setdefault(current_key, [])
                continue

            if (
                squashed in SKIP_LINES
                or squashed in BOOK_HEADER_LINES
                or re.fullmatch(r"\d+", squashed)
                or "bibletranslation.ws" in squashed.lower()
                or "public domain pdfs" in squashed.lower()
                or squashed.startswith("TO' KATA'")
                or squashed.startswith("A=GION")
                or not TRANS_MARK_RE.search(squashed)
            ):
                continue

            if current_key is not None:
                chapters[current_key].append(raw_line.rstrip())

    return chapters


def split_chapter_lines_to_verses(lines: list[str]) -> dict[int, str]:
    """Split one chapter's PDF lines into verse-level transliterated text."""
    text = " ".join(line.strip() for line in lines if line.strip())
    verses: dict[int, str] = {}
    matches = list(VERSE_START_RE.finditer(text))
    if not matches:
        return verses

    for index, match in enumerate(matches):
        verse = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start(1) if index + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        verses[verse] = chunk

    return verses


def extract_pdf_verses(pdf_path: Path) -> dict[tuple[str, int, int], str]:
    """Extract verse-level transliterated witness text from the PDF."""
    chapters = extract_pdf_chapters(pdf_path)
    verses: dict[tuple[str, int, int], str] = {}
    for (book, chapter), lines in chapters.items():
        for verse, text in split_chapter_lines_to_verses(lines).items():
            verses[(book, chapter, verse)] = text
    return verses


def extract_pdf_chapter_texts(pdf_path: Path) -> dict[tuple[str, int], str]:
    """Extract chapter-wide witness text, dropping embedded verse numbers."""
    chapters = extract_pdf_chapters(pdf_path)
    chapter_texts: dict[tuple[str, int], str] = {}
    for key, lines in chapters.items():
        text = " ".join(line.strip() for line in lines if line.strip())
        first_match = VERSE_START_RE.search(text)
        if first_match:
            text = text[first_match.start() :]
        text = re.sub(r"(?<!\d)\d{1,3}\s+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        chapter_texts[key] = text
    return chapter_texts


def _align_witness_fragments(
    base_tokens: list[str],
    fragments: list[str],
    normalizer,
    start_index: int,
) -> tuple[list[AlignedWitnessToken], int] | None:
    """Attempt one greedy alignment starting at a specific witness fragment index."""
    aligned: list[AlignedWitnessToken] = []
    frag_index = start_index

    for base_token in base_tokens:
        target = normalizer(base_token)
        best_end: int | None = None
        best_raw: str | None = None
        best_score: float = -1.0
        merged_parts: list[str] = []

        for lookahead in range(frag_index, min(frag_index + 8, len(fragments))):
            merged_parts.append(fragments[lookahead])
            merged_raw = "".join(merged_parts)
            merged_norm = normalize_witness_fragment(merged_raw)
            if not merged_norm:
                continue

            score = match_score(target, merged_norm)
            if score is not None and score > best_score:
                best_end = lookahead + 1
                best_raw = merged_raw
                best_score = score
                if score >= 0.999:
                    break

            if len(merged_norm) > len(target) + 4:
                break

        if best_end is None or best_raw is None:
            return None

        frag_index = best_end
        merged_full = best_raw
        while frag_index < len(fragments) and not normalize_witness_fragment(fragments[frag_index]):
            merged_full += fragments[frag_index]
            frag_index += 1

        beta_word, after = split_token_and_after(merged_full)
        greek_word = beta_to_unicode(beta_word)
        aligned.append(
            AlignedWitnessToken(
                beta_word=beta_word,
                greek_word=greek_word,
                after=after,
                merged_source=merged_full,
            )
        )

    return aligned, frag_index


def align_witness_tokens(
    base_tokens: list[str],
    witness_text: str,
    normalizer,
) -> list[AlignedWitnessToken] | None:
    """
    Align one witness token stream to the base token sequence.

    The base token order is authoritative. We greedily accumulate witness
    fragments until their normalized form matches the next base token.
    """
    cleaned_witness = clean_witness_text(witness_text)
    fragments = [frag for frag in re.split(r"\s+", cleaned_witness.strip()) if frag]
    if not fragments:
        return None

    attempts: list[tuple[list[AlignedWitnessToken], int]] = []
    for start_index in range(0, min(20, len(fragments))):
        attempt = _align_witness_fragments(base_tokens, fragments, normalizer, start_index)
        if attempt is None:
            continue
        attempts.append(attempt)
        if start_index == 0:
            break

    for aligned, frag_index in attempts:
        if frag_index < len(fragments):
            remainder = "".join(fragments[frag_index:]).strip()
            if remainder:
                aligned[-1].after = cleanup_after((aligned[-1].after + remainder).strip())
        return aligned

    return None


def align_witness_verse(base_tokens: list[str], witness_text: str) -> list[AlignedWitnessToken] | None:
    """Align one PDF witness verse to the repo's STV transliteration sequence."""
    return align_witness_tokens(base_tokens, witness_text, normalize_base_token)


def align_witness_chapter(base_tokens: list[str], witness_text: str) -> list[AlignedWitnessToken] | None:
    """Align a whole PDF chapter against the STV chapter token sequence."""
    return align_witness_verse(base_tokens, witness_text)


def align_witness_to_greek_base(base_tokens: list[str], witness_text: str) -> list[AlignedWitnessToken] | None:
    """Align the witness directly against Greek base tokens when STV is insufficient."""
    return align_witness_tokens(base_tokens, witness_text, normalize_greek_base_token)
