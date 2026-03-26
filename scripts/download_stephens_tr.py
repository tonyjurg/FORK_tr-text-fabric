#!/usr/bin/env python3
"""
Download or refresh the public-domain Stephens 1550 text source.

Source repo:
    https://github.com/byztxt/greektext-stephens

Output:
    data/source/tr_source.csv

The script keeps a shallow local checkout of the upstream repository in
`data/source/greektext-stephens/` and converts the Unicode per-book text files
into a normalized word-level CSV for the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.config import load_config


BOOK_MAP = {
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
    "1PE": "1PE",
    "2PE": "2PE",
    "1JO": "1JN",
    "2JO": "2JN",
    "3JO": "3JN",
    "JUDE": "JUD",
    "RE": "REV",
}

VERSE_RE = re.compile(r"^(\d+):(\d+)\s+(.*)$")


def run_git(args: list[str], cwd: Path | None = None) -> None:
    """Run a git command with helpful error propagation."""
    command = ["git", *args]
    try:
        subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to acquire greektext-stephens") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"git command failed: {' '.join(command)}") from exc


def ensure_repo(repo_url: str, repo_dir: Path, branch: str, fresh: bool = False) -> Path:
    """Clone or refresh the upstream public-domain source repository."""
    if repo_dir.exists() and (repo_dir / ".git").exists():
        if fresh:
            run_git(["checkout", branch], cwd=repo_dir)
            run_git(["pull", "--ff-only", "origin", branch], cwd=repo_dir)
        return repo_dir

    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run_git(["clone", "--depth", "1", "--branch", branch, repo_url, str(repo_dir)])
    return repo_dir


def parse_repo_to_rows(repo_dir: Path) -> list[dict[str, object]]:
    """Parse the Unicode text files into a word-level row list."""
    rows = []
    unicode_dir = repo_dir / "textonly" / "unicode"

    for source_code, book_code in BOOK_MAP.items():
        path = unicode_dir / f"{source_code}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing expected source file: {path}")

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("["):
                continue

            match = VERSE_RE.match(line)
            if not match:
                continue

            chapter = int(match.group(1))
            verse = int(match.group(2))
            text = match.group(3).strip()
            words = text.split()

            for word_rank, word in enumerate(words, start=1):
                rows.append(
                    {
                        "book": book_code,
                        "chapter": chapter,
                        "verse": verse,
                        "word_rank": word_rank,
                        "word": word,
                        "after": " ",
                        "strong": "",
                        "morph": "",
                    }
                )

    return rows


def download_all(fresh: bool = False) -> Path:
    """Refresh the local source checkout and export the normalized CSV."""
    config = load_config()
    source_dir = Path(config["paths"]["data"]["source"])
    repo_url = config["sources"]["tr"]["repo_url"]
    repo_branch = config["sources"]["tr"].get("repo_branch", "master")
    repo_dir = Path(config["sources"]["tr"]["local_checkout"])
    output_path = source_dir / "tr_source.csv"

    ensure_repo(repo_url, repo_dir, repo_branch, fresh=fresh)
    rows = parse_repo_to_rows(repo_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["book", "chapter", "verse", "word_rank", "word", "after", "strong", "morph"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} words to {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="refresh the local git checkout")
    args = parser.parse_args()

    path = download_all(fresh=args.fresh)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
