# Textus Receptus Text-Fabric Dataset

A Text-Fabric dataset for the Stephanus 1550 Textus Receptus Greek New Testament with linguistic annotations.

## Why This Dataset?

This dataset combines the TR text with annotations (morphology, semantic domains, glosses) derived from N1904, filling a gap for TR users who want similar tooling to what exists for critical text editions.

| Resource | Text | Morphology | Semantic Domains | Glosses | Syntax Trees |
|----------|------|------------|------------------|---------|--------------|
| MACULA Greek | Nestle 1904 | ✅ | ✅ | ✅ | ✅ |
| Robinson-Pierpont | Byzantine | ✅ | ❌ | ❌ | ❌ |
| **This Dataset** | **TR 1550** | **✅** | **✅** | **✅** | **✅** |

### Hierarchical Structure

This dataset includes clause, phrase, and word group nodes for all verses. Structure is derived from N1904 where possible, with automatic inference for remaining verses:

| Structure Source | Verses | Coverage |
|-----------------|--------|----------|
| Direct transplant (100% word alignment) | 1,812 | 22.8% |
| Inferred (known words, different positions) | 3,214 | 40.4% |
| Generated from word assignments | 2,931 | 36.8% |

**Totals**: 7,964 clauses, 72,806 phrases, 19,520 word groups

Confidence breakdown: 99.3% high confidence (≥80%), 0.7% medium confidence (60-80%).

## Disclaimer

This dataset is a derivative work, not original academic scholarship. The annotations are transplanted from the [N1904 dataset](https://github.com/CenterBLC/N1904), which was created by biblical scholars at the Center for Biblical Languages and Computing. The author of this repository is not a biblical scholar. This dataset is provided for convenience and experimentation; please use it with discretion and verify findings against authoritative sources when accuracy matters.

## Overview

This project creates an annotated Text-Fabric dataset for the TR using a "Graft and Patch" strategy:

- **~89% of words**: Annotations transplanted from the aligned N1904 dataset
- **~11% of words**: Annotations generated via NLP + lexicon lookup (for TR-only variants)

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total words | 140,726 |
| Total verses | 7,957 |
| Books | 27 (complete NT) |
| Clauses | 7,964 |
| Phrases | 72,806 |
| Word groups | 19,520 |
| **Total nodes** | **249,260** |
| Unique lemmas | 7,943 |
| Word annotations from N1904 | 88.8% |
| Word annotations from NLP | 11.2% |

### High-Profile TR Variants Included

- Comma Johanneum (1 John 5:7-8)
- Eunuch's Confession (Acts 8:37)
- Pericope Adulterae (John 7:53-8:11)
- Longer Ending of Mark (Mark 16:9-20)
- Lord's Prayer Doxology (Matthew 6:13)

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.9+
- pandas
- text-fabric
- stanza (for NLP parsing)
- requests, beautifulsoup4 (for BLB download)

## Project Structure

```
tr/
├── config.yaml           # Pipeline configuration
├── run_pipeline.py       # Main pipeline runner
├── requirements.txt      # Python dependencies
├── scripts/
│   ├── download_blb_tr.py    # Download TR from Blue Letter Bible
│   ├── compare_tr_n1904.py   # Validation script
│   ├── phase1/               # Data acquisition
│   ├── phase2/               # Alignment & syntax transplant
│   ├── phase3/               # NLP for gaps
│   ├── phase4/               # Text-Fabric generation
│   ├── phase5/               # Quality assurance
│   └── utils/                # Shared utilities
├── data/
│   ├── source/               # Raw input data
│   ├── intermediate/         # Pipeline working files
│   └── output/               # Final TF dataset & reports
└── logs/                     # Pipeline execution logs
```

## Usage

### Running the Full Pipeline

```bash
python run_pipeline.py
```

Or run specific phases:

```bash
python run_pipeline.py --phase 2    # Run only Phase 2
python run_pipeline.py --from 3     # Run from Phase 3 onwards
python run_pipeline.py --dry-run    # Preview without executing
```

### Downloading Fresh TR Data

To download the TR from Blue Letter Bible (with caching):

```bash
python scripts/download_blb_tr.py          # Uses cache if available
python scripts/download_blb_tr.py --fresh  # Ignore cache, re-download everything
python scripts/download_blb_tr.py --clear-cache  # Delete cached HTML files
```

HTML pages are cached in `data/source/blb_cache/` to avoid re-scraping on subsequent runs.

### Validating Against N1904

```bash
python scripts/compare_tr_n1904.py
```

## Pipeline Phases

### Phase 1: Data Acquisition
- Download/load TR source text
- Download/load N1904 Text-Fabric dataset
- Validate source data

### Phase 2: Alignment & Syntax Transplant
- Align TR verses with N1904 verses
- Match words between aligned verses
- Transplant syntactic annotations from N1904 to TR

### Phase 3: NLP Gap Filling
- Identify words without transplanted syntax (TR-only variants)
- Parse with Stanza NLP
- Map Universal Dependencies labels to N1904 format

### Phase 4: Text-Fabric Generation
- Merge aligned and NLP-parsed data
- Fill glosses to achieve 100% coverage
- Fix systematic NLP errors (lemma/POS corrections using N1904 reference)
- Generate TF node features (word, lemma, pos, case, etc.)
- Generate TF edge features (parent relationships)
- Build container nodes (book, chapter, verse)
- **Structure transplant**: Classify verses, transplant clause/phrase/wg nodes from N1904
- **Structure inference**: Resolve unknown word forms, integrate structure into TF

### Phase 5: Quality Assurance
- Check for cycles in syntax trees
- Verify orphan nodes
- Validate feature completeness
- Spot-check high-profile variants
- Generate QA report

## Configuration

Edit `config.yaml` to customize:

- Data paths
- Alignment thresholds
- NLP settings
- Output features
- QA thresholds

## Data Sources

- **TR Text**: Stephanus 1550 from [Blue Letter Bible](https://www.blueletterbible.org/)
- **N1904 Syntax**: [CenterBLC/N1904](https://github.com/CenterBLC/N1904) Text-Fabric dataset

## Output

The final Text-Fabric dataset is generated in `data/output/tf/` with features including:

**Word Features:**

| Feature | Description | Coverage |
|---------|-------------|----------|
| unicode | Surface form (Greek) | 100% |
| text | Surface form (alias) | 100% |
| normalized | Unicode NFC normalized | 100% |
| lemma | Dictionary form | 100% |
| sp | Part of speech | 100% |
| cls | Word class (noun/verb/adj) | 100% |
| strong | Strong's number | 100% |
| morph | Morphology code | 100% |
| function | Syntactic function | 41% |
| role | Syntactic role (s/o/v/etc) | 76% |
| case | Grammatical case | 57% |
| gloss | English gloss | 100% |
| translit | Latin transliteration | 100% |
| lemmatranslit | Lemma transliteration | 100% |
| unaccent | Greek without diacritics | 100% |
| after | Trailing punctuation/space | 100% |
| trailer | Trailing material (alias) | 100% |
| ln | Louw-Nida semantic domains | 97% |
| bookshort | Book abbreviation (MAT) | 100% |
| num | Word position in verse | 100% |
| ref | Reference string (MAT 1:1!1) | 100% |
| id | Unique word ID | 100% |
| trans | Contextual translation | 97% |
| domain | Semantic domain codes | 90% |
| typems | Morphological subtype | 32% |

**Structure Features (clause/phrase/wg nodes):**

| Feature | Description |
|---------|-------------|
| typ | Phrase/clause type (NP, PP, VP) |
| function | Syntactic function (Subj, Pred, Objc, Cmpl) |
| rela | Relation to context |
| rule | Word group syntactic rule |
| clausetype | Clause type |
| structure_source | Origin: direct/inferred |
| structure_confidence | Confidence score (0-1) |

### Gloss Coverage

100% gloss coverage is achieved automatically as part of Phase 4:

```bash
# Just run the pipeline - glosses are filled automatically
python run_pipeline.py
```

| Source | Coverage |
|--------|----------|
| N1904 aligned | 88.8% |
| N1904 + lexicon lookup | 97.9% |
| Manual glosses + fallbacks | 100% |

See [docs/GLOSS_COVERAGE.md](docs/GLOSS_COVERAGE.md) for details.

## Using the Dataset

Load the dataset with Text-Fabric:

```python
from tf.fabric import Fabric

TF = Fabric(locations='data/output/tf')
api = TF.load('unicode lemma sp function typ')
F, T, L = api.F, api.T, api.L

# Navigate to a verse
verse = T.nodeFromSection(('I_Corinthians', 1, 5))
words = L.d(verse, otype='w')
print(f"Verse has {len(words)} words")

# Get word annotations
for w in words[:5]:
    print(f"{F.unicode.v(w)} - {F.lemma.v(w)} ({F.sp.v(w)})")

# Navigate structure (for verses with transplanted structure)
for w in words:
    phrases = L.u(w, otype='phrase')
    if phrases:
        p = phrases[0]
        print(f"Word '{F.unicode.v(w)}' in phrase with function: {F.function.v(p)}")
```

### Node Types

| Type | Count | Description |
|------|-------|-------------|
| w | 140,726 | Word (slot) nodes |
| verse | 7,957 | Verse containers |
| chapter | 260 | Chapter containers |
| book | 27 | Book containers |
| clause | 7,964 | Clause structure |
| phrase | 72,806 | Phrase structure |
| wg | 19,520 | Word group structure |

## License

MIT License. See [LICENSE](LICENSE) for details.
