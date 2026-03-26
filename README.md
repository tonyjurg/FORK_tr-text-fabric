# Textus Receptus Text-Fabric Dataset

A Text-Fabric dataset for the Stephanus 1550 Textus Receptus Greek New Testament with canonical slot order and reusable linguistic annotations.

## Why This Dataset?

This project combines:

- a public-domain Stephens 1550 base text from `byztxt/greektext-stephens`
- an optional PDF witness overlay for accents and punctuation
- lexical and syntactic annotations derived from the N1904 Text-Fabric corpus
- a canonical TF export so the corpus starts at Matthew 1:1 and runs through Revelation 22:21

## Current Build Snapshot

| Metric | Value |
|--------|-------|
| Total words | 140,764 |
| Total verses | 7,957 |
| Chapters | 260 |
| Books | 27 |
| Clauses | 19,256 |
| Phrases | 68,249 |
| Word groups | 31,966 |
| Total nodes | 268,479 |
| Directly aligned N1904 words | 121,768 |
| NLP gap words | 18,996 |
| Strong/morph from direct N1904 alignment | 121,768 |
| Strong/morph additionally projected by `word+lemma+sp` | 3,583 |
| Strong/morph still unresolved | 15,413 |

## Structure Quality

This dataset includes clause, phrase, and word-group nodes for all verses. Structure provenance is mixed:

| Structure Source | Verses | Coverage |
|-----------------|--------|----------|
| Direct transplant | 1,529 | 19.2% |
| Inferred | 1,411 | 17.7% |
| Unknown-only / generated fallback | 5,017 | 63.1% |

This means the dataset is complete and navigable, but much more of the corpus depends on inferred/generated structure than the earlier pre-rebase experiments did. For analyses that need the strongest alignment to N1904 syntax, filter to `structure_source=direct`.

## Source And Annotation Model

The current pipeline uses these layers:

- **Base text**: public-domain Stephens 1550 from [byztxt/greektext-stephens](https://github.com/byztxt/greektext-stephens)
- **Punctuation/accent witness**: a configured PDF witness used only as an overlay on the plain-text base
- **Syntax and lexical reference**: [CenterBLC/N1904](https://github.com/CenterBLC/N1904) Text-Fabric dataset

Important distinctions:

- The Stephens plain text is the authoritative token sequence.
- The PDF witness can improve `after` and accented surface forms, but it does not replace the base token order.
- `strong` and `morph` now come from the original N1904 TF source where possible.
- For NLP-only words, `strong` and `morph` may also be projected conservatively when N1904 has a unique `word+lemma+sp` match.

High-profile TR variants remain present in the base text, including:

- Comma Johanneum (`1 John 5:7-8`)
- Eunuch's Confession (`Acts 8:37`)
- Pericope Adulterae (`John 7:53-8:11`)
- Longer Ending of Mark (`Mark 16:9-20`)
- Lord's Prayer Doxology (`Matthew 6:13`)

## Disclaimer

This dataset is a derivative work, not original academic scholarship. The annotations are transplanted or projected from the [N1904 dataset](https://github.com/CenterBLC/N1904), which was created by biblical scholars at the Center for Biblical Languages and Computing. The author of this repository is not a biblical scholar. Please verify important findings against authoritative sources.

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:

- Python 3.9+
- pandas
- text-fabric
- stanza
- git

The rebuild path also expects a local N1904 TF checkout under `data/source/N1904/`.

## Project Layout

```text
tr/
|-- config.yaml
|-- run_pipeline.py
|-- scripts/
|   |-- download_stephens_tr.py
|   |-- phase1/
|   |-- phase2/
|   |-- phase3/
|   |-- phase4/
|   `-- utils/
|-- data/
|   |-- source/
|   `-- intermediate/
|-- tf/
|   `-- 1.0/
|-- reports/
`-- logs/
```

## Usage

### Full Pipeline

```bash
python run_pipeline.py
```

### Run Specific Parts

```bash
python run_pipeline.py --phase 2
python run_pipeline.py --phase 4 --step 4
python run_pipeline.py --dry-run
```

### Acquire Fresh Stephens Source

```bash
python scripts/download_stephens_tr.py
python scripts/download_stephens_tr.py --fresh
```

This updates `data/source/tr_source.csv`.

### Apply The PDF Witness Overlay

```bash
python -m scripts.phase1.p1_04b_apply_punctuation_witness
python -m scripts.phase1.p1_05_build_tr_dataframe
```

When present, `data/source/tr_source_prepared.csv` becomes the preferred phase-1 input.

### No-Claude Rebuild Path

The current rebuild path does not require the optional `p3_06_review_variants` step.

```bash
python run_pipeline.py --phase 2
python run_pipeline.py --phase 3 --step 1
python run_pipeline.py --phase 4 --step 1
```

Or explicitly from phase 4:

```bash
python -m scripts.phase4.p4_01_merge_data
python -m scripts.phase4.p4_01b_fill_glosses
python -m scripts.phase4.p4_01c_fix_nlp_errors
python -m scripts.phase4.p4_01d_project_strong_morph
python -m scripts.phase4.p4_02_generate_containers
python -m scripts.phase4.p4_08a_prepare_structure_data
python -m scripts.phase4.p4_08b_transplant_structure
python -m scripts.phase4.p4_08c_infer_structure
python -m scripts.phase4.p4_08d_handle_unknowns
python -m scripts.phase4.p4_08e_generate_structure_tf
python -m scripts.phase4.p4_08h_generate_clauses_wg
python -m scripts.phase4.p4_04_generate_features
python -m scripts.phase4.p4_07_verify_build
```

## Pipeline Phases

### Phase 1: Data Acquisition

- acquire the public-domain Stephens source
- optionally overlay punctuation and accents from the configured PDF witness
- build `tr_words.parquet`

### Phase 2: Alignment

- extract N1904 word data from the original TF source
- align Stephens verses and words to N1904
- transplant aligned lexical and syntactic features

### Phase 3: NLP Gap Filling

- identify unaligned words
- parse them locally with Stanza
- convert UD-style output into the project's N1904-like feature surface

### Phase 4: Text-Fabric Generation

- merge aligned and NLP-generated annotations
- fill glosses to 100%
- fix systematic NLP lemma/POS errors
- project `strong` and `morph` by unique `word+lemma+sp` matches
- regenerate canonical containers and structure nodes
- export `tf/<version>/` through `tf.convert.walker.CV.walk()`

### Phase 5: QA

- verify build integrity
- run cycle/orphan/feature checks
- generate QA reports

## How The `.tf` Files Are Created

The final `.tf` files are produced by [p4_04_generate_features.py](D:/Onedrive/GitHub/FORK_tr-text-fabric/scripts/phase4/p4_04_generate_features.py) with Text-Fabric's `tf.convert.walker.CV.walk()`.

The build process is:

1. Load `tr_complete.parquet`, `tr_containers.parquet`, and `tr_structure_nodes.parquet`.
2. Sort the word table in canonical NT order.
3. Emit slots in canonical order so slot `1` is Matthew 1:1.
4. Emit `book`, `chapter`, and `verse` section nodes while slots are created.
5. Emit explicit `clause`, `phrase`, and `wg` nodes from `tr_structure_nodes.parquet`.
6. Write scalar node features and the `parent` edge.
7. Let Text-Fabric serialize the final feature files in `tf/<version>/`.

See [docs/TF_BUILD.md](D:/Onedrive/GitHub/FORK_tr-text-fabric/docs/TF_BUILD.md) for the fuller build walkthrough and metadata reference.

## Feature Highlights

Common exported features include:

- word features: `unicode`, `lemma`, `strong`, `morph`, `sp`, `gloss`, `after`, `translit`, `unaccent`, `normalized`
- section features: `book`, `chapter`, `verse`
- syntax features: `function`, `role`, `parent`
- structure features: `typ`, `rela`, `rule`, `clausetype`, `structure_source`, `structure_confidence`
- lexical provenance features: `strong_source`, `morph_source`, `strong_confidence`, `morph_confidence`

Lexical provenance values currently include:

- `n1904_aligned`
- `n1904_projected_word_lemma_sp`

## Manual Metadata In `.tf` Headers

Header metadata is configurable in `config.yaml` under `tf_output`.

Three metadata layers are supported:

- dataset-level metadata from `project.*`, `tf_output.dataset_name`, `tf_output.version`, and `tf_output.language`
- shared feature metadata from `tf_output.global_feature_metadata`
- per-feature overrides from `tf_output.feature_metadata`

Example:

```yaml
tf_output:
  global_feature_metadata:
    corpus: "Textus Receptus"
    editor: "Your Name"
    license: "MIT"

  feature_metadata:
    unicode:
      description: "Greek surface form"
    strong_source:
      description: "Provenance of the Strong number"
    morph_confidence:
      description: "Confidence score for projected morphology"
```

This changes TF headers such as `@description`, but it does not create new data-bearing features by itself.

## Using The Dataset

```python
from tf.fabric import Fabric

TF = Fabric(locations="tf/1.0")
api = TF.load("unicode lemma sp strong morph strong_source")
F, T, L = api.F, api.T, api.L

print(T.sectionFromNode(1))

verse = T.nodeFromSection(("I_Corinthians", 1, 5))
words = L.d(verse, otype="w")

for w in words[:5]:
    print(
        F.unicode.v(w),
        F.lemma.v(w),
        F.sp.v(w),
        F.strong.v(w),
        F.strong_source.v(w),
    )
```

## License

MIT License. See [LICENSE](LICENSE) for details.
