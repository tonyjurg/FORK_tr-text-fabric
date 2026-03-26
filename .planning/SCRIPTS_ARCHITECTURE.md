# Scripts Architecture

## Design Principles

1. Idempotent: each script can be run multiple times safely.
2. Configurable: paths and parameters come from `config.yaml`.
3. Logged: each script writes to `logs/`.
4. Checkpointed: intermediate outputs are saved under `data/`.
5. Testable: scripts should support `--dry-run` where practical.

---

## Directory Structure

```text
D:/Onedrive/GitHub/FORK_tr-text-fabric/
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
|   |-- intermediate/
|   `-- output/
|-- tf/
|-- reports/
`-- logs/
```

---

## Phase Overview

### Phase 1

- `p1_04_acquire_tr.py`: acquire the public-domain Stephens source
- `p1_04b_apply_punctuation_witness.py`: optional PDF-based punctuation/accent overlay
- `p1_05_build_tr_dataframe.py`: build `tr_words.parquet`

### Phase 2

- `p2_01_extract_n1904.py`: extract N1904 lexical and syntactic features from the original TF source, including `strong` and `morph`
- `p2_02` to `p2_05`: align Stephens to N1904 and build ID mappings
- `p2_06_transplant_syntax.py`: transplant aligned annotations into TR rows

### Phase 3

- `p3_01` to `p3_05`: analyze gaps, parse them locally with Stanza, and convert them to the project's feature surface
- `p3_06_review_variants.py`: optional review step; not required for the current no-Claude rebuild path

### Phase 4

- `p4_01_merge_data.py`: merge aligned and NLP-generated annotations
- `p4_01b_fill_glosses.py`: achieve full gloss coverage
- `p4_01c_fix_nlp_errors.py`: correct systematic NLP lemma/POS issues
- `p4_01d_project_strong_morph.py`: project `strong` and `morph` onto some NLP-only words by unique `word+lemma+sp` matching against N1904
- `p4_02_generate_containers.py`: regenerate canonical `book/chapter/verse` containers
- `p4_08a` to `p4_08h`: rebuild structure layers
- `p4_04_generate_features.py`: build the final TF dataset via `CV.walk()`
- `p4_07_verify_build.py`: verify canonical order and dataset integrity

### Phase 5

- QA and verification scripts over the generated TF corpus

---

## Script Dependency Graph

```text
Phase 1:
p1_01 -> p1_02 -> p1_03
              \
p1_04 -> p1_04b -> p1_05

Phase 2:
p1_05 + p1_02 -> p2_01 -> p2_02 -> p2_03 -> p2_04 -> p2_05 -> p2_06

Phase 3:
p2_04 -> p3_01 -> p3_02 -> p3_03 -> p3_04 -> p3_05 -> p3_06(optional)

Phase 4:
p2_06 + p3_05 -> p4_01 -> p4_01b -> p4_01c -> p4_01d -> p4_02 -> p4_03 -> p4_08a -> p4_08b -> p4_08c -> p4_08d -> p4_08e -> p4_08h -> p4_04 -> p4_07 -> p4_08g

Phase 5:
p4_07 -> p5_01 -> p5_02 -> p5_03 -> p5_04 -> p5_05 -> p5_06 -> p5_07 -> p5_08
```

---

## Key Checkpoints

| Script | Output | Format |
|--------|--------|--------|
| `p1_04` | `data/source/tr_source.csv` | CSV |
| `p1_04b` | `data/source/tr_source_prepared.csv` | CSV |
| `p1_05` | `data/intermediate/tr_words.parquet` | Parquet |
| `p2_01` | `data/intermediate/n1904_words.parquet` | Parquet |
| `p2_04` | `data/intermediate/alignment_map.parquet` | Parquet |
| `p2_04` | `data/intermediate/gaps.csv` | CSV |
| `p2_05` | `data/intermediate/id_translation.parquet` | Parquet |
| `p2_06` | `data/intermediate/tr_transplanted.parquet` | Parquet |
| `p3_05` | `data/intermediate/gap_syntax.parquet` | Parquet |
| `p4_01` | `data/intermediate/tr_complete.parquet` | Parquet |
| `p4_01d` | `data/intermediate/tr_complete.parquet` | Parquet |
| `p4_02` | `data/intermediate/tr_containers.parquet` | Parquet |
| `p4_08e` | `data/intermediate/tr_structure_nodes.parquet` | Parquet |
| `p4_04` | `tf/<version>/` | TF dataset |

---

## Notes On The Current Build

- The authoritative source base is no longer BLB; it is the public-domain Stephens text.
- The PDF witness is an overlay source only.
- `strong` and `morph` now come from the original N1904 TF data where aligned, with an additional conservative projection pass for some NLP-only words.
- The final TF dataset is built only through `tf.convert.walker.CV.walk()`; the older handwritten TF writers are legacy/debug-only.
- The no-Claude rebuild path is fully supported as long as local Stanza resources are available.
