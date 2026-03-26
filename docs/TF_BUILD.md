# TF Build And Metadata

This document explains how the final Text-Fabric dataset is generated, which upstream data it depends on, and which parts of the resulting `.tf` files can be configured manually.

## Build Inputs

The canonical dataset is built by [p4_04_generate_features.py](D:/Onedrive/GitHub/FORK_tr-text-fabric/scripts/phase4/p4_04_generate_features.py) with Text-Fabric's `tf.convert.walker.CV.walk()`.

The phase-4 export depends on these intermediate artifacts:

- `data/intermediate/tr_complete.parquet`
- `data/intermediate/tr_containers.parquet`
- `data/intermediate/tr_structure_nodes.parquet`

The output is written to:

- `tf/<version>/`

The version comes from `config.yaml` at `tf_output.version`.

## Upstream Data Flow

The current source and annotation flow is:

1. `byztxt/greektext-stephens` provides the authoritative plain-text Stephens token sequence.
2. `p1_04b_apply_punctuation_witness.py` can overlay accents and punctuation from the configured PDF witness.
3. `p2_01_extract_n1904.py` extracts lexical and syntactic features from the original N1904 TF source, including `strong` and `morph`.
4. `p2_06_transplant_syntax.py` copies those features onto directly aligned TR words.
5. `p3_04_parse_gaps.py` and `p3_05_convert_parses.py` fill the remaining syntax gaps locally with Stanza-based parsing.
6. `p4_01d_project_strong_morph.py` projects `strong` and `morph` onto some NLP-only words when N1904 has a unique `word+lemma+sp` match.
7. `p4_08*` scripts regenerate the structure layers.
8. `p4_04_generate_features.py` writes the final TF dataset.

## What The Builder Does

1. Load the intermediate parquet tables.
2. Sort the word table in canonical NT order.
3. Validate that canonical order still matches ascending `word_id`.
4. Create slot nodes in that canonical order so slot `1` is Matthew 1:1.
5. Emit `book`, `chapter`, and `verse` nodes while slots are being created.
6. Add explicit `clause`, `phrase`, and `wg` nodes from `tr_structure_nodes.parquet`.
7. Write scalar node features and the `parent` edge with `CV.walk()`.
8. Let Text-Fabric serialize the final `.tf` files and headers.

## Which `.tf` Files Are Generated

The builder writes standard TF feature files such as:

- `otype.tf`
- `oslots.tf`
- `otext.tf`
- scalar node feature files like `unicode.tf`, `lemma.tf`, `book.tf`, `chapter.tf`, `verse.tf`
- lexical provenance files like `strong_source.tf`, `morph_source.tf`, `strong_confidence.tf`, `morph_confidence.tf`
- structure feature files like `typ.tf`, `function.tf`, `rela.tf`, `rule.tf`, `clausetype.tf`
- edge feature files like `parent.tf`

The exact set depends on which columns are populated in the intermediate data.

## Where Feature Values Come From

The values inside the feature bodies come from the intermediate data, not from the config file.

Examples:

- `unicode`, `lemma`, `gloss`, `sp`, `strong`, `morph` come from `tr_complete.parquet`
- `strong_source`, `morph_source`, `strong_confidence`, `morph_confidence` come from the lexical projection/transplant pipeline in phase 4
- `book`, `chapter`, `verse` are assigned during canonical slot and section construction
- `typ`, `function`, `rela`, `rule`, `clausetype`, `structure_source`, `structure_confidence` come from `tr_structure_nodes.parquet`
- `parent` is written as a TF edge from the mapped dependency relation in `tr_complete.parquet`

If you want to change the actual annotation data, update the upstream pipeline outputs and rebuild. The metadata configuration only changes headers such as `@description`, not the word or structure values themselves.

## Lexical Provenance

`strong` and `morph` no longer come from the old BLB source path.

They now have two documented provenance values:

- `n1904_aligned`: copied directly from aligned N1904 words
- `n1904_projected_word_lemma_sp`: projected onto NLP-only words when the N1904 corpus has a unique `word+lemma+sp` signature

Confidence values are stored separately in:

- `strong_confidence`
- `morph_confidence`

## Manually Configurable Metadata

The generated `.tf` headers combine three metadata sources.

### 1. Dataset Metadata

These values come from:

- `project.description`
- `tf_output.dataset_name`
- `tf_output.version`
- `tf_output.language`

They feed dataset-level TF metadata such as:

- `@name`
- `@version`
- `@language`
- `@description`
- `@source`

### 2. Shared Per-Feature Metadata

Set `tf_output.global_feature_metadata` in `config.yaml`.

Every key-value pair in this dictionary is copied into every generated feature file as a header line.

```yaml
tf_output:
  global_feature_metadata:
    corpus: "Textus Receptus"
    editor: "Your Name"
    license: "MIT"
    repository: "https://github.com/your/repo"
```

### 3. Per-Feature Metadata

Set `tf_output.feature_metadata` in `config.yaml`.

This is keyed by TF feature name, and each nested dictionary is merged into that feature's header.

```yaml
tf_output:
  feature_metadata:
    unicode:
      description: "Greek surface form"
      uiName: "Unicode Text"
    strong_source:
      description: "Provenance of the Strong number"
    morph_confidence:
      description: "Confidence score for projected morphology"
```

This is the mechanism to override `@description` for an individual feature.

## Features You Can Target In `feature_metadata`

You can define per-feature metadata for any generated feature, including:

### Section And Edge Features

- `book`
- `chapter`
- `verse`
- `parent`

### Word Features

- `unicode`
- `lemma`
- `strong`
- `morph`
- `strong_source`
- `morph_source`
- `strong_confidence`
- `morph_confidence`
- `sp`
- `function`
- `role`
- `case`
- `gender`
- `number`
- `person`
- `tense`
- `voice`
- `mood`
- `gloss`
- `source`
- `translit`
- `lemmatranslit`
- `unaccent`
- `after`
- `ln`
- `bookshort`
- `text`
- `normalized`
- `num`
- `ref`
- `id`
- `cls`
- `trans`
- `domain`
- `typems`

### Structure Features

- `typ`
- `function`
- `rela`
- `clausetype`
- `rule`
- `structure_source`
- `structure_confidence`

If a feature is not emitted by the build, metadata for that feature will not create a new data layer by itself.

## Important Boundary

`config.yaml` can control:

- header metadata written into generated `.tf` files
- dataset naming and versioning
- shared annotations that should appear in every feature header

`config.yaml` does not control:

- adding a new data-bearing feature file by itself
- changing slot order
- changing words, lemmas, glosses, `strong`, `morph`, or structure spans directly

Those changes must happen upstream in the pipeline data.

## `otext` Metadata

The builder writes `otext.tf` with:

- `fmt:text-orig-full={unicode}{after}`
- `sectionTypes=book,chapter,verse`
- `sectionFeatures=book,chapter,verse`

Clause, phrase, and word-group nodes are present in the dataset, but they are not declared as TF `structureTypes` in `otext` because the current graph is not strictly nested enough for Text-Fabric's structure precompute.
