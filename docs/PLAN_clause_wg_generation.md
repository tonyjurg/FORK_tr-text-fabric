# Plan: Generate Clauses and Word Groups for All Verses

## Current State

| Node Type | Direct (23%) | Inferred/Unknown (77%) |
|-----------|--------------|------------------------|
| clause | 7,964 ✅ | 0 ❌ |
| phrase | 13,428 ✅ | 53,929 ✅ |
| wg | 19,520 ✅ | 0 ❌ |

## Goal

Generate clause and word group nodes for the 77% of verses that currently only have phrases.

---

## N1904 Structure Analysis

### Word Group Statistics (N1904)

N1904 has **106,868 word groups** with the following `rule` patterns:

| Rule | Count | Description |
|------|-------|-------------|
| DetNP | 15,696 | Determiner + Noun Phrase (ὁ λόγος) |
| PrepNp | 11,044 | Preposition + NP (ἐν τῷ κόσμῳ) |
| NPofNP | 6,819 | Genitive chain (υἱὸς τοῦ θεοῦ) |
| AdjpNp | 4,127 | Adjective + NP (ἀγαθὸς ἄνθρωπος) |
| NpAdjp | 3,891 | NP + Adjective (ἄνθρωπος ἀγαθός) |
| NpPp | 3,654 | NP + PP (θεὸς ἐν οὐρανῷ) |
| NpaNp | 2,876 | NP and NP coordination |
| sub-CL | 2,341 | Subordinate clause |
| Conj-CL | 1,987 | Conjunction + clause |
| RelCL | 1,654 | Relative clause |
| S-V-O | 506 | Subject-Verb-Object |
| P-VC-S | 423 | Predicate-VerbCopula-Subject |
| Other | ~52,000 | Various patterns |

### Hierarchical Nesting

N1904 word groups are hierarchically nested:

```
Verse: John 1:1
└── Conj3CL (ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος)
    ├── P-VC-S (ἐν ἀρχῇ ἦν ὁ λόγος)
    │   ├── PrepNp (ἐν ἀρχῇ)
    │   │   └── NP (ἀρχῇ)
    │   ├── VP (ἦν)
    │   └── DetNP (ὁ λόγος)
    ├── Conj-CL (καὶ ὁ λόγος ἦν πρὸς τὸν θεόν)
    │   └── ...
    └── Conj-CL (καὶ θεὸς ἦν ὁ λόγος)
        └── ...
```

### Key Insight

N1904's structure is **syntactically annotated by human scholars**. Full replication would require:
- Dependency parsing with Greek-specific training
- Manual annotation of ~100k word group relationships
- Complex hierarchical tree building

**Realistic approach**: Generate **flat word groups** with basic rules, not full hierarchical trees.

---

## Revised Phase 1: Clause Boundary Detection

### 1.1 Clause Boundary Signals

| Signal | Example | Confidence |
|--------|---------|------------|
| Coordinating conjunction at sentence start | καί, δέ, ἀλλά, γάρ, οὖν | 90% |
| Subordinating conjunction | ὅτι, ἵνα, ὅταν, εἰ, ἐάν | 95% |
| Relative pronoun | ὅς, ἥ, ὅ, ὅστις | 90% |
| Finite verb boundary (mood change) | indicative → subjunctive | 85% |
| Major punctuation | . ; · | 95% |

### 1.2 Clause Type Inference

| Pattern | Clause Type | N1904 Equivalent |
|---------|-------------|------------------|
| ὅτι + indicative | content | Cmpl (content clause) |
| ἵνα + subjunctive | purpose | Purp (purpose clause) |
| εἰ/ἐάν + verb | conditional | Cond (conditional) |
| ὅταν + subjunctive | temporal | Time (temporal) |
| ὅς/ἥ/ὅ + verb | relative | Attr (attributive) |
| Main clause | primary | Main |

### 1.3 Algorithm

```python
def detect_clauses(verse_words):
    clauses = []
    current_start = 0

    SUBORDINATORS = {'ὅτι', 'ἵνα', 'ὅταν', 'ὅπως', 'ὥστε', 'εἰ', 'ἐάν', 'ἐπεί'}
    RELATIVES = {'ὅς', 'ἥ', 'ὅ', 'ὅστις', 'ἥτις', 'ὅ,τι'}

    for i, word in enumerate(verse_words):
        lemma = word['lemma']

        # Check for clause boundary
        is_boundary = False
        clause_type = None

        if lemma in SUBORDINATORS:
            is_boundary = True
            clause_type = infer_type_from_subordinator(lemma, verse_words[i:])
        elif lemma in RELATIVES:
            is_boundary = True
            clause_type = 'relative'
        elif word['sp'] == 'conj' and i > 0 and is_finite_verb_before(verse_words[:i]):
            is_boundary = True
            clause_type = 'coordinate'

        if is_boundary and i > current_start:
            clauses.append({
                'words': verse_words[current_start:i],
                'type': 'main' if not clauses else 'unknown',
                'confidence': 0.85
            })
            current_start = i

    # Final clause
    if current_start < len(verse_words):
        clauses.append({
            'words': verse_words[current_start:],
            'type': clause_type or 'main',
            'confidence': 0.85
        })

    return clauses
```

---

## Revised Phase 2: Word Group Generation

### 2.1 N1904-Compatible Rule Patterns

Generate word groups with `rule` values matching N1904:

| Rule | Pattern | Detection Method |
|------|---------|------------------|
| DetNP | art + subs/adjv | Article followed by nominal |
| PrepNp | prep + NP | Preposition followed by NP |
| NPofNP | NP + gen NP | Noun followed by genitive |
| AdjpNp | adjv + subs | Adjective before noun |
| NpAdjp | subs + adjv | Noun followed by adjective |
| NpaNp | NP + conj + NP | Coordinated NPs |

### 2.2 Algorithm

```python
def generate_word_groups(phrase_words):
    wgs = []
    i = 0

    while i < len(phrase_words):
        word = phrase_words[i]
        sp = word['sp']

        # DetNP: article + nominal(s)
        if sp == 'art':
            wg_words, rule = match_det_np(phrase_words, i)
            if wg_words:
                wgs.append({'words': wg_words, 'rule': 'DetNP', 'confidence': 0.90})
                i += len(wg_words)
                continue

        # PrepNp: preposition + NP
        if sp == 'prep':
            wg_words, rule = match_prep_np(phrase_words, i)
            if wg_words:
                wgs.append({'words': wg_words, 'rule': 'PrepNp', 'confidence': 0.90})
                i += len(wg_words)
                continue

        # NPofNP: noun + genitive
        if sp in ('subs', 'nmpr'):
            wg_words = match_genitive_chain(phrase_words, i)
            if len(wg_words) > 1:
                wgs.append({'words': wg_words, 'rule': 'NPofNP', 'confidence': 0.85})
                i += len(wg_words)
                continue

        # AdjpNp: adjective + noun
        if sp == 'adjv' and i + 1 < len(phrase_words):
            next_sp = phrase_words[i + 1]['sp']
            if next_sp in ('subs', 'nmpr'):
                wgs.append({
                    'words': phrase_words[i:i+2],
                    'rule': 'AdjpNp',
                    'confidence': 0.85
                })
                i += 2
                continue

        i += 1

    return wgs

def match_det_np(words, start):
    """Match article + modifiers + head noun."""
    if words[start]['sp'] != 'art':
        return None, None

    end = start + 1
    while end < len(words):
        sp = words[end]['sp']
        if sp in ('adjv', 'pron'):
            end += 1
        elif sp in ('subs', 'nmpr'):
            end += 1
            break  # Head noun found
        else:
            break

    if end > start + 1:
        return words[start:end], 'DetNP'
    return None, None
```

---

## Phase 3: Phrase Relations (rela)

### 3.1 Simplified Relation Inference

Since full syntactic parsing is beyond scope, infer basic relations:

| Context | Relation | Confidence |
|---------|----------|------------|
| Genitive NP after noun | Attr | 85% |
| PP following verb | Cmpl | 80% |
| Nominative NP with verb | Subj | 75% |
| Accusative NP after verb | Objc | 75% |
| Adjacent same-case NPs | Appo | 70% |

### 3.2 Algorithm

```python
def infer_phrase_relations(clause_phrases):
    verb_idx = find_main_verb_index(clause_phrases)

    for i, phrase in enumerate(clause_phrases):
        case = get_phrase_case(phrase)
        ptype = phrase['typ']

        # Genitive after nominal = Attr
        if case == 'genitive' and i > 0:
            prev_type = clause_phrases[i-1]['typ']
            if prev_type == 'NP':
                phrase['rela'] = 'Attr'
                phrase['rela_confidence'] = 0.85
                continue

        # PP after verb = Cmpl
        if ptype == 'PP' and verb_idx is not None and i > verb_idx:
            phrase['rela'] = 'Cmpl'
            phrase['rela_confidence'] = 0.80
            continue

        # Nominative before/after verb = Subj
        if case == 'nominative' and ptype == 'NP':
            phrase['rela'] = 'Subj'
            phrase['rela_confidence'] = 0.75
            continue

        # Accusative after verb = Objc
        if case == 'accusative' and ptype == 'NP' and verb_idx is not None:
            if i > verb_idx:
                phrase['rela'] = 'Objc'
                phrase['rela_confidence'] = 0.75
                continue
```

---

## Implementation Plan

### New Script: `p4_08h_generate_clauses_wg.py`

```
Phase 4 Step 8h: Generate Clauses and Word Groups

Generates clause and word group nodes for inferred/unknown verses:
1. Detect clause boundaries from conjunctions/subordinators
2. Infer clause types (relative, purpose, conditional, etc.)
3. Generate word groups with N1904-compatible rules
4. Infer basic phrase relations from syntactic context
```

### Pipeline Integration

Add to `run_pipeline.py` after step 14 (Generate Structure TF):

```python
ScriptInfo(
    phase=4, step=15,  # Renumber existing 15-16 to 16-17
    module="scripts.phase4.p4_08h_generate_clauses_wg",
    name="Generate Clauses & WG",
    description="Generate clause and word group nodes for non-direct verses",
    inputs=["data/intermediate/tr_structure_nodes.parquet"],
    outputs=["data/intermediate/tr_structure_nodes.parquet"]  # Updates in place
)
```

### Expected Output

| Node Type | Before | After | Notes |
|-----------|--------|-------|-------|
| clause | 7,964 | ~15,000 | One clause per major sentence boundary |
| phrase | 67,357 | 67,357 | Unchanged |
| wg | 19,520 | ~60,000 | DetNP, PrepNp, NPofNP, AdjpNp patterns |

### Updated Limitations Table

| Feature | Direct (23%) | Generated (77%) | Notes |
|---------|--------------|-----------------|-------|
| Clause boundaries | ✅ From N1904 | ✅ Generated | Punctuation/conjunction-based |
| Clause types | ✅ Full N1904 | ⚠️ Basic types | 5-6 types vs N1904's ~20 |
| Word group rules | ✅ Full hierarchy | ⚠️ Flat patterns | DetNP, PrepNp, NPofNP only |
| Phrase relations | ✅ From N1904 | ⚠️ Inferred | 70-85% confidence |
| Nested hierarchy | ✅ Full tree | ❌ Flat | No hierarchical nesting |

---

## Verification

### Test Cases

1. **John 1:1** (direct) - Should remain unchanged
2. **Acts 8:37** (inferred) - Should now have clauses and word groups
3. **1 John 5:7** (unknown) - Should now have clauses and word groups

### Accuracy Metrics

- Clause count per verse: Compare distribution to N1904
- Word group coverage: % of words in word groups (target: >70%)
- Rule distribution: Should roughly match N1904 proportions

### Validation Queries

```python
# Check clause count distribution
clause_per_verse = nodes[nodes['node_type'] == 'clause'].groupby('verse').size()
print(f"Avg clauses per verse: {clause_per_verse.mean():.1f}")
# N1904 average: ~1.5 clauses per verse

# Check WG rule distribution
wg_rules = nodes[nodes['node_type'] == 'wg']['rule'].value_counts()
print(wg_rules.head(10))
# Should see DetNP, PrepNp, NPofNP as top patterns
```

---

## Limitations vs N1904

### What This Script Generates
- Basic clause boundaries from punctuation/conjunctions
- Common word group patterns (DetNP, PrepNp, NPofNP)
- Simple phrase relations (Attr, Subj, Objc, Cmpl)

### What Requires Human Annotation
- Full hierarchical nesting (N1904 has 3-5 levels deep)
- Complex clause relationships (embedded clauses)
- Semantic role labeling
- Discourse-level structure
- Rare syntactic patterns

### Honest Assessment

The generated structure will be **functional but simpler** than N1904:
- Good enough for basic navigation (find clauses, group related words)
- Not suitable for deep syntactic analysis
- Confidence scores indicate reliability

---

## Implementation Steps

1. **Create `p4_08h_generate_clauses_wg.py`**
   - Load inferred/unknown verses from structure data
   - Implement clause boundary detection
   - Implement word group pattern matching
   - Implement phrase relation inference
   - Save updated structure nodes

2. **Integrate into pipeline**
   - Add ScriptInfo to run_pipeline.py
   - Update step numbering

3. **Test and verify**
   - Run on sample verses
   - Compare to N1904 patterns
   - Validate accuracy metrics

4. **Update documentation**
   - Update README limitations table
   - Update node counts in README
