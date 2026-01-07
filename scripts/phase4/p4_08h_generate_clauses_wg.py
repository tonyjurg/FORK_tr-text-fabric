#!/usr/bin/env python3
"""
Phase 4 Step 8h: Generate Clauses and Word Groups

Generates clause and word group nodes for inferred/unknown verses that
currently only have phrase nodes.

Uses N1904-compatible patterns:
- Clause boundaries: conjunctions, subordinators, punctuation
- Word groups: DetNP, PrepNp, NPofNP, AdjpNp patterns
"""

import pandas as pd
import json
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils.logging import ScriptLogger
from scripts.utils.config import load_config


# Subordinating conjunctions that start subordinate clauses
SUBORDINATORS = {
    'ὅτι', 'ἵνα', 'ὅταν', 'ὅπως', 'ὥστε', 'ἐπεί', 'ἐπειδή',
    'πρίν', 'ἕως', 'ἄχρι', 'μέχρι', 'ὅτε', 'ὁπότε', 'ἡνίκα',
    'ὡς', 'καθώς', 'ὥσπερ', 'καθάπερ', 'διότι', 'ὁπόταν'
}

# Conditional particles
CONDITIONALS = {'εἰ', 'ἐάν', 'ἄν', 'εἴπερ', 'εἴτε', 'κἄν'}

# Relative pronouns (start relative clauses)
RELATIVES = {'ὅς', 'ἥ', 'ὅ', 'ὅστις', 'ἥτις', 'ὅ,τι', 'ὃς', 'ᾧ', 'ἧς', 'οὗ', 'ἅ', 'αἵ', 'οἵ'}

# Coordinating conjunctions (can start new main clause after a complete clause)
COORDINATORS = {'καί', 'δέ', 'ἀλλά', 'γάρ', 'οὖν', 'μέν', 'τε', 'οὐδέ', 'μηδέ', 'ἤ'}


def detect_clause_boundaries(verse_words: pd.DataFrame) -> list:
    """
    Detect clause boundaries within a verse.

    Returns list of clauses, each with:
    - word_ids: list of word IDs in the clause
    - slots: list of slot positions
    - clausetype: inferred type (main, relative, purpose, conditional, etc.)
    - confidence: confidence score
    """
    if verse_words.empty:
        return []

    verse_words = verse_words.sort_values('word_rank').reset_index(drop=True)
    clauses = []
    current_clause_words = []
    current_clause_start_lemma = None
    clause_type = 'main'

    for idx, row in verse_words.iterrows():
        lemma = row.get('lemma', '')
        sp = row.get('sp', '')
        word_id = row.get('word_id')
        after = row.get('after', '')  # Trailing punctuation

        # Check for clause boundary signals
        is_boundary = False
        new_clause_type = None

        # Subordinator starts new subordinate clause
        if lemma in SUBORDINATORS:
            is_boundary = True
            if lemma in ('ἵνα', 'ὅπως'):
                new_clause_type = 'purpose'
            elif lemma in ('ὅτι', 'διότι'):
                new_clause_type = 'content'
            elif lemma in ('ὅταν', 'ὅτε', 'ἕως', 'πρίν', 'ἄχρι', 'μέχρι'):
                new_clause_type = 'temporal'
            elif lemma in ('ὡς', 'καθώς', 'ὥσπερ', 'καθάπερ'):
                new_clause_type = 'comparative'
            elif lemma in ('ὥστε',):
                new_clause_type = 'result'
            else:
                new_clause_type = 'subordinate'

        # Conditional starts conditional clause
        elif lemma in CONDITIONALS:
            is_boundary = True
            new_clause_type = 'conditional'

        # Relative pronoun starts relative clause
        elif lemma in RELATIVES or (sp == 'pron' and row.get('morph', '').startswith('R')):
            is_boundary = True
            new_clause_type = 'relative'

        # Major punctuation can indicate clause boundary
        # (but coordinate conjunctions after punctuation start new main clause)
        elif '.' in after or ';' in after or '·' in after:
            # The clause ends here, next word starts new clause
            current_clause_words.append({
                'word_id': word_id,
                'slot': idx + 1  # Will be corrected later with actual slots
            })
            if current_clause_words:
                clauses.append({
                    'words': current_clause_words,
                    'clausetype': clause_type,
                    'confidence': 0.85
                })
            current_clause_words = []
            clause_type = 'main'  # Next clause defaults to main
            continue

        # Coordinator after complete clause can start new main clause
        # But only if the current clause has a verb
        elif lemma in COORDINATORS and idx > 0:
            # Check if current clause has a verb
            has_verb = any(w.get('sp') == 'verb' for w in current_clause_words)
            if has_verb and len(current_clause_words) >= 2:
                is_boundary = True
                new_clause_type = 'coordinate'

        if is_boundary and current_clause_words:
            # Save current clause
            clauses.append({
                'words': current_clause_words,
                'clausetype': clause_type,
                'confidence': 0.85
            })
            current_clause_words = []
            clause_type = new_clause_type or 'main'

        # Add word to current clause
        current_clause_words.append({
            'word_id': word_id,
            'slot': idx + 1,  # Temporary - will be corrected
            'sp': sp,
            'lemma': lemma
        })

    # Save final clause
    if current_clause_words:
        clauses.append({
            'words': current_clause_words,
            'clausetype': clause_type,
            'confidence': 0.85
        })

    return clauses


def detect_word_groups(phrase_words: list, all_word_data: dict) -> list:
    """
    Detect word group patterns within phrase words.

    Patterns detected (N1904-compatible rule names):
    - DetNP: article + (adj)* + noun
    - PrepNp: preposition + NP
    - NPofNP: noun + genitive noun
    - AdjpNp: adjective + noun
    - NpAdjp: noun + adjective

    Returns list of word groups with rule assignments.
    """
    if not phrase_words:
        return []

    wgs = []
    i = 0

    while i < len(phrase_words):
        word_info = phrase_words[i]
        word_id = word_info['word_id']
        slot = word_info['slot']
        word_data = all_word_data.get(word_id, {})
        sp = word_data.get('sp', '')
        case = word_data.get('case', '')

        matched = False

        # DetNP: article + (modifiers)* + head noun
        if sp == 'art':
            wg_words = [word_info]
            j = i + 1
            head_found = False

            while j < len(phrase_words):
                next_info = phrase_words[j]
                next_data = all_word_data.get(next_info['word_id'], {})
                next_sp = next_data.get('sp', '')

                if next_sp in ('adjv', 'pron', 'num'):
                    wg_words.append(next_info)
                    j += 1
                elif next_sp in ('subs', 'nmpr'):
                    wg_words.append(next_info)
                    head_found = True
                    j += 1
                    break
                else:
                    break

            if len(wg_words) > 1 and head_found:
                wgs.append({
                    'words': wg_words,
                    'rule': 'DetNP',
                    'confidence': 0.90
                })
                i = j
                matched = True

        # PrepNp: preposition + NP
        if not matched and sp == 'prep':
            wg_words = [word_info]
            j = i + 1

            # Collect following nominals
            while j < len(phrase_words):
                next_info = phrase_words[j]
                next_data = all_word_data.get(next_info['word_id'], {})
                next_sp = next_data.get('sp', '')

                if next_sp in ('art', 'adjv', 'subs', 'nmpr', 'pron', 'num'):
                    wg_words.append(next_info)
                    j += 1
                else:
                    break

            if len(wg_words) > 1:
                wgs.append({
                    'words': wg_words,
                    'rule': 'PrepNp',
                    'confidence': 0.85
                })
                i = j
                matched = True

        # NPofNP: noun + genitive noun
        if not matched and sp in ('subs', 'nmpr'):
            if i + 1 < len(phrase_words):
                next_info = phrase_words[i + 1]
                next_data = all_word_data.get(next_info['word_id'], {})
                next_sp = next_data.get('sp', '')
                next_case = next_data.get('case', '')

                # Check for genitive article or noun
                if next_sp == 'art' and next_case == 'genitive':
                    # Article + genitive nominal
                    wg_words = [word_info, next_info]
                    j = i + 2
                    while j < len(phrase_words):
                        check_info = phrase_words[j]
                        check_data = all_word_data.get(check_info['word_id'], {})
                        check_sp = check_data.get('sp', '')
                        check_case = check_data.get('case', '')
                        if check_sp in ('subs', 'nmpr', 'adjv') and check_case == 'genitive':
                            wg_words.append(check_info)
                            j += 1
                        else:
                            break
                    if len(wg_words) >= 3:  # noun + art + gen_noun
                        wgs.append({
                            'words': wg_words,
                            'rule': 'NPofNP',
                            'confidence': 0.85
                        })
                        i = j
                        matched = True

                elif next_sp in ('subs', 'nmpr') and next_case == 'genitive':
                    # Direct genitive noun (no article)
                    wgs.append({
                        'words': [word_info, next_info],
                        'rule': 'NPofNP',
                        'confidence': 0.80
                    })
                    i += 2
                    matched = True

        # AdjpNp: adjective + noun
        if not matched and sp == 'adjv':
            if i + 1 < len(phrase_words):
                next_info = phrase_words[i + 1]
                next_data = all_word_data.get(next_info['word_id'], {})
                next_sp = next_data.get('sp', '')
                if next_sp in ('subs', 'nmpr'):
                    wgs.append({
                        'words': [word_info, next_info],
                        'rule': 'AdjpNp',
                        'confidence': 0.85
                    })
                    i += 2
                    matched = True

        # NpAdjp: noun + adjective
        if not matched and sp in ('subs', 'nmpr'):
            if i + 1 < len(phrase_words):
                next_info = phrase_words[i + 1]
                next_data = all_word_data.get(next_info['word_id'], {})
                next_sp = next_data.get('sp', '')
                if next_sp == 'adjv':
                    # Check case agreement
                    next_case = next_data.get('case', '')
                    if case == next_case or not case or not next_case:
                        wgs.append({
                            'words': [word_info, next_info],
                            'rule': 'NpAdjp',
                            'confidence': 0.80
                        })
                        i += 2
                        matched = True

        if not matched:
            i += 1

    return wgs


def generate_clauses_and_wgs(nodes_df: pd.DataFrame, complete_df: pd.DataFrame,
                              logger) -> pd.DataFrame:
    """
    Generate clause and word group nodes for verses without them.

    Args:
        nodes_df: Existing structure nodes
        complete_df: Complete word data
        logger: Logger instance

    Returns:
        Updated nodes DataFrame with new clause and wg nodes
    """
    # Build word_id to slot mapping
    complete_df = complete_df.sort_values(
        ['book', 'chapter', 'verse', 'word_rank']
    ).reset_index(drop=True)
    word_to_slot = {row['word_id']: idx + 1 for idx, row in complete_df.iterrows()}

    # Build word data lookup
    word_data_lookup = {}
    for _, row in complete_df.iterrows():
        word_data_lookup[row['word_id']] = {
            'sp': row.get('sp', ''),
            'lemma': row.get('lemma', ''),
            'case': row.get('case', ''),
            'morph': row.get('morph', ''),
            'after': row.get('after', '')
        }

    # Find verses that need clauses/wgs generated
    # Group nodes by verse
    verse_nodes = defaultdict(list)
    for _, row in nodes_df.iterrows():
        key = (row['book'], row['chapter'], row['verse'])
        verse_nodes[key].append(row)

    # Find verses with phrases but no clauses or wgs (inferred/unknown_only source)
    verses_needing_clauses = []
    verses_needing_wgs = []

    for verse_key, nodes in verse_nodes.items():
        sources = set(n['source'] for n in nodes)
        otypes = set(n['otype'] for n in nodes)

        # Only process non-direct verses
        if 'direct' not in sources:
            if 'clause' not in otypes:
                verses_needing_clauses.append(verse_key)
            if 'wg' not in otypes and 'phrase' in otypes:
                verses_needing_wgs.append(verse_key)

    logger.info(f"Verses needing clauses: {len(verses_needing_clauses)}")
    logger.info(f"Verses needing word groups: {len(verses_needing_wgs)}")

    # Get next node ID
    next_id = nodes_df['node_id'].max() + 1

    new_clauses = []
    new_wgs = []

    # Generate clauses
    for book, chapter, verse in verses_needing_clauses:
        verse_words = complete_df[
            (complete_df['book'] == book) &
            (complete_df['chapter'] == chapter) &
            (complete_df['verse'] == verse)
        ]

        if verse_words.empty:
            continue

        # Detect clause boundaries
        clauses = detect_clause_boundaries(verse_words)

        for clause in clauses:
            if not clause['words']:
                continue

            # Get actual slots
            word_ids = [w['word_id'] for w in clause['words']]
            slots = [word_to_slot.get(wid) for wid in word_ids if wid in word_to_slot]
            if not slots:
                continue

            # Determine source from verse's existing nodes
            verse_source = 'generated'
            existing_nodes = verse_nodes.get((book, chapter, verse), [])
            if existing_nodes:
                verse_source = existing_nodes[0]['source']

            new_clauses.append({
                'node_id': next_id,
                'otype': 'clause',
                'book': book,
                'chapter': chapter,
                'verse': verse,
                'first_slot': min(slots),
                'last_slot': max(slots),
                'typ': None,
                'clausetype': clause['clausetype'],
                'cltype': None,
                'function': None,
                'rela': None,
                'rule': None,
                'n1904_node_id': None,
                'source': verse_source,
                'confidence': clause['confidence']
            })
            next_id += 1

    logger.info(f"Generated {len(new_clauses)} clause nodes")

    # Generate word groups from phrases
    for book, chapter, verse in verses_needing_wgs:
        # Get phrases for this verse
        verse_phrase_nodes = [
            n for n in verse_nodes.get((book, chapter, verse), [])
            if n['otype'] == 'phrase'
        ]

        for phrase_node in verse_phrase_nodes:
            first_slot = phrase_node['first_slot']
            last_slot = phrase_node['last_slot']

            # Get words in this phrase
            phrase_words = complete_df[
                (complete_df['book'] == book) &
                (complete_df['chapter'] == chapter) &
                (complete_df['verse'] == verse)
            ]
            phrase_words = phrase_words[
                phrase_words.apply(
                    lambda r: first_slot <= word_to_slot.get(r['word_id'], 0) <= last_slot,
                    axis=1
                )
            ]

            if phrase_words.empty:
                continue

            # Build word info list for pattern detection
            word_list = []
            for _, row in phrase_words.sort_values('word_rank').iterrows():
                word_list.append({
                    'word_id': row['word_id'],
                    'slot': word_to_slot.get(row['word_id'])
                })

            # Detect word group patterns
            wgs = detect_word_groups(word_list, word_data_lookup)

            for wg in wgs:
                if not wg['words']:
                    continue

                wg_slots = [w['slot'] for w in wg['words'] if w['slot']]
                if not wg_slots:
                    continue

                new_wgs.append({
                    'node_id': next_id,
                    'otype': 'wg',
                    'book': book,
                    'chapter': chapter,
                    'verse': verse,
                    'first_slot': min(wg_slots),
                    'last_slot': max(wg_slots),
                    'typ': None,
                    'clausetype': None,
                    'cltype': None,
                    'function': None,
                    'rela': None,
                    'rule': wg['rule'],
                    'n1904_node_id': None,
                    'source': phrase_node['source'],
                    'confidence': wg['confidence']
                })
                next_id += 1

    logger.info(f"Generated {len(new_wgs)} word group nodes")

    # Combine all nodes
    if new_clauses or new_wgs:
        new_nodes_df = pd.DataFrame(new_clauses + new_wgs)
        # Ensure same columns
        for col in nodes_df.columns:
            if col not in new_nodes_df.columns:
                new_nodes_df[col] = None
        new_nodes_df = new_nodes_df[nodes_df.columns]
        result_df = pd.concat([nodes_df, new_nodes_df], ignore_index=True)
    else:
        result_df = nodes_df

    return result_df


def main(config=None):
    """Main entry point."""
    with ScriptLogger('p4_08h_generate_clauses_wg', config) as logger:
        # Load configuration
        if config is None:
            config = load_config()

        intermediate = Path('data/intermediate')

        # Load existing structure nodes
        nodes_path = intermediate / 'tr_structure_nodes.parquet'
        if not nodes_path.exists():
            logger.error(f"Structure nodes not found: {nodes_path}")
            return

        nodes_df = pd.read_parquet(nodes_path)
        logger.info(f"Loaded {len(nodes_df)} existing structure nodes")

        # Log current counts
        otype_counts = nodes_df['otype'].value_counts()
        logger.info(f"Current node counts:")
        for otype, count in otype_counts.items():
            logger.info(f"  {otype}: {count:,}")

        # Load complete word data
        complete_path = intermediate / 'tr_complete.parquet'
        if not complete_path.exists():
            logger.error(f"Complete data not found: {complete_path}")
            return

        complete_df = pd.read_parquet(complete_path)
        logger.info(f"Loaded {len(complete_df)} words")

        # Generate clauses and word groups
        updated_nodes = generate_clauses_and_wgs(nodes_df, complete_df, logger)

        # Log updated counts
        new_otype_counts = updated_nodes['otype'].value_counts()
        logger.info(f"Updated node counts:")
        for otype, count in new_otype_counts.items():
            diff = count - otype_counts.get(otype, 0)
            if diff > 0:
                logger.info(f"  {otype}: {count:,} (+{diff:,})")
            else:
                logger.info(f"  {otype}: {count:,}")

        # Save updated nodes
        updated_nodes.to_parquet(nodes_path)
        logger.info(f"Saved updated structure nodes to {nodes_path}")

        # Save summary for README
        summary = {
            'clauses_before': int(otype_counts.get('clause', 0)),
            'clauses_after': int(new_otype_counts.get('clause', 0)),
            'wgs_before': int(otype_counts.get('wg', 0)),
            'wgs_after': int(new_otype_counts.get('wg', 0)),
            'phrases': int(new_otype_counts.get('phrase', 0))
        }

        summary_path = intermediate / 'clause_wg_generation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved generation summary to {summary_path}")


if __name__ == '__main__':
    main()
