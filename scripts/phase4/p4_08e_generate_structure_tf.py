#!/usr/bin/env python3
"""
Phase 4 Step 8e: Generate Structure TF Files

Combines structure data from:
- Direct transplant (100% aligned verses)
- Inferred structure (known words, different positions)
- Unknown word resolutions

Generates TF files for clause, phrase, and word group nodes.
"""

import pandas as pd
import json
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils.logging import ScriptLogger
from scripts.utils.config import load_config


def load_structure_data() -> tuple:
    """Load all structure data from previous phases."""
    intermediate = Path('data/intermediate')

    # Direct transplant (100% aligned)
    with open(intermediate / 'tr_structure_direct.json', 'r') as f:
        direct = json.load(f)

    # Inferred structure
    with open(intermediate / 'tr_structure_inferred.json', 'r') as f:
        inferred = json.load(f)

    # Unknown word resolutions
    with open(intermediate / 'unknown_word_resolutions.json', 'r') as f:
        unknown_resolutions = json.load(f)

    # Word classification data
    classified = pd.read_parquet(intermediate / 'tr_structure_classified.parquet')

    return direct, inferred, unknown_resolutions, classified


def build_verse_key(book: str, chapter: int, verse: int) -> str:
    """Create a consistent verse key."""
    return f"{book}:{chapter}:{verse}"


def merge_structures(direct: list, inferred: list, unknown_res: list,
                     classified: pd.DataFrame) -> dict:
    """
    Merge all structure sources into unified verse structures.

    Returns dict of verse_key -> structure
    """
    merged = {}

    # Index direct transplant by verse
    for item in direct:
        key = build_verse_key(item['book'], item['chapter'], item['verse'])
        merged[key] = {
            'book': item['book'],
            'chapter': item['chapter'],
            'verse': item['verse'],
            'clauses': item.get('clauses', []),
            'phrases': item.get('phrases', []),
            'wgs': item.get('wgs', []),
            'subphrases': item.get('subphrases', []),
            'source': 'direct',
            'confidence': 1.0
        }

    # Add inferred structures
    for item in inferred:
        key = build_verse_key(item['book'], item['chapter'], item['verse'])
        if key not in merged:
            merged[key] = {
                'book': item['book'],
                'chapter': item['chapter'],
                'verse': item['verse'],
                'clauses': item.get('clauses', []),
                'phrases': item.get('phrases', []),
                'wgs': item.get('wgs', []),
                'subphrases': [],
                'word_assignments': item.get('word_assignments', {}),
                'source': 'inferred',
                'confidence': item.get('confidence', 0.85)
            }

    # Build unknown word resolution lookup (word -> phrase_type, function)
    unknown_lookup = {}
    for res in unknown_res:
        word = res.get('original', '')
        unknown_lookup[word] = {
            'phrase_type': res.get('phrase_type'),
            'function': res.get('function'),
            'confidence': res.get('confidence', 0.5),
            'method': res.get('method', 'unknown')
        }

    # Find verses with unknown words that don't have structure yet
    unknown_words = classified[classified['structure_status'] == 'unknown']
    verse_unknowns = defaultdict(list)

    for _, row in unknown_words.iterrows():
        key = build_verse_key(row['book'], row['chapter'], row['verse'])
        word = row['word']
        resolution = unknown_lookup.get(word, {})

        verse_unknowns[key].append({
            'word_id': row['word_id'],
            'word': word,
            'phrase_type': resolution.get('phrase_type'),
            'function': resolution.get('function'),
            'confidence': resolution.get('confidence', 0.5),
            'method': resolution.get('method', 'unknown')
        })

    # For verses that only have unknown resolutions (not in direct or inferred)
    for key, unknowns in verse_unknowns.items():
        if key not in merged:
            parts = key.split(':')
            merged[key] = {
                'book': parts[0],
                'chapter': int(parts[1]),
                'verse': int(parts[2]),
                'clauses': [],
                'phrases': [],
                'wgs': [],
                'subphrases': [],
                'unknown_words': unknowns,
                'source': 'unknown_only',
                'confidence': sum(u['confidence'] for u in unknowns) / len(unknowns)
            }
        else:
            # Add unknown word info to existing structure
            merged[key]['unknown_words'] = unknowns

    return merged


def generate_structure_nodes(merged: dict, complete_df: pd.DataFrame) -> tuple:
    """
    Generate clause, phrase, and word group container nodes.

    Returns:
        - clause_nodes: list of clause node dicts
        - phrase_nodes: list of phrase node dicts
        - wg_nodes: list of word group node dicts
        - next_id: next available node ID
    """
    # Get starting node ID (after words, books, chapters, verses)
    # Load containers to find next available ID
    containers = pd.read_parquet('data/intermediate/tr_containers.parquet')
    next_id = containers['node_id'].max() + 1

    # Build word_id to slot mapping
    complete_df = complete_df.sort_values(
        ['book', 'chapter', 'verse', 'word_rank']
    ).reset_index(drop=True)
    word_to_slot = {row['word_id']: idx + 1 for idx, row in complete_df.iterrows()}

    clause_nodes = []
    phrase_nodes = []
    wg_nodes = []

    for verse_key, structure in merged.items():
        # Process clauses
        for clause in structure.get('clauses', []):
            tr_word_ids = clause.get('tr_word_ids', [])
            if not tr_word_ids:
                continue

            slots = [word_to_slot.get(wid) for wid in tr_word_ids if wid in word_to_slot]
            if not slots:
                continue

            node = {
                'node_id': next_id,
                'otype': 'clause',
                'book': structure['book'],
                'chapter': structure['chapter'],
                'verse': structure['verse'],
                'first_slot': min(slots),
                'last_slot': max(slots),
                'typ': clause.get('typ'),
                'clausetype': clause.get('clausetype'),
                'cltype': clause.get('cltype'),
                'n1904_node_id': clause.get('n1904_node_id'),
                'source': structure['source'],
                'confidence': clause.get('confidence', structure['confidence'])
            }
            clause_nodes.append(node)
            next_id += 1

        # Process phrases
        for phrase in structure.get('phrases', []):
            tr_word_ids = phrase.get('tr_word_ids', [])
            if not tr_word_ids:
                continue

            slots = [word_to_slot.get(wid) for wid in tr_word_ids if wid in word_to_slot]
            if not slots:
                continue

            node = {
                'node_id': next_id,
                'otype': 'phrase',
                'book': structure['book'],
                'chapter': structure['chapter'],
                'verse': structure['verse'],
                'first_slot': min(slots),
                'last_slot': max(slots),
                'typ': phrase.get('typ'),
                'function': phrase.get('function'),
                'rela': phrase.get('rela'),
                'n1904_node_id': phrase.get('n1904_node_id'),
                'source': structure['source'],
                'confidence': phrase.get('confidence', structure['confidence'])
            }
            phrase_nodes.append(node)
            next_id += 1

        # Process word groups
        for wg in structure.get('wgs', []):
            tr_word_ids = wg.get('tr_word_ids', [])
            if not tr_word_ids:
                continue

            slots = [word_to_slot.get(wid) for wid in tr_word_ids if wid in word_to_slot]
            if not slots:
                continue

            node = {
                'node_id': next_id,
                'otype': 'wg',
                'book': structure['book'],
                'chapter': structure['chapter'],
                'verse': structure['verse'],
                'first_slot': min(slots),
                'last_slot': max(slots),
                'typ': wg.get('typ'),
                'function': wg.get('function'),
                'rela': wg.get('rela'),
                'rule': wg.get('rule'),
                'n1904_node_id': wg.get('n1904_node_id'),
                'source': structure['source'],
                'confidence': wg.get('confidence', structure['confidence'])
            }
            wg_nodes.append(node)
            next_id += 1

    return clause_nodes, phrase_nodes, wg_nodes, next_id


def write_otype_file(nodes_by_type: dict, output_dir: Path):
    """Write the otype.tf file with all node type definitions."""
    output_path = output_dir / 'otype.tf'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("@node\n")
        f.write("@description=object type for each node\n")
        f.write("@valueType=str\n")
        f.write("\n")

        # Write in node order
        for otype in ['w', 'subphrase', 'wg', 'phrase', 'clause', 'sentence',
                      'verse', 'chapter', 'book']:
            if otype in nodes_by_type:
                nodes = sorted(nodes_by_type[otype], key=lambda n: n['node_id'])
                if nodes:
                    first = nodes[0]['node_id']
                    last = nodes[-1]['node_id']
                    f.write(f"{first}-{last}\t{otype}\n")


def write_structure_features(clause_nodes: list, phrase_nodes: list,
                            wg_nodes: list, output_dir: Path):
    """Write TF feature files for structure nodes."""

    # Combine all structure nodes
    all_nodes = clause_nodes + phrase_nodes + wg_nodes

    # Write typ feature
    typ_path = output_dir / 'typ.tf'
    with open(typ_path, 'w', encoding='utf-8') as f:
        f.write("@node\n")
        f.write("@description=syntactic type of phrase/clause\n")
        f.write("@valueType=str\n")
        f.write("\n")
        for node in sorted(all_nodes, key=lambda n: n['node_id']):
            if node.get('typ'):
                f.write(f"{node['node_id']}\t{node['typ']}\n")

    # Write function feature
    func_path = output_dir / 'function.tf'
    with open(func_path, 'w', encoding='utf-8') as f:
        f.write("@node\n")
        f.write("@description=syntactic function of phrase/word group\n")
        f.write("@valueType=str\n")
        f.write("\n")
        for node in sorted(phrase_nodes + wg_nodes, key=lambda n: n['node_id']):
            if node.get('function'):
                f.write(f"{node['node_id']}\t{node['function']}\n")

    # Write rela feature
    rela_path = output_dir / 'rela.tf'
    with open(rela_path, 'w', encoding='utf-8') as f:
        f.write("@node\n")
        f.write("@description=relation to context\n")
        f.write("@valueType=str\n")
        f.write("\n")
        for node in sorted(phrase_nodes + wg_nodes, key=lambda n: n['node_id']):
            if node.get('rela'):
                f.write(f"{node['node_id']}\t{node['rela']}\n")

    # Write clausetype feature
    cltype_path = output_dir / 'clausetype.tf'
    with open(cltype_path, 'w', encoding='utf-8') as f:
        f.write("@node\n")
        f.write("@description=clause type\n")
        f.write("@valueType=str\n")
        f.write("\n")
        for node in sorted(clause_nodes, key=lambda n: n['node_id']):
            if node.get('clausetype'):
                f.write(f"{node['node_id']}\t{node['clausetype']}\n")

    # Write rule feature for word groups
    rule_path = output_dir / 'rule.tf'
    with open(rule_path, 'w', encoding='utf-8') as f:
        f.write("@node\n")
        f.write("@description=syntactic rule for word group\n")
        f.write("@valueType=str\n")
        f.write("\n")
        for node in sorted(wg_nodes, key=lambda n: n['node_id']):
            if node.get('rule'):
                f.write(f"{node['node_id']}\t{node['rule']}\n")

    # Write structure_source feature (for transparency)
    source_path = output_dir / 'structure_source.tf'
    with open(source_path, 'w', encoding='utf-8') as f:
        f.write("@node\n")
        f.write("@description=source of structure (direct/inferred/unknown_only)\n")
        f.write("@valueType=str\n")
        f.write("\n")
        for node in sorted(all_nodes, key=lambda n: n['node_id']):
            if node.get('source'):
                f.write(f"{node['node_id']}\t{node['source']}\n")

    # Write structure_confidence feature
    conf_path = output_dir / 'structure_confidence.tf'
    with open(conf_path, 'w', encoding='utf-8') as f:
        f.write("@node\n")
        f.write("@description=confidence score for structure inference (0-1)\n")
        f.write("@valueType=float\n")
        f.write("\n")
        for node in sorted(all_nodes, key=lambda n: n['node_id']):
            if node.get('confidence') is not None:
                f.write(f"{node['node_id']}\t{node['confidence']:.2f}\n")


def write_otext_update(output_dir: Path):
    """Update otext.tf to include structure nodes in sections."""
    # Read existing otext
    otext_path = output_dir / 'otext.tf'
    if otext_path.exists():
        with open(otext_path, 'r') as f:
            content = f.read()

        # Check if structure types need to be added
        if 'clause' not in content:
            # Append structure level info
            with open(otext_path, 'a') as f:
                f.write("\n# Structure levels\n")
                f.write("@structureTypes=clause,phrase,wg\n")


def save_structure_summary(clause_nodes: list, phrase_nodes: list,
                          wg_nodes: list, output_path: Path):
    """Save structure nodes as parquet for further processing."""
    import pandas as pd

    all_nodes = []
    for node in clause_nodes + phrase_nodes + wg_nodes:
        all_nodes.append({
            'node_id': node['node_id'],
            'otype': node['otype'],
            'book': node['book'],
            'chapter': node['chapter'],
            'verse': node['verse'],
            'first_slot': node['first_slot'],
            'last_slot': node['last_slot'],
            'typ': node.get('typ'),
            'function': node.get('function'),
            'rela': node.get('rela'),
            'rule': node.get('rule'),
            'clausetype': node.get('clausetype'),
            'source': node.get('source'),
            'confidence': node.get('confidence'),
            'n1904_node_id': node.get('n1904_node_id')
        })

    df = pd.DataFrame(all_nodes)
    df.to_parquet(output_path, index=False)


def main():
    """Main entry point."""
    config = load_config()

    with ScriptLogger('p4_08e_generate_structure_tf') as logger:
        # Load all structure data
        logger.info("Loading structure data...")
        direct, inferred, unknown_res, classified = load_structure_data()
        logger.info(f"  Direct transplant verses: {len(direct):,}")
        logger.info(f"  Inferred verses: {len(inferred):,}")
        logger.info(f"  Unknown word forms resolved: {len(unknown_res):,}")

        # Merge structures
        logger.info("Merging structures...")
        merged = merge_structures(direct, inferred, unknown_res, classified)
        logger.info(f"  Total verses with structure: {len(merged):,}")

        # Count by source
        by_source = defaultdict(int)
        for v in merged.values():
            by_source[v['source']] += 1
        for source, count in sorted(by_source.items()):
            logger.info(f"    {source}: {count:,}")

        # Load complete word data for slot mapping
        logger.info("Loading word data...")
        complete = pd.read_parquet('data/intermediate/tr_complete.parquet')
        logger.info(f"  Total words: {len(complete):,}")

        # Generate structure nodes
        logger.info("Generating structure nodes...")
        clause_nodes, phrase_nodes, wg_nodes, next_id = generate_structure_nodes(
            merged, complete
        )
        logger.info(f"  Clauses: {len(clause_nodes):,}")
        logger.info(f"  Phrases: {len(phrase_nodes):,}")
        logger.info(f"  Word groups: {len(wg_nodes):,}")
        logger.info(f"  Next node ID: {next_id:,}")

        # Write TF files
        output_dir = Path('data/output/tf')
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Writing TF feature files...")
        write_structure_features(clause_nodes, phrase_nodes, wg_nodes, output_dir)

        # Save structure summary
        summary_path = Path('data/intermediate/tr_structure_nodes.parquet')
        save_structure_summary(clause_nodes, phrase_nodes, wg_nodes, summary_path)
        logger.info(f"Saved structure summary to: {summary_path}")

        # Statistics
        logger.info("\n" + "=" * 60)
        logger.info("STRUCTURE GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total structure nodes: {len(clause_nodes) + len(phrase_nodes) + len(wg_nodes):,}")
        logger.info(f"  Clauses: {len(clause_nodes):,}")
        logger.info(f"  Phrases: {len(phrase_nodes):,}")
        logger.info(f"  Word groups: {len(wg_nodes):,}")

        # Confidence distribution
        all_nodes = clause_nodes + phrase_nodes + wg_nodes
        high_conf = sum(1 for n in all_nodes if n.get('confidence', 0) >= 0.8)
        med_conf = sum(1 for n in all_nodes if 0.6 <= n.get('confidence', 0) < 0.8)
        low_conf = sum(1 for n in all_nodes if n.get('confidence', 0) < 0.6)

        logger.info(f"\nConfidence distribution:")
        logger.info(f"  High (>=80%): {high_conf:,}")
        logger.info(f"  Medium (60-80%): {med_conf:,}")
        logger.info(f"  Low (<60%): {low_conf:,}")

        logger.info(f"\nTF files written to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
