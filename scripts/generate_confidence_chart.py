#!/usr/bin/env python3
"""
Generate confidence distribution chart for README.

Creates a bar chart showing the distribution of confidence scores
across structure nodes (clauses, phrases, word groups).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def main():
    # Load structure nodes
    nodes = pd.read_parquet('data/intermediate/tr_structure_nodes.parquet')

    print(f"Total structure nodes: {len(nodes):,}")

    # Define confidence bins
    bins = [0, 0.6, 0.8, 0.9, 0.95, 1.0, 1.01]
    labels = ['<60%', '60-80%', '80-90%', '90-95%', '95-100%', '100%']
    nodes['conf_bin'] = pd.cut(nodes['confidence'], bins=bins, labels=labels, right=False)

    # Get distribution
    dist = nodes['conf_bin'].value_counts().sort_index()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Color scheme
    colors = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#43a047', '#2e7d32']

    # Plot 1: Overall confidence distribution
    bars = ax1.bar(range(len(dist)), dist.values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(len(dist)))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_xlabel('Confidence Level', fontsize=12)
    ax1.set_ylabel('Number of Nodes', fontsize=12)
    ax1.set_title('Structure Node Confidence Distribution', fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, val in zip(bars, dist.values):
        height = bar.get_height()
        pct = 100 * val / len(nodes)
        ax1.annotate(f'{val:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax1.set_ylim(0, max(dist.values) * 1.15)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: By source type
    source_data = []
    source_labels = []
    source_colors = ['#1976d2', '#7b1fa2', '#c2185b']

    for source in ['direct', 'inferred', 'unknown_only']:
        src_nodes = nodes[nodes['source'] == source]
        source_data.append(len(src_nodes))
        avg_conf = src_nodes['confidence'].mean()
        source_labels.append(f'{source}\n({avg_conf:.0%} avg)')

    bars2 = ax2.bar(range(len(source_data)), source_data, color=source_colors,
                    edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(source_data)))
    ax2.set_xticklabels(source_labels, fontsize=10)
    ax2.set_xlabel('Structure Source', fontsize=12)
    ax2.set_ylabel('Number of Nodes', fontsize=12)
    ax2.set_title('Structure Nodes by Source', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars2, source_data):
        height = bar.get_height()
        pct = 100 * val / len(nodes)
        ax2.annotate(f'{val:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax2.set_ylim(0, max(source_data) * 1.15)
    ax2.grid(axis='y', alpha=0.3)

    # Add legend explaining sources
    legend_text = [
        'direct: 100% word alignment with N1904',
        'inferred: Known words, different positions',
        'unknown_only: TR-only words resolved via NLP'
    ]

    plt.tight_layout()

    # Save figure
    output_path = Path('docs/confidence_distribution.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved chart to: {output_path}")

    # Also save a simpler single chart version
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Stacked bar by source and confidence
    sources = ['direct', 'inferred', 'unknown_only']
    source_names = ['Direct Transplant', 'Inferred', 'Unknown Resolution']
    x = range(len(labels))
    width = 0.25

    for i, (source, name, color) in enumerate(zip(sources, source_names, source_colors)):
        src_nodes = nodes[nodes['source'] == source]
        src_dist = src_nodes['conf_bin'].value_counts().sort_index()
        # Ensure all bins are present
        src_counts = [src_dist.get(label, 0) for label in labels]
        offset = (i - 1) * width
        bars = ax.bar([xi + offset for xi in x], src_counts, width,
                     label=name, color=color, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel('Confidence Level', fontsize=12)
    ax.set_ylabel('Number of Nodes', fontsize=12)
    ax.set_title('Structure Node Confidence by Source Type', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path2 = Path('docs/confidence_by_source.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved chart to: {output_path2}")

    # Print summary stats
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"Total nodes: {len(nodes):,}")
    print(f"\nBy confidence level:")
    for label, count in dist.items():
        pct = 100 * count / len(nodes)
        print(f"  {label:>10}: {count:>6,} ({pct:5.1f}%)")

    print(f"\nBy source:")
    for source in sources:
        src_nodes = nodes[nodes['source'] == source]
        avg_conf = src_nodes['confidence'].mean()
        print(f"  {source:>12}: {len(src_nodes):>6,} nodes, avg conf {avg_conf:.1%}")

    plt.close('all')


if __name__ == '__main__':
    main()
