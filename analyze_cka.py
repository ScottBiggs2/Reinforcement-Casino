#!/usr/bin/env python3
"""
Analyze and visualize CKA (Centered Kernel Alignment) outputs
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['figure.figsize'] = (14, 10)


def load_cka_file(filepath):
    """Load CKA JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_single_cka(cka_data, filepath):
    """Analyze and print a single CKA file"""
    print(f"\n{'='*80}")
    print(f"File: {filepath}")
    print(f"{'='*80}")
    
    # Basic info
    print(f"\nComparison Type: {cka_data['compare']}")
    print(f"Label A: {cka_data['label_a']}")
    print(f"Label B: {cka_data['label_b']}")
    print(f"Metadata A: {cka_data['metadata_a']}")
    print(f"Metadata B: {cka_data['metadata_b']}")
    
    # Summary statistics
    cka_stats = cka_data['cka']
    print(f"\n{'SUMMARY STATISTICS':^80}")
    print(f"Mean CKA:        {cka_stats['mean']:.6f}")
    print(f"Min CKA:         {cka_stats['min']:.6f}")
    print(f"Max CKA:         {cka_stats['max']:.6f}")
    print(f"Std Dev:         {np.std(list(cka_data['per_layer_cka'].values())):.6f}")
    print(f"Num Layers:      {cka_stats['n_layers']}")
    print(f"Num Skipped:     {cka_stats['n_skipped']}")
    
    # Per-layer analysis
    per_layer = cka_data['per_layer_cka']
    print(f"\n{'PER-LAYER CKA SCORES':^80}")
    
    # Sort by layer number
    sorted_layers = sorted(per_layer.items(), 
                          key=lambda x: int(x[0].split('.')[2]))
    
    print(f"{'Layer':<30} {'CKA Score':<15} {'Category':<20}")
    print("-" * 65)
    
    for layer_name, cka_score in sorted_layers:
        if cka_score >= 0.8:
            category = "High Similarity"
        elif cka_score >= 0.6:
            category = "Medium Similarity"
        elif cka_score >= 0.4:
            category = "Low Similarity"
        else:
            category = "Very Low Similarity"
        print(f"{layer_name:<30} {cka_score:<15.6f} {category:<20}")
    
    # Analysis by layer groups
    print(f"\n{'LAYER GROUP ANALYSIS':^80}")
    layers = sorted([(int(name.split('.')[2]), score) for name, score in per_layer.items()])
    
    n_layers = len(layers)
    early = layers[:n_layers//3]
    middle = layers[n_layers//3:2*n_layers//3]
    deep = layers[2*n_layers//3:]
    
    print(f"Early layers (0-{early[-1][0]}):   {np.mean([s for _, s in early]):.6f}")
    print(f"Middle layers ({middle[0][0]}-{middle[-1][0]}): {np.mean([s for _, s in middle]):.6f}")
    print(f"Deep layers ({deep[0][0]}-{deep[-1][0]}):   {np.mean([s for _, s in deep]):.6f}")
    
    # Insights
    print(f"\n{'KEY INSIGHTS':^80}")
    min_layer = min(per_layer.items(), key=lambda x: x[1])
    max_layer = max(per_layer.items(), key=lambda x: x[1])
    
    print(f"✓ Highest agreement: {max_layer[0]} (CKA: {max_layer[1]:.4f})")
    print(f"✗ Lowest agreement:  {min_layer[0]} (CKA: {min_layer[1]:.4f})")
    
    if cka_stats['mean'] >= 0.8:
        print(f"• Overall: Very similar representations (both methods preserve similar features)")
    elif cka_stats['mean'] >= 0.6:
        print(f"• Overall: Reasonably similar representations (good agreement overall)")
    else:
        print(f"• Overall: Moderate to low similarity (notable differences between methods)")
    
    return cka_data, sorted_layers


def visualize_cka(cka_data, sorted_layers, output_path=None):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data
    layers = [str(name).split('.')[2] for name, _ in sorted_layers]
    scores = [score for _, score in sorted_layers]
    
    # Plot 1: Per-layer CKA scores (bar chart)
    ax = axes[0, 0]
    colors = ['green' if s >= 0.8 else 'orange' if s >= 0.6 else 'red' for s in scores]
    ax.bar(range(len(layers)), scores, color=colors, alpha=0.7)
    ax.axhline(y=cka_data['cka']['mean'], color='blue', linestyle='--', 
               linewidth=2, label=f"Mean: {cka_data['cka']['mean']:.4f}")
    ax.set_xlabel('Layer')
    ax.set_ylabel('CKA Score')
    ax.set_title('Per-Layer CKA Similarity Scores')
    ax.set_xticks(range(0, len(layers), max(1, len(layers)//10)))
    ax.set_xticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers)//10))], 
                        rotation=45)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Line plot showing trend
    ax = axes[0, 1]
    ax.plot(range(len(layers)), scores, marker='o', linewidth=2, markersize=6)
    ax.fill_between(range(len(layers)), scores, alpha=0.3)
    ax.axhline(y=cka_data['cka']['mean'], color='red', linestyle='--', 
               label=f"Mean: {cka_data['cka']['mean']:.4f}")
    ax.set_xlabel('Layer')
    ax.set_ylabel('CKA Score')
    ax.set_title('CKA Trend Across Layers')
    ax.set_xticks(range(0, len(layers), max(1, len(layers)//10)))
    ax.set_xticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers)//10))], 
                        rotation=45)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Distribution histogram
    ax = axes[1, 0]
    ax.hist(scores, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=cka_data['cka']['mean'], color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {cka_data['cka']['mean']:.4f}")
    ax.axvline(x=np.median(scores), color='green', linestyle='--', 
               linewidth=2, label=f"Median: {np.median(scores):.4f}")
    ax.set_xlabel('CKA Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of CKA Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    COMPARISON: {cka_data['compare']}
    
    Method A: {cka_data['metadata_a'].get('method', 'N/A')} @ {cka_data['metadata_a'].get('sparsity', 'N/A')}%
    Method B: {cka_data['metadata_b'].get('method', 'N/A')} @ {cka_data['metadata_b'].get('sparsity', 'N/A')}%
    
    STATISTICS:
    Mean CKA:    {cka_data['cka']['mean']:.6f}
    Median CKA:  {np.median(scores):.6f}
    Std Dev:     {np.std(scores):.6f}
    Min CKA:     {cka_data['cka']['min']:.6f}
    Max CKA:     {cka_data['cka']['max']:.6f}
    
    Num Layers:  {cka_data['cka']['n_layers']}
    Skipped:     {cka_data['cka']['n_skipped']}
    
    INTERPRETATION:
    • Mean Score indicates overall similarity
    • Higher values (>0.8) = highly similar
    • Lower values (<0.4) = divergent behavior
    • Trend shows layer-wise agreement pattern
    """
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
    
    return fig


if __name__ == "__main__":
    # Find and analyze all CKA files
    cka_files = list(Path('/home/xie.yiyi/Reinforcement-Casino').rglob('*cka*.json'))
    
    if not cka_files:
        print("No CKA files found!")
    else:
        all_results = []
        for cka_file in sorted(cka_files):
            cka_data = load_cka_file(cka_file)
            result, sorted_layers = analyze_single_cka(cka_data, cka_file)
            all_results.append((cka_file, result, sorted_layers))
            
            # Create visualization for each file
            output_name = cka_file.stem + "_visualization.png"
            output_path = cka_file.parent / output_name
            visualize_cka(result, sorted_layers, output_path)
        
        print(f"\n{'='*80}")
        print(f"Analysis complete! Processed {len(all_results)} CKA file(s)")
        print(f"{'='*80}\n")
