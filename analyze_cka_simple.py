#!/usr/bin/env python3
"""
Simple CKA Analysis - focuses on the specific file
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_cka_file(filepath):
    """Load and analyze CKA file"""
    with open(filepath, 'r') as f:
        cka_data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"CKA Analysis: {filepath.split('/')[-1]}")
    print(f"{'='*80}\n")
    
    # Basic info
    print(f"Comparison: {cka_data['compare']}")
    print(f"Label A: {cka_data['label_a']}")
    print(f"Label B: {cka_data['label_b']}")
    print(f"\nMethod A: {cka_data['metadata_a']}")
    print(f"Method B: {cka_data['metadata_b']}")
    
    # Summary statistics
    cka_stats = cka_data['cka']
    per_layer = cka_data['per_layer_cka']
    scores = list(per_layer.values())
    
    print(f"\n{'SUMMARY STATISTICS':^80}")
    print(f"Mean CKA:        {cka_stats['mean']:.6f}")
    print(f"Min CKA:         {cka_stats['min']:.6f}")
    print(f"Max CKA:         {cka_stats['max']:.6f}")
    print(f"Median CKA:      {np.median(scores):.6f}")
    print(f"Std Dev:         {np.std(scores):.6f}")
    print(f"Num Layers:      {cka_stats['n_layers']}")
    
    # Per-layer analysis
    print(f"\n{'PER-LAYER CKA SCORES':^80}")
    
    sorted_layers = sorted(per_layer.items(), 
                          key=lambda x: int(x[0].split('.')[2]))
    
    print(f"{'Layer':<35} {'CKA Score':<15} {'Category':<20}")
    print("-" * 70)
    
    for layer_name, cka_score in sorted_layers:
        if cka_score >= 0.8:
            category = "High Similarity ✓"
        elif cka_score >= 0.6:
            category = "Medium Similarity"
        elif cka_score >= 0.4:
            category = "Low Similarity"
        else:
            category = "Very Low ✗"
        print(f"{layer_name:<35} {cka_score:<15.6f} {category:<20}")
    
    # Layer group analysis
    print(f"\n{'LAYER GROUP ANALYSIS':^80}")
    layers = sorted([(int(name.split('.')[2]), score) for name, score in per_layer.items()])
    
    n_layers = len(layers)
    early = layers[:n_layers//3]
    middle = layers[n_layers//3:2*n_layers//3]
    deep = layers[2*n_layers//3:]
    
    print(f"Early layers (0-{early[-1][0]}):   Mean CKA = {np.mean([s for _, s in early]):.6f}")
    print(f"Middle layers ({middle[0][0]}-{middle[-1][0]}): Mean CKA = {np.mean([s for _, s in middle]):.6f}")
    print(f"Deep layers ({deep[0][0]}-{deep[-1][0]}):   Mean CKA = {np.mean([s for _, s in deep]):.6f}")
    
    # Insights
    print(f"\n{'KEY INSIGHTS':^80}")
    min_layer = min(per_layer.items(), key=lambda x: x[1])
    max_layer = max(per_layer.items(), key=lambda x: x[1])
    
    print(f"✓ Highest agreement: {max_layer[0]:<30} CKA = {max_layer[1]:.4f}")
    print(f"✗ Lowest agreement:  {min_layer[0]:<30} CKA = {min_layer[1]:.4f}")
    print(f"  Difference: {max_layer[1] - min_layer[1]:.4f}")
    
    print(f"\nInterpretation:")
    if cka_stats['mean'] >= 0.8:
        print(f"  → Very similar representations (both methods preserve very similar features)")
    elif cka_stats['mean'] >= 0.6:
        print(f"  → Reasonably similar representations (good overall agreement)")
    elif cka_stats['mean'] >= 0.4:
        print(f"  → Moderate similarity (notable differences between methods)")
    else:
        print(f"  → Low similarity (significantly different pruning strategies)")
    
    # Check divergence pattern
    early_mean = np.mean([s for _, s in early])
    deep_mean = np.mean([s for _, s in deep])
    if early_mean > deep_mean:
        print(f"  → Pattern: Both methods MORE similar in early layers, DIVERGE in deep layers")
    elif deep_mean > early_mean:
        print(f"  → Pattern: Both methods MORE similar in deep layers, DIVERGE in early layers")
    else:
        print(f"  → Pattern: Consistent similarity across all layers")
    
    return cka_data, sorted_layers, (early, middle, deep)


def create_visualization(cka_data, sorted_layers, layer_groups):
    """Create visualization"""
    early, middle, deep = layer_groups
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data
    layers = [str(name).split('.')[2] for name, _ in sorted_layers]
    scores = [score for _, score in sorted_layers]
    
    # Plot 1: Per-layer CKA scores (bar chart)
    ax = axes[0, 0]
    colors = ['green' if s >= 0.8 else 'orange' if s >= 0.6 else 'red' for s in scores]
    ax.bar(range(len(layers)), scores, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=cka_data['cka']['mean'], color='blue', linestyle='--', 
               linewidth=2.5, label=f"Mean: {cka_data['cka']['mean']:.4f}")
    ax.set_xlabel('Layer Index', fontsize=11)
    ax.set_ylabel('CKA Score', fontsize=11)
    ax.set_title('Per-Layer CKA Similarity Scores', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 2: Line plot showing trend
    ax = axes[0, 1]
    layer_indices = list(range(len(layers)))
    ax.plot(layer_indices, scores, marker='o', linewidth=2.5, markersize=7, color='darkblue')
    ax.fill_between(layer_indices, scores, alpha=0.3, color='skyblue')
    ax.axhline(y=cka_data['cka']['mean'], color='red', linestyle='--', 
               linewidth=2.5, label=f"Mean: {cka_data['cka']['mean']:.4f}")
    ax.set_xlabel('Layer Index', fontsize=11)
    ax.set_ylabel('CKA Score', fontsize=11)
    ax.set_title('CKA Trend Across Layers', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 3: Layer group comparison
    ax = axes[1, 0]
    group_names = ['Early\nLayers', 'Middle\nLayers', 'Deep\nLayers']
    group_means = [
        np.mean([s for _, s in early]),
        np.mean([s for _, s in middle]),
        np.mean([s for _, s in deep])
    ]
    colors_group = ['green' if m >= 0.8 else 'orange' if m >= 0.6 else 'red' for m in group_means]
    bars = ax.bar(group_names, group_means, color=colors_group, alpha=0.7, edgecolor='black', width=0.6)
    ax.axhline(y=cka_data['cka']['mean'], color='blue', linestyle='--', 
               linewidth=2, label=f"Overall Mean: {cka_data['cka']['mean']:.4f}")
    ax.set_ylabel('Mean CKA Score', fontsize=11)
    ax.set_title('CKA Similarity by Layer Group', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, group_means)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    COMPARISON SUMMARY
    {'═' * 35}
    
    Method A: {cka_data['metadata_a'].get('method', 'N/A')} 
              @ {cka_data['metadata_a'].get('sparsity', 'N/A')}% sparsity
    
    Method B: {cka_data['metadata_b'].get('method', 'N/A')}
              @ {cka_data['metadata_b'].get('sparsity', 'N/A')}% sparsity
    
    {'─' * 35}
    STATISTICS:
    {'─' * 35}
    Mean CKA:     {cka_data['cka']['mean']:.6f}
    Median CKA:   {np.median(scores):.6f}
    Std Deviation:{np.std(scores):.6f}
    Min CKA:      {cka_data['cka']['min']:.6f}
    Max CKA:      {cka_data['cka']['max']:.6f}
    
    Num Layers:   {cka_data['cka']['n_layers']}
    Samples:      {cka_data['n_samples']}
    
    {'═' * 35}
    SCORE INTERPRETATION:
    0.8 - 1.0 → Very similar (High)
    0.6 - 0.8 → Similar (Medium)
    0.4 - 0.6 → Somewhat different (Low)
    0.0 - 0.4 → Very different (Very Low)
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))
    
    plt.tight_layout()
    
    output_path = '/home/xie.yiyi/Reinforcement-Casino/masks/cka_analysis_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    filepath = '/home/xie.yiyi/Reinforcement-Casino/masks/cka_mask_vs_mask_cold_start_cav_90pct_vs_cold_start_snip_90pct.json'
    
    cka_data, sorted_layers, layer_groups = analyze_cka_file(filepath)
    create_visualization(cka_data, sorted_layers, layer_groups)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")
