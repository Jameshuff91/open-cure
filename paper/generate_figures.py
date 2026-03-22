"""Generate publication-quality figures for bioRxiv preprint."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

output_dir = os.path.dirname(os.path.abspath(__file__))


def fig1_confidence_tiers():
    """Figure 1: Confidence tier precision with prediction counts."""
    tiers = ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']
    precision = [87.1, 83.4, 38.5, 11.3, 9.2]
    stderr = [2.7, 4.0, 3.6, 0.5, 0.5]
    counts = [991, 1168, 914, 9113, 8978]
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']

    fig, ax1 = plt.subplots(figsize=(8, 5))

    bars = ax1.bar(tiers, precision, yerr=stderr, capsize=4,
                   color=colors, edgecolor='white', linewidth=0.5,
                   error_kw={'linewidth': 1.2, 'color': '#333'})
    ax1.set_ylabel('Holdout Precision (%)')
    ax1.set_ylim(0, 100)
    ax1.axhline(y=50, color='#bbb', linestyle='--', linewidth=0.8, alpha=0.5)

    # Add count labels on bars
    for bar, count, prec in zip(bars, counts, precision):
        ax1.text(bar.get_x() + bar.get_width() / 2, prec + 4.5,
                 f'n={count:,}', ha='center', va='bottom', fontsize=9, color='#555')

    ax1.set_title('Confidence Tier Holdout Precision', fontweight='bold', pad=15)
    ax1.set_xlabel('Confidence Tier')

    # Add annotation
    ax1.annotate('GOLDEN+HIGH: 2,159 predictions\nat >83% precision',
                 xy=(0.5, 85), xytext=(3.2, 75),
                 fontsize=9, color='#27ae60',
                 arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#27ae60', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_confidence_tiers.png'))
    plt.savefig(os.path.join(output_dir, 'fig1_confidence_tiers.pdf'))
    plt.close()
    print('Figure 1: Confidence tiers - DONE')


def fig2_rare_disease_performance():
    """Figure 2: Performance by disease rarity (known treatments)."""
    categories = ['1\n(orphan)', '2-5', '6-10', '11+']
    r30 = [13.5, 20.0, 32.2, 27.8]
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(categories, r30, color=colors, edgecolor='white', linewidth=0.5, width=0.6)

    ax.set_ylabel('Recall@30 (%)')
    ax.set_xlabel('Number of Known Treatments')
    ax.set_title('Performance Scales with Disease Rarity', fontweight='bold', pad=15)
    ax.set_ylim(0, 40)

    # Add value labels
    for bar, val in zip(bars, r30):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add 2.4x annotation
    ax.annotate('2.4× gap', xy=(0, 13.5), xytext=(1.5, 8),
                fontsize=10, color='#c0392b', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))
    ax.annotate('', xy=(2, 32.2), xytext=(1.8, 9),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_rare_disease.png'))
    plt.savefig(os.path.join(output_dir, 'fig2_rare_disease.pdf'))
    plt.close()
    print('Figure 2: Rare disease performance - DONE')


def fig3_method_comparison():
    """Figure 3: Method comparison across evaluation paradigms."""
    methods = [
        'kNN k=20\n(original)',
        'kNN k=20\n(honest)',
        'XGBoost\n(tuned)',
        'KEGG\nPathway kNN',
        'TxGNN\n(upper)',
        'TxGNN\n(lower)',
    ]
    r30 = [36.59, 26.06, 25.85, 15.73, 14.5, 6.7]
    stderr = [3.90, 3.84, 4.06, 1.82, 0, 0]
    colors = ['#bdc3c7', '#2980b9', '#3498db', '#1abc9c', '#9b59b6', '#9b59b6']
    hatches = ['', '', '', '', '//', '//']

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(methods, r30, yerr=stderr, capsize=4,
                  color=colors, edgecolor='#555', linewidth=0.5,
                  error_kw={'linewidth': 1.2, 'color': '#333'})
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_ylabel('Recall@30 (%)')
    ax.set_title('Method Comparison Across Evaluation Paradigms', fontweight='bold', pad=15)
    ax.set_ylim(0, 45)

    # Bracket for "Transductive"
    ax.annotate('', xy=(0, 42), xytext=(2, 42),
                arrowprops=dict(arrowstyle='-', color='#2980b9', lw=1.5))
    ax.text(1, 43, 'Transductive', ha='center', fontsize=9, color='#2980b9', fontweight='bold')

    # Bracket for "Inductive"
    ax.annotate('', xy=(3, 42), xytext=(5, 42),
                arrowprops=dict(arrowstyle='-', color='#1abc9c', lw=1.5))
    ax.text(4, 43, 'Inductive', ha='center', fontsize=9, color='#1abc9c', fontweight='bold')

    # Leakage annotation
    ax.annotate('10.5 pp\nleakage', xy=(0.5, 31), xytext=(0.5, 31),
                fontsize=8, ha='center', color='#7f8c8d',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#ecf0f1', edgecolor='none'))

    # Add "includes treatment edges" note
    ax.text(0, 1.5, '*includes\ntreatment edges', ha='center', fontsize=7, color='#95a5a6', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_method_comparison.png'))
    plt.savefig(os.path.join(output_dir, 'fig3_method_comparison.pdf'))
    plt.close()
    print('Figure 3: Method comparison - DONE')


def fig4_bimodal_coverage():
    """Figure 4: Bimodal coverage distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [1, 1.3]})

    # Left: pie/donut chart of coverage
    sizes = [84.7, 15.3]
    colors_pie = ['#2ecc71', '#e74c3c']
    explode = (0.02, 0.05)
    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, colors=colors_pie,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    ax1.set_title('Disease Coverage', fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', label='With kNN coverage'),
        mpatches.Patch(facecolor='#e74c3c', label='Zero coverage')
    ]
    ax1.legend(handles=legend_elements, loc='lower center', fontsize=9)

    # Right: bar chart showing performance split
    cats = ['With Coverage\n(84.7% diseases)', 'Zero Coverage\n(15.3% diseases)', 'Overall\n(reported)']
    vals = [24.2, 0.0, 26.06]
    bar_colors = ['#2ecc71', '#e74c3c', '#3498db']

    bars = ax2.bar(cats, vals, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.5)
    ax2.set_ylabel('Recall@30 (%)')
    ax2.set_title('Performance by Coverage Status', fontweight='bold')
    ax2.set_ylim(0, 35)

    for bar, val in zip(bars, vals):
        label = f'{val}%' if val > 0 else '0.0%'
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout(w_pad=3)
    plt.savefig(os.path.join(output_dir, 'fig4_bimodal_coverage.png'))
    plt.savefig(os.path.join(output_dir, 'fig4_bimodal_coverage.pdf'))
    plt.close()
    print('Figure 4: Bimodal coverage - DONE')


def fig5_ceiling_analysis():
    """Figure 5: Performance ceiling analysis."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ['Node2Vec\nCosine\n(no ML)', 'TxGNN\n(lower)', 'TxGNN\n(upper)',
               'KEGG\nPathway', 'XGBoost\n(honest)', 'kNN\n(honest)',
               'kNN\n(original)', 'DRKG\nCeiling', 'Oracle\nCeiling']
    values = [1.27, 6.7, 14.5, 15.73, 25.85, 26.06, 36.59, 37.0, 60.0]
    colors = ['#bdc3c7', '#d2b4de', '#9b59b6', '#1abc9c', '#3498db', '#2980b9',
              '#bdc3c7', '#e67e22', '#e74c3c']

    bars = ax.barh(methods, values, color=colors, edgecolor='white', linewidth=0.5, height=0.6)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{val}%', ha='left', va='center', fontsize=9)

    ax.set_xlabel('Recall@30 (%)')
    ax.set_title('Performance Ceiling Analysis', fontweight='bold', pad=15)
    ax.set_xlim(0, 70)

    # Ceiling lines
    ax.axvline(x=37, color='#e67e22', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=60, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5)

    # Annotations
    ax.text(38, 8.5, 'DRKG ceiling\n(37%)', fontsize=8, color='#e67e22', style='italic')
    ax.text(61, 8.5, 'Oracle\n(60%)', fontsize=8, color='#e74c3c', style='italic')

    # 62% of ceiling annotation
    ax.annotate('62% of\ntheoretical max',
                xy=(36.59, 6), xytext=(48, 4),
                fontsize=9, color='#2980b9', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.2))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_ceiling_analysis.png'))
    plt.savefig(os.path.join(output_dir, 'fig5_ceiling_analysis.pdf'))
    plt.close()
    print('Figure 5: Ceiling analysis - DONE')


if __name__ == '__main__':
    fig1_confidence_tiers()
    fig2_rare_disease_performance()
    fig3_method_comparison()
    fig4_bimodal_coverage()
    fig5_ceiling_analysis()
    print(f'\nAll figures saved to: {output_dir}')
