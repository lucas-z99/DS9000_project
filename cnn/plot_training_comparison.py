"""Plot recorded MLP and repaired CNN learning curves (English labels)."""
import json
import math
from pathlib import Path
from cnn.experiment_paths import model_run_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    base = Path(__file__).resolve().parent
    source = base / 'runs' / 'fixed_100epochs'
    output = base / 'figures'
    output.mkdir(exist_ok=True)
    results = json.loads((source / 'results.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(left=.07, right=.97, bottom=.19, top=.79, wspace=.23)
    fig.suptitle('MLP vs. Repaired CNN: Training Comparison', fontsize=20,
                 fontweight='bold', x=.07, ha='left', y=.96)
    fig.text(.07, .885, '100 epochs | Same 48 x 48 images and writer-based data split',
             color='#526170', fontsize=12)
    styles = [('mlp', 'MLP (fully connected)', '#2764B0', '-'),
              ('cnn', 'Repaired CNN (LeakyReLU)', '#D65B21', '--')]
    for key, label, color, linestyle in styles:
        rows = json.loads((model_run_dir(source, key) / f'{key}_history.json').read_text())
        assert [r['epoch'] for r in rows] == list(range(1, 101))
        assert all(math.isfinite(r['train_loss']) and 0 <= r['val_acc'] <= 1 for r in rows)
        best = max(rows, key=lambda r: r['val_acc'])
        assert best['epoch'] == results[key]['best_epoch']
        for ax, metric, scale in zip(axes, ['train_loss', 'val_acc'], [1, 100]):
            ax.plot([r['epoch'] for r in rows], [r[metric]*scale for r in rows],
                    color=color, linestyle=linestyle, linewidth=2.2, label=label)
            ax.axvline(best['epoch'], color=color, linestyle=':', alpha=.55, linewidth=1.3)
            ax.scatter(best['epoch'], best[metric]*scale, s=145, marker='*',
                       color=color, edgecolor='white', linewidth=.8, zorder=5)
        xy = (best['epoch'], best['val_acc']*100)
        target = (51, 92) if key == 'cnn' else (65, 52)
        axes[1].annotate(f"{'CNN' if key == 'cnn' else 'MLP'} best: epoch {best['epoch']}\n"
                         f"Validation accuracy: {best['val_acc']:.2%}",
                         xy=xy, xytext=target, color=color, fontsize=10,
                         arrowprops=dict(arrowstyle='-', color=color, lw=1.1),
                         bbox=dict(boxstyle='round,pad=.4', fc='white', ec=color, alpha=.95))
        axes[0].annotate(f"Epoch {best['epoch']}",
                         (best['epoch'], best['train_loss']),
                         xytext=(-65, 18) if key == 'mlp' else (12, 26),
                         textcoords='offset points', color=color, fontsize=10,
                         arrowprops=dict(arrowstyle='-', color=color))
        print(f"{label}: best epoch {best['epoch']}, validation {best['val_acc']:.2%}")
    for ax in axes:
        ax.set_xlim(0, 103)
        ax.set_xticks([1, 20, 40, 60, 80, 100])
        ax.set_xlabel('Epoch')
        ax.grid(axis='y', color='#E4E9EF', linewidth=.8)
        ax.set_axisbelow(True)
    axes[0].set_title('Training Loss', loc='left', fontweight='bold', pad=14)
    axes[0].set_ylabel('Cross-entropy loss')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc='upper right', fontsize=10)
    axes[1].set_title('Validation Accuracy', loc='left', fontweight='bold', pad=14)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0, 100)
    axes[1].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    fig.text(.07, .085, 'Stars and dotted lines mark the epoch with highest validation accuracy for each model.',
             fontsize=10, color='#526170')
    fig.text(.07, .045, 'Loss markers use the same selected epochs. Curves show recorded values without smoothing.',
             fontsize=10, color='#526170')
    for suffix in ['png', 'svg', 'pdf']:
        path = output / f'training_comparison.{suffix}'
        fig.savefig(path, dpi=220, facecolor='white')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
