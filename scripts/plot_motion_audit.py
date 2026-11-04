"""Export all three predeclared W1 effects, including intervals crossing zero."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('docs/results/motion-audit-e7')
rows = json.loads((root / 'summary.json').read_text())['paired_w1']
specs = [('speed_w1_mps', 'Speed', 'm/s'),
         ('acceleration_w1_mps2', 'Acceleration', 'm/s²'),
         ('travel_w1_m', 'Travel distance', 'm')]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
for ax, (key, name, unit) in zip(axes, specs):
    row = rows[key]; lo, hi = row['paired_percentile95']
    value = row['empirical_minus_diffusion']
    ax.axvline(0, color='#808080', linestyle='--', lw=1)
    ax.plot([lo, hi], [0, 0], color='#2878a5', lw=3)
    ax.plot(value, 0, 'o', color='#2878a5', markersize=8)
    ax.set_title(name)
    ax.set_yticks([])
    ax.set_xlabel(f'Empirical minus diffusion W1 ({unit})')
    span = max(hi, 0) - min(lo, 0)
    ax.set_xlim(min(lo, 0) - .15 * span, max(hi, 0) + .15 * span)
    ax.set_ylim(-.5, .5)
    ax.text(.5, .83, f'{value:.3f} [{lo:.3f}, {hi:.3f}]', transform=ax.transAxes,
            ha='center', fontsize=10)
    ax.spines[['top', 'left', 'right']].set_visible(False)
fig.suptitle('Validation: paired vehicle uncertainty on fixed candidate pools', fontsize=13)
fig.text(.5, .02, 'Positive favors diffusion. 2,000 paired class-stratified resamples; fixed models/pools; no multiplicity adjustment.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .07, 1, .92))
fig.savefig(root / 'paired_w1.png', dpi=160)
fig.savefig(root / 'paired_w1.pdf')
plt.close(fig)
