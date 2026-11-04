"""Final-report figures only; no sampling, tuning or alteration of locked results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

root = Path('docs/results/final-test-e7')
data = json.loads((root / 'summary.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
for ax, (key, title, unit) in zip(axes, [('speed_w1_mps', 'Speed', 'm/s'),
    ('acceleration_w1_mps2', 'Acceleration', 'm/s²'), ('travel_w1_m', 'Travel distance', 'm')]):
    r = data['paired_w1'][key]; lo, hi = r['percentile95']; point = r['empirical_minus_diffusion']
    ax.axvline(0, color='#777777', ls='--', lw=1)
    ax.plot([lo, hi], [0, 0], color='#2878a5', lw=3); ax.plot(point, 0, 'o', color='#2878a5', ms=8)
    span = max(hi, 0) - min(lo, 0)
    ax.set(xlim=(min(lo, 0)-.15*span, max(hi, 0)+.15*span), ylim=(-.5, .5),
           yticks=[], title=title, xlabel=f'Empirical minus diffusion W1 ({unit})')
    ax.text(.5, .83, f'{point:.3f} [{lo:.3f}, {hi:.3f}]', transform=ax.transAxes, ha='center', fontsize=10)
    ax.spines[['top', 'left', 'right']].set_visible(False)
fig.suptitle('Locked final test: paired vehicle uncertainty', fontsize=13)
fig.text(.5, .02, 'Positive favors diffusion. 2,000 paired resamples; fixed models and pools; no multiplicity adjustment.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, .07, 1, .92))
for ext in ['png', 'pdf']: fig.savefig(root / f'paired_w1.{ext}', dpi=160)
plt.close(fig)

groups = ['repeat', 'timewarp', 'synthetic_random', 'synthetic_gaf', 'synthetic_tcn']
labels = ['Repeated real', 'Time warp', 'Random synthetic', 'GAF-selected synthetic', 'TCN-selected synthetic']
colors = ['#777777', '#777777', '#4477AA', '#EE7733', '#228833']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for ax, arch in zip(axes, ['tcn', 'gru']):
    for i, (group, color) in enumerate(zip(groups, colors)):
        d = data['downstream']['models'][arch][group]['paired_delta_f1']['real_only']
        lo, hi = [x*100 for x in d['percentile95']]
        ax.plot([lo, hi], [i, i], color=color, lw=2); ax.plot(d['point']*100, i, 'o', color=color)
    ax.axvline(0, color='black', ls='--', lw=1)
    ax.set(title=arch.upper()+' downstream learner', yticks=range(len(labels)), yticklabels=labels,
           xlabel='Macro-F1 change vs real-only (percentage points)')
    ax.grid(axis='x', alpha=.2)
axes[0].invert_yaxis()
fig.suptitle('Locked final test: synthetic-data utility on real trajectories')
fig.text(.5, .02, 'Paired 95% vehicle-bootstrap intervals; three fixed learners, one augmentation generator and fixed training pool.', ha='center', fontsize=8)
fig.tight_layout(rect=(0, .06, 1, .94))
for ext in ['png', 'pdf']: fig.savefig(root / f'utility.{ext}', dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 5.8)); ax.set(xlim=(0, 12), ylim=(0, 5.8)); ax.axis('off')
def box(x, y, text, color='#e7f0f6', w=2.5, h=1.):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.06', facecolor=color, edgecolor='#60717c'))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=10)
def arrow(a, b): ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '->', 'color': '#40515a', 'lw': 1.5})
box(.2, 4.0, 'SUMO road graph\nTraining-only route prior')
box(3.2, 4.0, 'Class + start\nIndependent planned paths')
box(6.2, 4.0, 'Trained conditional diffusion\n511 log arc increments')
box(9.2, 4.0, 'Shared path decoder\nK=8 coordinate trajectories')
arrow((2.75, 4.5), (3.12, 4.5)); arrow((5.75, 4.5), (6.12, 4.5)); arrow((8.75, 4.5), (9.12, 4.5))
box(6.2, 2.0, 'Inherited Hilbert encoding\nGASF + GADF + MTF\nSelf-trained CNN score', '#fff0dc', h=1.2)
box(9.2, 2.0, 'Shared validity mask\nSame-budget selection', '#fff0dc', h=1.2)
arrow((10.45, 3.94), (10.45, 3.28)); arrow((9.25, 4.), (8.25, 3.28)); arrow((8.75, 2.6), (9.12, 2.6))
box(9.2, .35, 'Synthetic timed coordinates\ntime, id, x, y, type', '#e8f1e5')
arrow((10.45, 1.94), (10.45, 1.42))
ax.text(.2, 2.6, 'Controls: random / GAF / raw TCN / shuffled target\nSame saved candidate pools and K\nNo image inversion; no gradient through Hilbert', fontsize=10, va='center')
ax.text(.2, .7, 'Evaluate geometry, dynamics, diversity and real held-out utility.\nRoad validity comes mainly from the decoder; CNN scores are not utility.', fontsize=10, va='center')
fig.suptitle('Implemented framework and inherited GAF connection', fontsize=14)
fig.tight_layout(rect=(0, 0, 1, .96))
for ext in ['png', 'pdf']: fig.savefig(root / f'method.{ext}', dpi=160)
plt.close(fig)
