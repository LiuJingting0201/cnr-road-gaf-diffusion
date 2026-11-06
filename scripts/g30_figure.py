"""Three-source diagnosis; all sources and all prespecified motion endpoints."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("g30-data")
DOMAINS = ("cnr", "sumo", "native")
NAMES = ("Original CNR", "Random-route SUMO", "Native MoST demand")
COLORS = ("#246B9D", "#D56A35", "#278368")


def run():
    summary = json.loads((ROOT/"summary.json").read_text())
    structure = json.loads((ROOT/"structure.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for column, (name, title) in enumerate((("spatial", "Geometry only"), ("motion", "Motion only"))):
        scores = [structure["domains"][domain]["readouts"][name]["nonlinear"]["val"]["macro_f1"]*100
                  for domain in DOMAINS]
        bars = axes[column].bar(np.arange(3), scores, color=COLORS)
        axes[column].bar_label(bars, fmt="%.1f", padding=4)
        axes[column].set_title(title+" / fixed feature model")
    scores = [summary["models"][d]["mean"]["validation"]["macro_f1"]*100 for d in DOMAINS]
    bars = axes[2].bar(np.arange(3), scores, color=COLORS)
    axes[2].bar_label(bars, fmt="%.1f", padding=4)
    for i,d in enumerate(DOMAINS):
        axes[2].scatter([i-.07,i+.07], [summary["models"][d]["by_seed"][str(s)]["validation"]["macro_f1"]*100
                                     for s in (42,43)], color="black", s=18, zorder=3)
    axes[2].set_title("Inherited GAF / ResNet18, two seeds")
    for axis in axes:
        axis.set_xticks(np.arange(3), NAMES, rotation=20, ha="right")
        axis.set_ylim(0,100)
        axis.set_ylabel("Validation macro-F1 (%)")
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("G30: native demand versus random routes on the same road network")
    fig.supxlabel("Same six classes and class counts; source entity and episode partitions preserved.\n"
                  "Native MoST repeats public demand with fresh simulation randomness; this is not a new-route test.", fontsize=9)
    fig.savefig(ROOT/"source-comparison.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
