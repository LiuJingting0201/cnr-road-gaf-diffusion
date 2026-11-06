"""Plot every prespecified primary source/encoding arm, without selected cases."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(root, output):
    structure = json.loads((root / "structure.json").read_text())
    domains = ("cnr", "sumo")
    colors = ("#246B9D", "#D56A35")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for i, domain in enumerate(domains):
        values = [structure["domains"][domain]["readouts"][task]["nonlinear"]["val"]["macro_f1"] * 100
                  for task in ("spatial", "motion")]
        bars = axes[0].bar(np.arange(2)+(i-.5)*.32, values, width=.32, color=colors[i],
                          label=("Original CNR", "New random-route SUMO")[i])
        axes[0].bar_label(bars, fmt="%.1f", padding=3)
        arms = ("xy", "grid_xy", "native_h", "squeezed_h", "unit_h", "gaf")
        scores = []
        for arm in arms:
            values = [json.loads((root/"models"/f"{domain}-full-{arm}-{seed}"/"complete.json").read_text())
                      ["validation"]["macro_f1"] * 100 for seed in (42, 43)]
            scores.append(values)
        scores = np.array(scores)
        axes[1].plot(np.arange(6), scores.mean(1), color=colors[i], marker="o",
                     label=("Original CNR", "New random-route SUMO")[i])
        axes[1].scatter(np.repeat(np.arange(6), 2), scores.flatten(), s=15, color=colors[i], alpha=.5)
        for x, y in enumerate(scores.mean(1)):
            axes[1].annotate(f"{y:.1f}", (x, y), xytext=(0, 8 if i == 0 else -16),
                             textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_xticks([0, 1], ["Geometry only", "Motion only"])
    axes[0].set_title("Independent fixed feature classifiers")
    axes[1].set_xticks(np.arange(6), ["XY", "Grid XY", "Native H", "Squeezed H", "Unit H", "GAF RGB"], rotation=20)
    axes[1].set_title("Encoding-stage classifiers (two seeds)")
    for axis in axes:
        axis.set_ylim(0, 100)
        axis.set_ylabel("Validation macro-F1 (%)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("G29: same six classes and class counts; 1,983 training / 257 validation IDs", fontsize=13)
    fig.supxlabel("Development splits only. CNR: ID holdout; new SUMO: episode holdout.\n"
                  "Stage scores measure learnability; classifier families differ at GAF RGB.", fontsize=9)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/"source-and-encoding.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("g29-data"))
    parser.add_argument("--output", type=Path, default=Path("g29-data"))
    args = parser.parse_args()
    main(args.root, args.output)
