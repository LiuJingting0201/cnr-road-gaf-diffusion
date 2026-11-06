"""Plot matched physical fidelity, paired effects and within-request diversity."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("B", "P", "S", "V", "SV", "SH", "E")
NAMES = {
    "B": "B  Denoising alone", "P": "P  Class centroids",
    "S": "S  Spatial GAF", "V": "V  Speed GAF",
    "SV": "SV  Paired GAFs", "SH": "SH  Shuffled pairs",
    "E": "E  Empirical transfer",
}
CONTRASTS = ("SV-B", "S-B", "V-B", "P-B", "SV-SH", "SV-E")
COLORS = {method: "#7b8590" for method in METHODS}
COLORS.update(B="#31688e", SV="#ba5a24", E="#786092")


def read(path):
    return json.loads(path.read_text())


def format_axes(axis):
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", color="#e3e7ea", linewidth=.7)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)


def main(root, output):
    result = read(root / "results.json")
    probes = read(root / "probes.json")
    checks = read(root / "checks.json")
    columns = result["columns"]
    metrics = [
        ("longest_steady_W1_s", "Longest steady interval", "Wasserstein distance (s)"),
        ("duration_W1_s", "Trip duration", "Wasserstein distance (s)"),
        ("lag4_W1_mps", "Four-second speed change", "Wasserstein distance (m/s)"),
    ]
    for method in METHODS:
        if method not in result["arms"]:
            raise ValueError(f"Missing frozen method {method}")
    values = np.asarray([result["arms"][method]["macro"] for method in METHODS])
    if not np.isfinite(values).all():
        raise ValueError("Physical metrics must all be finite")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 7.6),
                               gridspec_kw={"height_ratios": [1, 1.06]})
    figure.subplots_adjust(left=.19, right=.985, top=.92, bottom=.17, hspace=.53, wspace=.28)
    positions = np.arange(len(METHODS))
    for column, (key, title, unit) in enumerate(metrics):
        axis = axes[0, column]
        index = columns.index(key)
        for position, method in enumerate(METHODS):
            value = values[position, index]
            axis.scatter(value, position, color=COLORS[method], s=40, zorder=3)
            axis.annotate(f"{value:.2f}" if column < 2 else f"{value:.3f}",
                          (value, position), xytext=(7, 0), textcoords="offset points",
                          va="center", fontsize=9, color=COLORS[method])
        axis.set_yticks(positions, [NAMES[method] for method in METHODS] if column == 0 else [""] * len(METHODS))
        axis.set_ylim(len(METHODS) - .4, -.6)
        axis.set_xlim(0, max(values[:, index].max() * 1.37, 1e-5))
        axis.set_title(title, pad=11)
        axis.set_xlabel(unit)
        format_axes(axis)
    for column, (key, title, _) in enumerate(metrics[:2]):
        axis = axes[1, column]
        index = columns.index(key)
        limits = [0.]
        for row, contrast in enumerate(CONTRASTS):
            table = result["differences"][contrast]["all"]
            estimate = table["delta"][index]
            lower, upper = table["ci"][index]
            if not np.isfinite([estimate, lower, upper]).all() or lower > upper:
                raise ValueError(f"Invalid interval for {contrast} / {key}")
            color = COLORS["SV"] if contrast == "SV-B" else "#7b8590"
            # Bootstrap percentile bounds need not contain the point estimate.
            axis.plot([lower, upper], [row, row], color=color, linewidth=1.8)
            axis.plot([lower, lower], [row - .09, row + .09], color=color, linewidth=1)
            axis.plot([upper, upper], [row - .09, row + .09], color=color, linewidth=1)
            axis.scatter(estimate, row, color=color, s=29, zorder=3)
            limits.extend([lower, upper, estimate])
        width = max(max(limits) - min(limits), .01)
        axis.set_xlim(min(limits) - .1 * width, max(limits) + .1 * width)
        axis.axvline(0, color="#303a43", linewidth=.8, linestyle="--")
        axis.set_ylim(len(CONTRASTS) - .4, -.6)
        labels = ["SV - B (primary)", "S - B", "V - B", "P - B", "SV - SH", "SV - E"]
        axis.set_yticks(range(len(CONTRASTS)), labels if column == 0 else [""] * len(CONTRASTS))
        axis.set_title(title + " difference", pad=11)
        axis.set_xlabel("Change in W1 (s), lower is better")
        format_axes(axis)
    axis = axes[1, 2]
    baseline = np.asarray(result["arms"]["B"]["diversity_macro"])
    if np.any(baseline <= 0):
        raise ValueError("Baseline diversity must be positive for a ratio")
    for index, (name, color, offset) in enumerate([
            ("Speed profiles", "#31688e", -.12), ("Duration", "#786092", .12)]):
        ratio = np.asarray([result["arms"][method]["diversity_macro"][index]
                            for method in METHODS]) / baseline[index]
        axis.scatter(ratio, positions + offset, color=color, s=25, label=name)
    axis.set_yticks(positions, METHODS)
    axis.set_ylim(len(METHODS) - .4, -.6)
    axis.axvline(1, color="#303a43", linewidth=.8, linestyle="--")
    axis.set_xlim(left=0)
    axis.set_title("Within-request diversity", pad=11)
    axis.set_xlabel("Ratio to denoising alone")
    axis.legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=1,
                fontsize=8, frameon=False, handletextpad=.35)
    format_axes(axis)
    caption = (f"Same map, {result['episodes']} new traffic episodes, {result['requests']} requests, "
               "two model seeds. Physical distances use all candidates.\n"
               "Intervals resample complete episodes and condition on the fitted models. "
               "Diversity measures variation across draws for each request.")
    figure.text(.19, .035, caption, fontsize=9, va="bottom", linespacing=1.5, color="#414a52")
    reference = checks["arms"]["reference"]
    provenance = {
        "spatial_reference_macro_f1": reference["spatial"].get("macro_f1"),
        "speed_reference_macro_f1": reference["speed"].get("macro_f1"),
        "joint_monitor_probe_macro_f1": probes["full8"]["methods"]["joint"]["scores"]["macro_f1"],
        "scope": checks["reader_independence"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata={"Title": "Paired GAF representation supervision for complete-trip diffusion",
                                     "Subject": json.dumps(provenance), "Creator": "g28_plot.py"})
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Directory with results, probes and checks JSON")
    parser.add_argument("--output", type=Path, default=Path("thesis/figures/paired-representations.pdf"))
    args = parser.parse_args()
    main(args.root, args.output)
