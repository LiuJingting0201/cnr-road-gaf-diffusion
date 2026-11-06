"""Paper-sized figures of completed G33 results; no scientific re-evaluation.

Reads the same aggregate receipts and fixed first cases as g33_plots.py. Writes
to a new directory only. No fitting, scoring, selection or test access occurs.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import g33_evaluate
from g33_evaluate import CLASSES, digest, read_pool, signals


COLORS = {"empirical": "#6b7280", "relative_257": "#c58125", "native_0_25s": "#176b94",
          "Uniform": "#6b7280", "S": "#9665af", "V": "#15836a", "SV": "#cb603b"}
FAMILIES = ("empirical", "relative_257", "native_0_25s")
SEEDS = ((-.065, 42, "o"), (.065, 43, "s"))
NAMES = ["Bus", "Motorcycle", "Private car", "Bicycle", "Taxi / on-demand"]
SUMMARY_METRICS = [
    ("mean_absolute_acceleration_mps2", "Mean absolute acceleration", "m/s²"),
    ("stopped_fraction", "Stopped fraction", "fraction"),
    ("longest_stop_s", "Longest stop", "s"),
    ("longest_steady_s", "Longest steady movement", "s"),
    ("mean_absolute_lag4_speed_change_mps", "Four-second speed change", "m/s"),
    ("speed_p95_mps", "95th-percentile speed", "m/s"),
]
NATIVE_METRICS = [("native_speed_w1_mps", "Native speed", "m/s"),
                  ("native_acceleration_w1_mps2", "Native acceleration", "m/s²")]
SELECTION_METRICS = [("duration_s", "Trip duration", "s"), *NATIVE_METRICS,
                     ("longest_stop_s", "Longest stop", "s")]


def completed(path, inputs):
    result = json.loads(path.read_text())
    if result.get("complete") is not True:
        raise ValueError("Completed receipt required: " + str(path))
    inputs[str(path)] = digest(path)
    return result


def metric_value(row, key):
    if key.startswith("native_"):
        values = [row["paired_by_class"][c][key] for c in CLASSES]
        support_key = key.replace("_w1_mps2", "_supported_requests").replace("_w1_mps", "_supported_requests")
        support = [row["paired_by_class"][c][support_key] for c in CLASSES]
        value = float(np.mean(values)) if all(v is not None for v in values) else None
    else:
        value = row["paired_macro_w1"][key]
        support = [row["paired_by_class"][c]["physical"][key]["paired_supported_requests"] for c in CLASSES]
    if value is not None and (not np.isfinite(value) or value < 0):
        raise ValueError("Invalid saved metric: " + key)
    return value, support


def save_figure(fig, output, name):
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def seed_legend(fig, y):
    handles = [Line2D([], [], color="#374151", marker=marker, linestyle="none", markersize=6,
                      label=f"Seed {seed}") for _, seed, marker in SEEDS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=2, frameon=False, fontsize=10)


def point(ax, x, y, marker, color):
    if y is None:
        ax.text(x, .04, "NA", transform=ax.get_xaxis_transform(), ha="center", fontsize=9.5, color=color)
    else:
        ax.scatter(x, y, marker=marker, s=40, color=color, zorder=3)


def positive_axis(ax, title, unit):
    ax.set(title=title, ylabel=f"Macro W1 ({unit})", xticks=range(3),
           xticklabels=["Empirical", "257 points", "0.25 s"])
    ax.set_ylim(0, max(.001, ax.get_ylim()[1]) * 1.10)
    ax.grid(axis="y", alpha=.22)


def aggregate(root, output, inputs):
    rows, selected = {}, {}
    for family in FAMILIES:
        for seed in (42, 43):
            row = completed(root / "evaluation" / f"{family}-{seed}" / "complete.json", inputs)
            if row["classes"] != CLASSES or row["seed"] != seed or row["coverage"]["requested"] != 100:
                raise ValueError("Fixed evaluation identity or request denominator changed")
            rows[family, seed] = row
    for seed in (42, 43):
        for method in ("Uniform", "S", "V", "SV"):
            row = completed(root / "evaluation" / f"selected-native_0_25s-{seed}" / method / "complete.json", inputs)
            if (row["classes"] != CLASSES or row["seed"] != seed or row["selection_method"] != method
                    or row["coverage"]["requested"] != 100):
                raise ValueError("Fixed selected evaluation identity changed")
            selected[seed, method] = row
    all_rows = [*rows.values(), *selected.values()]
    for source in ("reference_receipt_sha256", "map_receipt_sha256", "training_dataset_sha256"):
        if len({row["sources"][source] for row in all_rows}) != 1:
            raise ValueError("Comparison input differs: " + source)

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.5))
    for ax, (key, title, unit) in zip(axes.flat, SUMMARY_METRICS):
        for i, family in enumerate(FAMILIES):
            for offset, seed, marker in SEEDS:
                value, _ = metric_value(rows[family, seed], key)
                point(ax, i + offset, value, marker, COLORS[family])
        positive_axis(ax, title, unit)
    fig.suptitle("Complete-trip synthesis: physical summaries", fontsize=12)
    seed_legend(fig, .025)
    fig.text(.5, .012, "Five-class macro; lower is better. One reference episode; no confidence intervals.",
             ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, .07, 1, .965), h_pad=2.0, w_pad=1.7)
    save_figure(fig, output, "generation-quality")

    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    for ax, (key, title, unit) in zip(axes, NATIVE_METRICS):
        for i, family in enumerate(FAMILIES):
            for offset, seed, marker in SEEDS:
                value, _ = metric_value(rows[family, seed], key)
                point(ax, i + offset, value, marker, COLORS[family])
        positive_axis(ax, title, unit)
    fig.suptitle("Motion at the original 0.25-second resolution", fontsize=12)
    seed_legend(fig, .065)
    fig.text(.5, .02, "Equal requests, then candidates, then physical-interval weights.\n"
             "Lower is better. One reference episode; no confidence intervals.", ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, .19, 1, .94), w_pad=2)
    save_figure(fig, output, "native-motion")

    fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))
    for ax, (key, title, unit) in zip(axes.flat, SELECTION_METRICS):
        ax.axhline(0, color="#374151", linewidth=.9)
        for i, method in enumerate(("S", "V", "SV")):
            for offset, seed, marker in SEEDS:
                value, support = metric_value(selected[seed, method], key)
                uniform, baseline_support = metric_value(selected[seed, "Uniform"], key)
                delta = value - uniform if value is not None and uniform is not None and support == baseline_support else None
                point(ax, i + offset, delta, marker, COLORS[method])
        ax.set(title=title, ylabel=f"Δ macro W1 ({unit})", xticks=range(3),
               xticklabels=["Spatial", "Velocity", "Joint"])
        ax.grid(axis="y", alpha=.22)
    fig.suptitle("GAF feedback: the same eight candidates", fontsize=12)
    seed_legend(fig, .04)
    fig.text(.5, .012, "Difference from Uniform; below zero is better.\n"
             "Independent physical measures; terminal selection only; no confidence intervals.", ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, .12, 1, .95), h_pad=2.1, w_pad=1.9)
    save_figure(fig, output, "gaf-physical-effects")
    return {f"{family}-{seed}": row["coverage"] for (family, seed), row in rows.items()}


def cases(root, output, inputs):
    pools = {}
    for method in ("Uniform", "S", "V", "SV"):
        folder = root / "selected/native_0_25s-42" / method
        _, pools[method] = read_pool(folder, "samples.npz")
        for name in ("complete.json", "samples.npz"):
            inputs[str(folder/name)] = digest(folder/name)
    folder = root / "reference-extracted"
    _, reference = read_pool(folder, "references.npz")
    for name in ("complete.json", "references.npz"):
        inputs[str(folder/name)] = digest(folder/name)
    lookup = {m: {str(i): j for j, i in enumerate(p["request_ids"])} for m, p in pools.items()}
    reference_lookup = {str(i): j for j, i in enumerate(reference["ids"])}
    common = set(reference_lookup).intersection(*(set(ids) for ids in lookup.values()))
    original = completed(root / "figures/complete.json", inputs)
    original_cases = {row["class"]: row for row in original["cases"]}
    saved = []

    def track(data, i):
        txy = data["txy"][data["offsets"][i]:data["offsets"][i+1]].copy()
        txy[:, 0] -= txy[0, 0]
        return txy

    for label, name in enumerate(NAMES):
        eligible = sorted(i for i in common if int(reference["labels"][reference_lookup[i]]) == label)
        if not eligible:
            if original_cases[CLASSES[label]]["supported"]:
                raise ValueError("Previously supported fixed case disappeared")
            saved.append({"class": CLASSES[label], "supported": False})
            continue
        ident = eligible[0]
        identity_hash = hashlib.sha256(ident.encode()).hexdigest()
        if original_cases[CLASSES[label]].get("request_sha256") != identity_hash:
            raise ValueError("Case differs from original fixed first-request figure")
        tracks = {"SUMO reference": track(reference, reference_lookup[ident]),
                  **{method: track(pool, lookup[method][ident]) for method, pool in pools.items()}}
        origin = tracks["SUMO reference"][0, 1:]
        with plt.rc_context({"font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
                             "xtick.labelsize": 9.5, "ytick.labelsize": 9.5}):
            fig = plt.figure(figsize=(7.2, 6.5))
            grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.08))
            axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
            for method, txy in tracks.items():
                _, speed, _, _, _, arc = signals(txy[:, 0], txy[:, 1:])
                color = "#111827" if method == "SUMO reference" else COLORS[method]
                width = 1.6 if method in ("SUMO reference", "V") else 1.0
                linestyle = "--" if method == "SUMO reference" else "-"
                label_text = {"SUMO reference": "SUMO", "Uniform": "Uniform", "S": "Spatial", "V": "Velocity", "SV": "Joint"}[method]
                axes[0].plot(txy[:, 1]-origin[0], txy[:, 2]-origin[1], color=color, lw=width, ls=linestyle, alpha=.8)
                axes[1].plot(txy[:, 0]/txy[-1, 0], arc/max(arc[-1], 1e-12), color=color, lw=width, ls=linestyle)
                axes[2].plot(np.arange(len(speed))+.5, speed, label=label_text, color=color, lw=width, ls=linestyle, alpha=.85)
            axes[0].set(xlabel="Relative east (m)", ylabel="Relative north (m)", title="Route and SUMO realization")
            axes[0].set_aspect("equal", adjustable="datalim")
            axes[1].set(xlabel="Fraction of trip duration", ylabel="Fraction of travelled distance", title="Full-trip progress")
            axes[2].set(xlabel="Physical elapsed time (s)", ylabel="One-second speed (m/s)", title="Full-trip motion")
            for ax in axes:
                ax.grid(alpha=.2)
            handles, labels = axes[2].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .064), ncol=5,
                       fontsize=9.5, frameon=False, handlelength=1.8, columnspacing=1.1)
            fig.suptitle(name + ": first eligible request, seed 42", fontsize=12)
            fig.text(.5, .012, "Fixed first request, not selected by quality. Different durations are allowed.\n"
                     "Speed: complete one-second intervals; final partial second omitted.", ha="center", fontsize=9.5)
            fig.tight_layout(rect=(0, .13, 1, .95), h_pad=1.9, w_pad=1.8)
            save_figure(fig, output, "case-" + CLASSES[label])
        saved.append({"class": CLASSES[label], "supported": True, "eligible_requests": len(eligible),
                      "request_sha256": identity_hash})
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("g33-data"))
    parser.add_argument("--output", type=Path, default=Path("g33-data/paper-figures"))
    args = parser.parse_args()
    inputs = {}
    completed(args.root / "evaluation-complete.json", inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5,
                         "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    coverage = aggregate(args.root, args.output, inputs)
    fixed_cases = cases(args.root, args.output, inputs)
    result = {"complete": True, "scope": "paper_layout_only_completed_G33_results_no_new_scientific_evaluation",
              "code_sha256": digest(__file__), "dependency_code_sha256": {
                  "g33_evaluate.py": digest(inspect.getsourcefile(g33_evaluate)),
                  Path(inspect.getsourcefile(signals)).name: digest(inspect.getsourcefile(signals))},
              "inputs_sha256": inputs, "coverage": coverage,
              "summary_metrics": [key for key, _, _ in SUMMARY_METRICS],
              "feedback_metrics": [key for key, _, _ in SELECTION_METRICS],
              "case_rule": "Unchanged first lexicographic paired request per class, seed42; matched original figure hashes",
              "cases": fixed_cases, "no_confidence_intervals": True,
              "figures_preserve_old_outputs": True,
              "artifacts": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                            for p in sorted(args.output.iterdir())}}
    (args.output / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "artifacts": len(result["artifacts"])}))


if __name__ == "__main__":
    main()
