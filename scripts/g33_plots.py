"""Descriptive G33 figures from completed receipts; no outcome-based case choice.

Run after the finite evaluation queue. Aggregate figures show both fixed model
seeds separately, not confidence intervals. Cases are the lexicographically
first paired request in each of the five classes, model seed42. No coordinates
or source identifiers are written to the public figure manifest.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from g33_evaluate import CLASSES, digest, read_pool, signals


METRICS = [
    ("duration_s", "Trip duration", "s"),
    ("arc_length_m", "Travel distance", "m"),
    ("mean_speed_mps", "Mean speed", "m/s"),
    ("speed_p95_mps", "95th-percentile speed", "m/s"),
    ("mean_absolute_acceleration_mps2", "Mean absolute acceleration", "m/s²"),
    ("stopped_fraction", "Stopped fraction", "fraction"),
    ("longest_stop_s", "Longest stop", "s"),
    ("stop_episodes_per_minute", "Stop frequency", "episodes/min"),
    ("longest_steady_s", "Longest steady motion", "s"),
    ("mean_absolute_lag4_speed_change_mps", "Four-second speed change", "m/s"),
]
COLORS = {"empirical": "#6b7280", "relative_257": "#c58125", "native_0_25s": "#176b94",
          "Uniform": "#6b7280", "S": "#9665af", "V": "#15836a", "SV": "#cb603b"}
NAMES = ["Bus", "Motorcycle", "Private car", "Bicycle", "Taxi / on-demand"]


def receipt(path):
    value = json.loads(path.read_text())
    if value.get("complete") is not True:
        raise ValueError("Completed receipt required: " + str(path))
    return value


def finish(fig, output, name):
    fig.savefig(output/(name+".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output/(name+".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def aggregate(root, output):
    evaluation = root/"evaluation"
    families = ("empirical", "relative_257", "native_0_25s")
    rows = {(name, seed): receipt(evaluation/f"{name}-{seed}"/"complete.json")
            for name in families for seed in (42, 43)}
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    for ax, (metric, title, unit) in zip(axes.flat, METRICS):
        for i, name in enumerate(families):
            for offset, seed, marker in ((-.07, 42, "o"), (.07, 43, "s")):
                y = rows[name, seed]["paired_macro_w1"][metric]
                if y is not None:
                    ax.scatter(i+offset, y, s=45, marker=marker, color=COLORS[name], zorder=3)
        ax.set(title=title, ylabel="W1 ("+unit+")", xticks=range(3),
               xticklabels=["Empirical", "257 bins", "0.25 s"])
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.22)
    fig.suptitle("Complete-trip synthesis: independent SUMO comparison", fontsize=17)
    fig.text(.5, .012, "Five-class macro W1; lower is better. Circles: seed 42; squares: seed 43. "
             "Shared duration prior. One reference episode; no confidence intervals.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95))
    finish(fig, output, "generation-quality")

    selected = {seed: receipt(evaluation/f"selected-native_0_25s-{seed}"/"complete.json") for seed in (42, 43)}
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    for ax, (metric, title, unit) in zip(axes.flat, METRICS):
        ax.axhline(0, color="#374151", linewidth=.9)
        for i, method in enumerate(("S", "V", "SV")):
            for offset, seed, marker in ((-.07, 42, "o"), (.07, 43, "s")):
                y = selected[seed]["macro_w1_difference_from_uniform_lower_is_better"][method][metric]
                if y is not None:
                    ax.scatter(i+offset, y, s=45, marker=marker, color=COLORS[method], zorder=3)
        ax.set(title=title, ylabel="Δ W1 ("+unit+")", xticks=range(3), xticklabels=["Spatial", "Speed", "Both"])
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.22)
    fig.suptitle("GAF feedback: same eight candidates, different selection", fontsize=17)
    fig.text(.5, .012, "Difference from Uniform; below zero is better. Circles: seed 42; squares: seed 43. "
             "Independent physical measures, not classifier confidence. Terminal feedback only.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95))
    finish(fig, output, "gaf-physical-effects")
    return {"evaluation_receipts": {f"{name}-{seed}": digest(evaluation/f"{name}-{seed}"/"complete.json")
                                    for name, seed in rows},
            "selection_evaluation_receipts": {str(seed): digest(evaluation/f"selected-native_0_25s-{seed}"/"complete.json")
                                               for seed in selected}}


def cases(root, output):
    pools = {}
    for method in ("Uniform", "S", "V", "SV"):
        _, pools[method] = read_pool(root/"selected/native_0_25s-42"/method, "samples.npz")
    _, reference = read_pool(root/"reference-extracted", "references.npz")
    lookup = {m: {str(i): j for j, i in enumerate(p["request_ids"])} for m, p in pools.items()}
    reference_lookup = {str(i): j for j, i in enumerate(reference["ids"])}
    common = set(reference_lookup).intersection(*(set(ids) for ids in lookup.values()))
    saved = []
    def track(data, i):
        txy = data["txy"][data["offsets"][i]:data["offsets"][i+1]].copy()
        txy[:, 0] -= txy[0, 0]
        return txy
    for label, name in enumerate(NAMES):
        eligible = sorted(i for i in common if int(reference["labels"][reference_lookup[i]]) == label)
        if not eligible:
            saved.append({"class": CLASSES[label], "supported": False})
            continue
        ident = eligible[0]
        tracks = {"SUMO reference": track(reference, reference_lookup[ident]),
                  **{m: track(p, lookup[m][ident]) for m, p in pools.items()}}
        origin = tracks["SUMO reference"][0, 1:]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1.15, 1.7]})
        for method, txy in tracks.items():
            _, speed, _, _, _, arc = signals(txy[:, 0], txy[:, 1:])
            color = "#111827" if method == "SUMO reference" else COLORS[method]
            width = 1.7 if method in ("SUMO reference", "V") else 1.0
            line = "--" if method == "SUMO reference" else "-"
            axes[0].plot(txy[:, 1]-origin[0], txy[:, 2]-origin[1], color=color, lw=width, ls=line, alpha=.8)
            axes[1].plot(txy[:, 0]/txy[-1, 0], arc/max(arc[-1], 1e-12), color=color, lw=width, ls=line)
            axes[2].plot(np.arange(len(speed))+.5, speed, label=method, color=color, lw=width, ls=line, alpha=.85)
        axes[0].set(xlabel="Relative east (m)", ylabel="Relative north (m)", title="Declared route / SUMO realization")
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[1].set(xlabel="Fraction of trip duration", ylabel="Fraction of travelled distance", title="Full-trip progress")
        axes[2].set(xlabel="Physical elapsed time (s)", ylabel="One-second speed (m/s)", title="Full-trip motion")
        for ax in axes:
            ax.grid(alpha=.2)
        axes[2].legend(fontsize=8, ncol=2)
        fig.suptitle(name+": first eligible request, seed 42", fontsize=15)
        fig.text(.5, .01, "Case selected by request order, not quality. Samples may have different durations; "
                 "this is synthesis, not pointwise prediction of the reference.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .94))
        finish(fig, output, "case-"+CLASSES[label])
        saved.append({"class": CLASSES[label], "supported": True, "eligible_requests": len(eligible),
                      "request_sha256": __import__("hashlib").sha256(ident.encode()).hexdigest()})
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("g33-data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt(args.root/"evaluation-complete.json")
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    sources = aggregate(args.root, args.output)
    fixed_cases = cases(args.root, args.output)
    result = {"complete": True, "scope": "descriptive G33 figures; no new fitting or selection",
              "code_sha256": digest(__file__), "sources": sources,
              "case_rule": "First lexicographic reference-and-four-selection paired request per class, seed42",
              "cases": fixed_cases,
              "artifacts": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                            for p in sorted(args.output.iterdir())}}
    (args.output/"complete.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"complete": True, "figures": len(result["artifacts"])}))


if __name__ == "__main__":
    main()
