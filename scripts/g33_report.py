"""Read-only aggregation of the fixed G33 result matrix; no scores or refits.

Reads only completed aggregate evaluation JSON. Does not read trajectories,
weights, reference arrays or old tests. Missing results remain explicitly pending.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


CLASSES = ["bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
METRICS = {
    "duration_s": "s", "arc_length_m": "m", "longest_steady_s": "s",
    "mean_absolute_lag4_speed_change_mps": "m/s", "mean_speed_mps": "m/s",
    "mean_absolute_acceleration_mps2": "m/s2", "stopped_fraction": "fraction",
    "speed_p95_mps": "m/s", "longest_stop_s": "s", "stop_episodes_per_minute": "episodes/min",
}
NATIVE = ("native_speed_w1_mps", "native_acceleration_w1_mps2")


def expected():
    rows = [(f"{arm}-{seed}", f"{arm}-{seed}/complete.json", "full_pool", seed)
            for arm in ("relative_257", "native_0_25s", "empirical") for seed in (42, 43)]
    rows.extend((f"native_0_25s-{seed}/{method}", f"selected-native_0_25s-{seed}/{method}/complete.json", method, seed)
                for seed in (42, 43) for method in ("Uniform", "S", "V", "SV"))
    return rows


def mean_or_none(values):
    return sum(values)/len(values) if values and all(v is not None for v in values) else None


def finite_or_none(value):
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))


def parse(name, receipt, kind, seed):
    if receipt.get("complete") is not True or receipt.get("classes") != CLASSES or receipt.get("seed") != seed:
        raise ValueError("Incomplete or wrong-scope evaluation: "+name)
    expected_family = "empirical" if name.startswith("empirical-") else "diffusion"
    expected_arm = "training_motion_transfer" if expected_family == "empirical" else name.split("-")[0]
    if receipt.get("family") != expected_family or receipt.get("arm") != expected_arm:
        raise ValueError("Evaluation family/arm does not match matrix slot: "+name)
    coverage = receipt["coverage"]
    if coverage["requested"] != 100:
        raise ValueError("100-request denominator changed: "+name)
    paired = coverage["paired_reference_and_generation"]
    if not 0 <= paired <= min(coverage["reference_complete"], coverage["requests_with_generated_candidates"], coverage["map_ready"]) <= 100:
        raise ValueError("Inconsistent coverage: "+name)
    row = {"name": name, "kind": kind, "family": receipt["family"], "arm": receipt["arm"], "seed": seed,
           **{key: coverage[key] for key in ("requested", "map_ready", "reference_complete", "requests_with_generated_candidates",
                                           "paired_reference_and_generation", "generated_candidates_all", "generated_candidates_paired")},
           "physical_w1": {}, "native_w1": {}, "support": {}, "per_class": {},
           "diversity": {key: val.get("mean") for key, val in receipt["within_request_diversity"].items()},
           "training_shape_nearest": {key: val.get("mean") for key, val in receipt["training_shape_nearest"].items()},
           "sources": receipt["sources"]}
    if kind != "full_pool" and receipt.get("selection_method") != kind:
        raise ValueError("Selected method label changed: "+name)
    for metric in METRICS:
        values, classes = [], []
        row["support"][metric] = {}
        for label in CLASSES:
            item = receipt["paired_by_class"][label]["physical"][metric]
            value = item["w1"]
            if not finite_or_none(value) or (value is not None and value < 0):
                raise ValueError("Invalid W1: "+name+"/"+metric)
            if value is not None:
                classes.append(label); values.append(value)
            row["support"][metric][label] = item["paired_supported_requests"]
        calculated = mean_or_none(values) if len(classes) == len(CLASSES) else None
        saved = receipt["paired_macro_w1"][metric]
        if (calculated is None) != (saved is None) or (saved is not None and not math.isclose(calculated, saved, rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError("Five-class macro does not match saved class values: "+name+"/"+metric)
        if classes != receipt["paired_macro_supported_classes"][metric]:
            raise ValueError("Saved support class list changed: "+name+"/"+metric)
        row["physical_w1"][metric] = saved
    for key in NATIVE:
        values = [receipt["paired_by_class"][label][key] for label in CLASSES]
        if not all(finite_or_none(v) and (v is None or v >= 0) for v in values):
            raise ValueError("Invalid native W1")
        row["native_w1"][key] = mean_or_none(values)
        supported_key = key.replace("_w1_mps2", "_supported_requests").replace("_w1_mps", "_supported_requests")
        row["support"][key] = {label: receipt["paired_by_class"][label][supported_key] for label in CLASSES}
    for label in CLASSES:
        values = receipt["paired_by_class"][label]
        row["per_class"][label] = {"reference_requests": values["reference_requests"],
                                   "generated_candidates": values["generated_candidates"],
                                   "physical_w1": {key: values["physical"][key]["w1"] for key in METRICS},
                                   "native_w1": {key: values[key] for key in NATIVE}}
    return row


def paired_differences(rows):
    by_name = {r["name"]: r for r in rows}
    pairs = []
    for seed in (42, 43):
        for a, b in ((f"native_0_25s-{seed}", f"relative_257-{seed}"),
                     (f"native_0_25s-{seed}", f"empirical-{seed}"),
                     (f"relative_257-{seed}", f"empirical-{seed}")):
            pairs.append((a, b))
        pairs.extend((f"native_0_25s-{seed}/{method}", f"native_0_25s-{seed}/Uniform") for method in ("S", "V", "SV"))
    answer = []
    for a, b in pairs:
        if a not in by_name or b not in by_name:
            continue
        left, right = by_name[a], by_name[b]
        values = {}
        for field, keys in (("physical_w1", METRICS), ("native_w1", NATIVE)):
            for key in keys:
                u, v = left[field][key], right[field][key]
                supported_counts_equal = left["support"][key] == right["support"][key]
                values[key] = {"delta_lower_is_better": u-v if u is not None and v is not None and supported_counts_equal else None,
                               "supported_request_counts_equal": supported_counts_equal}
        answer.append({"left": a, "right": b, "metrics": values,
                       "interpretation": "Descriptive difference; equal support counts do not alone verify exact per-metric request identity."})
    return answer


def csv_table(path, fieldnames, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def render_number(value):
    return "NA" if value is None else f"{value:.4f}"


def main(root, output, allow_partial):
    inputs, rows, missing = {}, [], []
    for name, relative, kind, seed in expected():
        path = root/relative
        if not path.exists():
            missing.append(relative); continue
        receipt = json.loads(path.read_text())
        if not receipt.get("complete"):
            missing.append(relative); continue
        rows.append(parse(name, receipt, kind, seed))
        inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if missing and not allow_partial:
        raise ValueError("Fixed result matrix incomplete; no report written. Missing: "+", ".join(missing))
    common_sources = {}
    for source in ("reference_receipt_sha256", "map_receipt_sha256", "training_dataset_sha256"):
        hashes = sorted({row["sources"][source] for row in rows})
        if len(hashes) > 1:
            raise ValueError("Comparison data source differs: "+source)
        common_sources[source] = hashes[0] if hashes else None
    output.mkdir(parents=True, exist_ok=False)
    difference = paired_differences(rows)
    invariance = []
    for seed in (42, 43):
        group = [r for r in rows if r["kind"] == "full_pool" and r["seed"] == seed]
        if len(group) == 3:
            values = [r["physical_w1"]["duration_s"] for r in group]
            equal_support = all(r["support"]["duration_s"] == group[0]["support"]["duration_s"] for r in group)
            invariance.append({"seed": seed, "supported_request_counts_equal": equal_support,
                "macro_duration_w1": values,
                "shared_prior_expected_invariance": max(values)-min(values) < 1e-8 if equal_support and all(v is not None for v in values) else None})
    summary = {"complete": not missing, "partial": bool(missing), "expected_evaluations": 14,
               "completed_evaluations": len(rows), "missing_evaluations": missing,
               "inputs_sha256": inputs, "common_sources": common_sources, "rows": rows,
               "differences": difference, "shared_duration_checks": invariance,
               "limits": ["All physical errors are descriptive on one new SUMO episode, shared map and public demand templates; no confidence intervals.",
                          "Every row preserves the 100-request denominator; paired quality covers the stated intersection only.",
                          "Five-class macro is NA when any class lacks support; unsupported metrics are never zero-filled.",
                          "Full-pool clocks and empirical transfer share a duration prior; duration W1 is not evidence of motion-diffusion gain.",
                          "Conditional route geometry and endpoint constraints are shared infrastructure, not learned road legality.",
                          "GAF differences compare selected data with Uniform from the same candidate pool, not classifier scores.",
                          "Equal support counts do not prove identical per-metric request IDs when partial support differs; no paired significance claim.",
                          "Nearest normalized-progress similarity is descriptive; it does not establish privacy or absence of memorization."]}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    fields = ["name", "kind", "seed", "requested", "paired_reference_and_generation", "generated_candidates_all", *METRICS, *NATIVE]
    csv_table(output/"macro.csv", fields, [{**{k: r[k] for k in fields if k in r}, **r["physical_w1"], **r["native_w1"]} for r in rows])
    class_rows = []
    for r in rows:
        for label in CLASSES:
            for key in list(METRICS)+list(NATIVE):
                field = "physical_w1" if key in METRICS else "native_w1"
                class_rows.append({"name": r["name"], "class": label, "metric": key,
                                   "value": r["per_class"][label][field][key], "supported_requests": r["support"][key][label]})
    csv_table(output/"per-class.csv", ["name", "class", "metric", "value", "supported_requests"], class_rows)
    text = ["# G33固定结果矩阵", "", "状态："+("部分结果，未完成项列于下方。" if missing else "14项固定评价全部完成。"), "",
            "所有W1越低越好；不合成为单一总分，不计算单场伪独立置信区间。表中NA表示缺失支持。", "",
            "| 方法/种子 | 原请求 | 可配对参考 | 生成条数 | 原生速度W1 m/s | 原生加速度W1 m/s² |", "|---|---:|---:|---:|---:|---:|"]
    text.extend(f"| {r['name']} | {r['requested']} | {r['paired_reference_and_generation']} | {r['generated_candidates_all']} | {render_number(r['native_w1'][NATIVE[0]])} | {render_number(r['native_w1'][NATIVE[1]])} |" for r in rows)
    groups = [list(METRICS)[:5], list(METRICS)[5:]]
    for i, metrics in enumerate(groups, 1):
        text.extend(["", f"## 十项物理指标：表{i}", "", "| 方法/种子 | "+" | ".join(f"{m} ({METRICS[m]})" for m in metrics)+" |", "|---|"+"---:|"*len(metrics)])
        text.extend("| "+r["name"]+" | "+" | ".join(render_number(r["physical_w1"][m]) for m in metrics)+" |" for r in rows)
    text.extend(["", "逐类全部指标及支持数见per-class.csv；方法差值、条件多样性与训练形状近邻见summary.json。", "",
                 "共享时长先验检查：", *[f"- seed{r['seed']}: {r['shared_prior_expected_invariance']}；三种全池时长W1={r['macro_duration_w1']}。" for r in invariance], "",
                 *["- "+limit for limit in summary["limits"]]])
    if missing:
        text.extend(["", "未完成项：", *["- "+name for name in missing]])
    (output/"TABLES.md").write_text("\n".join(text)+"\n")
    print(json.dumps({"complete": not missing, "completed": len(rows), "missing": len(missing)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=Path("g33-data/evaluation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    main(args.evaluation_root, args.output, args.allow_partial)
