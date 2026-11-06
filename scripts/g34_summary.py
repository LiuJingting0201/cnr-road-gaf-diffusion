"""Read-only G34 mechanism checks and ten-evaluation descriptive report.

Trajectory arrays are compared in place; only aggregate assertions and hashes
are written. Old/new Uniform must be bitwise identical and all eight candidates
and all four selections must use the predetermined same per-request duration.
Never silently omit a failed equality check or unsuccessful slot.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from g33_evaluate import completed, digest, read_pool, save
from g33_feedback import METHODS, permutation
from g33_report import CLASSES, METRICS, NATIVE, csv_table, parse, render_number
from g34_fixed_duration import DURATION_SOURCE, EXPERIMENT_PHASE, K


def completed_json(path):
    value = json.loads(path.read_text())
    if value.get("complete") is not True:
        raise ValueError("Completed aggregate required: "+str(path))
    return value


def lookup(pool):
    ids = pool["ids"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate pool candidate ID")
    return {ident: i for i, ident in enumerate(ids)}


def track(pool, index):
    return pool["txy"][pool["offsets"][index]:pool["offsets"][index+1]]


def check_arrays(old, new, plans, selected, expected_requests):
    old_index, new_index = lookup(old), lookup(new)
    if len(plans) != expected_requests or len({p["request_id"] for p in plans}) != expected_requests:
        raise ValueError("Predetermined duration plan coverage differs")
    expected_ids = {f"{p['request_id']}/candidate-{k:02d}" for p in plans for k in range(K)}
    if set(old_index) != expected_ids or set(new_index) != expected_ids:
        raise ValueError("Old/new complete K8 support differs; missing slots cannot be discarded")
    max_error, selected_total = 0., 0
    for plan in plans:
        ident = plan["request_id"]
        uniform = f"{ident}/candidate-{int(permutation(ident, K)[0]):02d}"
        if plan["duration_source_candidate_id"] != uniform or plan["uniform_slot"] != int(permutation(ident, K)[0]):
            raise ValueError("Duration source is not the predetermined Uniform slot")
        a, b = track(old, old_index[uniform]), track(new, new_index[uniform])
        if a.shape != b.shape:
            raise ValueError("Old/new Uniform trajectory shapes differ")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite saved Uniform trajectory")
        max_error = max(max_error, float(np.abs(a-b).max()))
        if not np.array_equal(a, b):
            raise ValueError(f"Old/new Uniform trajectory is not bitwise identical; max_abs_error={max_error}")
        if a[0, 0] != 0 or float(a[-1, 0]) != plan["duration"]:
            raise ValueError("Shared duration differs from G33 Uniform duration")
        for k in range(K):
            name = f"{ident}/candidate-{k:02d}"
            index = new_index[name]
            row = track(new, index)
            if (str(new["request_ids"][index]) != ident or int(new["labels"][index]) != plan["label"]
                    or not np.isfinite(row).all() or row[0, 0] != 0 or np.any(np.diff(row[:, 0]) <= 0)
                    or float(row[-1, 0]) != plan["duration"]):
                raise ValueError("New candidate class/clock/shared duration differs")
    plans_by_id = {p["request_id"]: p for p in plans}
    for method in METHODS:
        pool = selected[method]
        lookup(pool)
        request_ids = pool["request_ids"].astype(str).tolist()
        if len(request_ids) != expected_requests or set(request_ids) != set(plans_by_id):
            raise ValueError("Selected request support differs; failures cannot be omitted")
        for i, (name, ident) in enumerate(zip(pool["ids"].astype(str), request_ids)):
            if name not in new_index or str(new["request_ids"][new_index[name]]) != ident:
                raise ValueError("Selection points to another pool/request")
            plan = plans_by_id[ident]
            row = track(pool, i)
            if not np.array_equal(row, track(new, new_index[name])):
                raise ValueError("Selected trajectory differs from source pool")
            if (row[0, 0] != 0 or float(row[-1, 0]) != plan["duration"]
                    or int(pool["labels"][i]) != plan["label"]):
                raise ValueError("Selected duration/class differs from old Uniform")
            if method == "Uniform" and name != plan["duration_source_candidate_id"]:
                raise ValueError("G34 Uniform changed its predetermined candidate ID")
            selected_total += 1
    return {"requests": expected_requests, "all_pool_candidates_checked": expected_requests*K,
            "old_new_Uniform_bitwise_equal": True, "old_new_Uniform_max_absolute_txy_error": max_error,
            "old_new_Uniform_requests_checked": expected_requests,
            "all_pool_duration_equal_to_G33_Uniform": True,
            "all_four_selected_durations_equal_to_G33_Uniform": True, "selected_trajectories_checked": selected_total}


def verify_seed(root, g33, seed):
    name = f"native_0_25s-{seed}"
    new_folder, old_folder = root/"pools"/name, g33/"pools"/name
    new_receipt, new = read_pool(new_folder, "samples.npz")
    completed(new_folder, ["duration-plan.json", "candidate-status.json"])
    old_receipt, old = read_pool(old_folder, "samples.npz")
    if (new_receipt.get("experiment_phase") != EXPERIMENT_PHASE
            or new_receipt.get("sampled_duration_source") != DURATION_SOURCE
            or new_receipt.get("g33_baseline_receipt_sha256") != digest(old_folder/"complete.json")
            or new_receipt.get("reference_data_opened") is not False
            or new_receipt.get("teacher_scores_opened") is not False):
        raise ValueError("G34 source/provenance declaration differs")
    for key in ("request_sha256", "map_receipt_sha256", "training_model_receipt_sha256", "duration_prior_sha256",
                "classes", "arm", "seed", "requested_conditions", "map_ready_conditions", "requested_candidates_per_condition"):
        if new_receipt[key] != old_receipt[key]:
            raise ValueError("Old/new fixed contract changed: "+key)
    if new_receipt["seed"] != seed or new_receipt["requested_conditions"] != 100 or new_receipt["requested_candidates_per_condition"] != K:
        raise ValueError("Protocol seed/budget changed")
    planned = json.loads((new_folder/"duration-plan.json").read_text())
    plans = planned["rows"]
    if planned["sampled_duration_source"] != DURATION_SOURCE:
        raise ValueError("Unexpected duration plan source")
    statuses = json.loads((new_folder/"candidate-status.json").read_text())["rows"]
    if (len(statuses) != len(plans)*K or any(r["status"] != "generated" for r in statuses)
            or {r["id"] for r in statuses} != set(new["ids"].astype(str))):
        raise ValueError("Unsuccessful generated slots prevent complete paired diagnostic; retain failure")
    selection = completed(root/"selected"/name, ["selection-status.json", "encoding-status.json", "scores.npz"])
    if selection["sources"]["generation_receipt_sha256"] != digest(new_folder/"complete.json"):
        raise ValueError("Selection consumed a different candidate pool")
    old_selection = completed_json(g33/"selected"/name/"complete.json")
    if selection["sources"]["teacher_receipts"] != old_selection["sources"]["teacher_receipts"]:
        raise ValueError("G33/G34 teachers differ")
    selected, source_hashes = {}, {}
    for method in METHODS:
        folder = root/"selected"/name/method
        receipt, selected[method] = read_pool(folder, "samples.npz")
        if (receipt["selection_method"] != method or receipt["source_pool_receipt_sha256"] != digest(new_folder/"complete.json")
                or receipt["request_sha256"] != new_receipt["request_sha256"]):
            raise ValueError("Selection receipt does not link this fixed-duration pool")
        source_hashes[method] = digest(folder/"complete.json")
    audit = check_arrays(old, new, plans, selected, new_receipt["map_ready_conditions"])
    audit.update(seed=seed, sources={"old_pool_receipt": digest(old_folder/"complete.json"),
                                   "new_pool_receipt": digest(new_folder/"complete.json"),
                                   "duration_plan": digest(new_folder/"duration-plan.json"),
                                   "old_samples": digest(old_folder/"samples.npz"), "new_samples": digest(new_folder/"samples.npz"),
                                   "selection_receipts": source_hashes})
    return audit


def summary(root, g33, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        audits = [verify_seed(root, g33, seed) for seed in (42, 43)]
        rows, inputs, old_rows = [], {}, []
        for seed in (42, 43):
            name = f"native_0_25s-{seed}"
            for method in ("full_pool", *METHODS):
                relative = (Path(name)/"complete.json" if method == "full_pool"
                            else Path(f"selected-{name}")/method/"complete.json")
                rowname = name if method == "full_pool" else name+"/"+method
                path = root/"evaluation"/relative
                receipt = completed_json(path)
                pool_receipt = (root/"pools"/name/"complete.json" if method == "full_pool"
                                else root/"selected"/name/method/"complete.json")
                if receipt["sources"]["generation_receipt_sha256"] != digest(pool_receipt):
                    raise ValueError("Evaluation source differs from fixed-duration pool/selection")
                rows.append(parse(rowname, receipt, method, seed))
                inputs[str(relative)] = digest(path)
                oldpath = g33/"evaluation"/relative
                old_rows.append(parse(rowname, completed_json(oldpath), method, seed))
                inputs["G33/"+str(relative)] = digest(oldpath)
        sources = {}
        for key in ("reference_receipt_sha256", "map_receipt_sha256", "training_dataset_sha256"):
            values = {r["sources"][key] for r in rows+old_rows}
            if len(values) != 1:
                raise ValueError("G33/G34 evaluation source differs: "+key)
            sources[key] = values.pop()
        by_name = {r["name"]: r for r in rows}
        old_by_name = {r["name"]: r for r in old_rows}
        differences = []
        for seed in (42, 43):
            name = f"native_0_25s-{seed}"
            uniform = by_name[name+"/Uniform"]
            old_uniform = old_by_name[name+"/Uniform"]
            for field in ("physical_w1", "native_w1"):
                if uniform[field] != old_uniform[field] or uniform["support"] != old_uniform["support"]:
                    raise ValueError("Bitwise-equal old/new Uniform has different evaluation")
            for method in ("full_pool", *METHODS):
                rowname = name if method == "full_pool" else name+"/"+method
                row = by_name[rowname]
                a, b = row["physical_w1"]["duration_s"], uniform["physical_w1"]["duration_s"]
                if (row["support"]["duration_s"] != uniform["support"]["duration_s"]
                        or a is None or b is None or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10)):
                    raise ValueError("Shared duration distributions or support differ across selectors")
                for baseline, baseline_row in (("G34_Uniform", uniform), ("G33_same_method", old_by_name[rowname])):
                    metrics = {}
                    for field, keys in (("physical_w1", METRICS), ("native_w1", NATIVE)):
                        for metric in keys:
                            left, right = row[field][metric], baseline_row[field][metric]
                            support = row["support"][metric] == baseline_row["support"][metric]
                            metrics[metric] = {"delta_lower_is_better": left-right if support and left is not None and right is not None else None,
                                               "supported_request_counts_equal": support}
                    differences.append({"name": rowname, "baseline": baseline, "metrics": metrics})
        result = {"complete": True, "experiment_phase": EXPERIMENT_PHASE,
                  "expected_evaluations": 10, "completed_evaluations": len(rows), "new_blind_reference": False,
                  "sampled_duration_source": DURATION_SOURCE, "array_contract_checks": audits,
                  "inputs_sha256": inputs, "common_sources": sources, "rows": rows, "differences": differences,
                  "code_sha256": digest(__file__),
                  "limits": ["Mechanism diagnostic chosen after inspecting G33; the same development reference is reused, not a new blind test.",
                             "No CNN/diffusion retraining, prior fitting or extra strength/step tuning; K8/DDIM100 remains fixed.",
                             "Per-request shared T is the old predetermined Uniform training-prior draw, never the observed reference duration.",
                             "All four selectors have equal duration distributions by construction; this is a control, not a duration-quality improvement.",
                             "Old/new Uniform is checked bitwise on all map-ready requests; other candidates retain noise keys but may have different sequence lengths.",
                             "Equal candidate/DDIM budgets do not imply equal tensor-element compute or wall time.",
                             "Each seed is descriptive on one SUMO episode and known route templates; no confidence intervals or broad generalization claim.",
                             "Teacher target confidence is auxiliary; physical W1 and coverage determine reported quality.",
                             "Equal metric support counts are not a significance test; unsupported metrics remain NA."]}
        save(output/"summary.json", result)
        fields = ["name", "kind", "seed", "requested", "paired_reference_and_generation", "generated_candidates_all", *METRICS, *NATIVE]
        csv_table(output/"macro.csv", fields, [{**{k: r[k] for k in fields if k in r}, **r["physical_w1"], **r["native_w1"]} for r in rows])
        per_class = [{"name": r["name"], "class": label, "metric": key,
                      "value": r["per_class"][label]["physical_w1" if key in METRICS else "native_w1"][key],
                      "supported_requests": r["support"][key][label]}
                     for r in rows for label in CLASSES for key in list(METRICS)+list(NATIVE)]
        csv_table(output/"per-class.csv", ["name", "class", "metric", "value", "supported_requests"], per_class)
        text = ["# G34固定时长机制诊断", "", "复用已查看的G33开发参考；不是新盲测。两种子共10项评价，所有W1越低越好。", "",
                "| 方法/种子 | 请求 | 配对参考 | 原生速度W1 m/s | 原生加速度W1 m/s² | 时长W1 s |", "|---|---:|---:|---:|---:|---:|"]
        text += [f"| {r['name']} | {r['requested']} | {r['paired_reference_and_generation']} | {render_number(r['native_w1'][NATIVE[0]])} | {render_number(r['native_w1'][NATIVE[1]])} | {render_number(r['physical_w1']['duration_s'])} |" for r in rows]
        text += ["", "逐点核验：所有地图支持请求的旧/新Uniform时间与坐标逐元素完全相等；所有K8候选和四种选择共享该训练先验时长。", "",
                 "十项全部指标、逐类支持数及对旧同方法/新Uniform差值见macro.csv、per-class.csv和summary.json。", ""]
        text += ["- "+limit for limit in result["limits"]]
        (output/"TABLES.md").write_text("\n".join(text)+"\n")
        print(json.dumps({"complete": True, "evaluations": len(rows), "array_checks": [{k: v for k, v in a.items() if k != "sources"} for a in audits]}))
        return result
    except Exception as error:
        save(output/"failed.json", {"complete": False, "experiment_phase": EXPERIMENT_PHASE, "error": repr(error)})
        raise


def self_test():
    ident = "synthetic-request"
    slot = int(permutation(ident, K)[0])
    names = [f"{ident}/candidate-{k:02d}" for k in range(K)]
    row = np.array([[0., 0., 0.], [.25, 1., 0.], [.5, 2., 0.]])
    pool = {"ids": np.asarray(names), "request_ids": np.asarray([ident]*K), "labels": np.zeros(K, np.int64),
            "offsets": np.arange(K+1)*3, "txy": np.concatenate([row]*K)}
    plan = {"request_id": ident, "label": 0, "uniform_slot": slot,
            "duration_source_candidate_id": names[slot], "duration": .5}
    selected = {m: {"ids": np.asarray([names[slot]]), "request_ids": np.asarray([ident]), "labels": np.asarray([0]),
                    "offsets": np.array([0, 3]), "txy": row.copy()} for m in METHODS}
    check_arrays(pool, pool, [plan], selected, 1)
    # Reject changed Uniform coordinates, a non-Uniform duration, a missing
    # slot, and a selection that differs from its declared source candidate.
    failures = []
    for kind in ("Uniform_coordinate_change", "nonuniform_duration_change", "missing_candidate", "selected_source_change"):
        new = {k: v.copy() for k, v in pool.items()}
        views = {m: {k: v.copy() for k, v in p.items()} for m, p in selected.items()}
        if kind == "Uniform_coordinate_change":
            new["txy"][slot*3+1, 1] += 1e-9
        elif kind == "nonuniform_duration_change":
            new["txy"][((slot+1)%K)*3+2, 0] += .1
        elif kind == "missing_candidate":
            new["ids"] = new["ids"][:-1]
        else:
            views["S"]["txy"][1, 1] += 1e-9
        try:
            check_arrays(pool, new, [plan], views, 1)
        except ValueError:
            failures.append(kind)
        else:
            raise AssertionError("Invalid diagnostic accepted: "+kind)
    result = {"complete": True, "scope": "synthetic array checks only", "rejections": failures}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("summarize", "checks"))
    parser.add_argument("--root", type=Path, default=Path("g34-data"))
    parser.add_argument("--g33-root", type=Path, default=Path("g33-data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "checks":
        self_test()
    else:
        if args.output is None:
            parser.error("summarize requires --output")
        summary(args.root, args.g33_root, args.output)
