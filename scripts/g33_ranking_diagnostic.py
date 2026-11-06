"""Read-only aggregate mechanism check of completed native G33 GAF selections.

No fitting, candidate generation, parameter changes, reference trajectories or
physical-quality claims. Candidate IDs/probabilities stay on the server; only
per-class aggregates and source hashes leave this tool.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


CLASSES = ["bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
METHODS = ("Uniform", "S", "V", "SV")
FLOOR = 1e-8


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe(values):
    values = list(values)
    array = np.asarray([v for v in values if v is not None], dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite supported summary input")
    return {"n": len(array), "unsupported_n": len(values)-len(array),
            **({"mean": float(array.mean()), "min": float(array.min()),
                "median": float(np.median(array)), "p95": float(np.quantile(array, .95)),
                "max": float(array.max())} if len(array) else {})}


def frequency(values):
    values = list(values)
    return {"numerator": int(sum(values)), "denominator": len(values),
            "fraction": float(np.mean(values)) if values else None}


def completed(folder):
    receipt = json.loads((folder/"complete.json").read_text())
    if (receipt.get("complete") is not True or receipt.get("phase") != "terminal_gaf_selection"
            or receipt.get("arm") != "native_0_25s" or receipt.get("family") != "diffusion"
            or receipt.get("classes") != CLASSES or receipt.get("reference_data_opened") is not False
            or receipt.get("K") != 8 or receipt.get("methods") != list(METHODS)
            or receipt["policy"]["probability_floor"] != FLOOR
            or receipt["policy"]["uniform_seed"] != 33073):
        raise ValueError("Completed fixed native G33 terminal selection required")
    for name in ("scores.npz", "encoding-status.json", "selection-status.json"):
        if digest(folder/name) != receipt["artifacts"][name]["sha256"]:
            raise ValueError("Frozen scoring artifact changed")
    return receipt


def load(folder):
    receipt = completed(folder)  # Must pass before reading potentially partial arrays.
    with np.load(folder/"scores.npz", allow_pickle=False) as z:
        scores = {k: z[k] for k in z.files}
    encoding = json.loads((folder/"encoding-status.json").read_text())["rows"]
    selections = json.loads((folder/"selection-status.json").read_text())["rows"]
    lookup = {str(ident): i for i, ident in enumerate(scores["ids"])}
    encoded = {row["id"]: row for row in encoding}
    if len(lookup) != len(scores["ids"]) or set(lookup) != set(encoded):
        raise ValueError("Encoding and probability candidate identities differ")
    if scores["probabilities"].shape != (len(lookup), 2, 5):
        raise ValueError("Expected two five-class probability vectors per candidate")
    groups, choices = defaultdict(list), defaultdict(dict)
    for i, request in enumerate(scores["request_ids"].astype(str)):
        groups[request].append(i)
    for row in selections:
        if row["method"] not in METHODS or row["method"] in choices[row["request_id"]]:
            raise ValueError("Unknown/duplicate method in selection record")
        choices[row["request_id"]][row["method"]] = row
    if len(choices) != receipt["pools"]:
        raise ValueError("Selection pool count changed")
    rows = []
    for request, selected in choices.items():
        if set(selected) != set(METHODS) or len({row["label"] for row in selected.values()}) != 1:
            raise ValueError("Methods have different request/class support")
        label = int(selected["Uniform"]["label"])
        indices = groups[request]
        if any(int(scores["labels"][i]) != label for i in indices):
            raise ValueError("Candidate and selected label differ")
        flags = {bool(r["shared_uniform_fallback"]) for r in selected.values()}
        if len(flags) != 1:
            raise ValueError("Fallback not shared by all methods")
        fallback = flags.pop()
        if fallback and len({r["id"] for r in selected.values()}) != 1:
            raise ValueError("Failed pool methods did not select same Uniform slot")
        hashes = {encoded[str(scores["ids"][i])]["spatial_image_sha256"] for i in indices
                  if encoded[str(scores["ids"][i])]["spatial_status"] == "encoded"}
        if len(hashes) != selected["Uniform"]["spatial_unique_images_in_original_pool"]:
            raise ValueError("Saved image diversity aggregate does not reproduce")
        ranges, log_ranges, available = {}, {}, {}
        for axis, view in enumerate(("S", "V")):
            prob = scores["probabilities"][indices, axis, label]
            finite = np.isfinite(prob)
            available[view] = int(finite.sum())
            if np.any((prob[finite] < 0) | (prob[finite] > 1)):
                raise ValueError("Invalid target probability")
            full = len(indices) == 8 and finite.all()
            ranges[view] = float(np.ptp(prob)) if full else None
            log_ranges[view] = float(np.ptp(np.log(np.maximum(prob, FLOOR)))) if full else None
        selected_probability = {}
        for method, chosen in selected.items():
            if chosen["id"] in lookup:
                i = lookup[chosen["id"]]
                if str(scores["request_ids"][i]) != request:
                    raise ValueError("Selection points to a different request")
                probability = scores["probabilities"][i, :, label]
            else:
                probability = [np.nan, np.nan]
            selected_probability[method] = {view: float(value) if np.isfinite(value) else None
                                            for view, value in zip(("S", "V"), probability)}
        rows.append({"label": label, "spatial_unique_images": len(hashes), "fallback": fallback,
                     "probability_range": ranges, "floored_logscore_range": log_ranges,
                     "finite_candidate_probabilities": available,
                     "changed_from_uniform": {m: selected[m]["id"] != selected["Uniform"]["id"] for m in METHODS},
                     "S_V_same_candidate": selected["S"]["id"] == selected["V"]["id"],
                     "selected_probability": selected_probability})
    method_receipt = json.loads((folder/"Uniform"/"complete.json").read_text())
    if (method_receipt.get("complete") is not True
            or method_receipt["selection_receipt_sha256"] != digest(folder/"selection-status.json")
            or method_receipt["source_pool_receipt_sha256"] != receipt["sources"]["generation_receipt_sha256"]):
        raise ValueError("Per-method and parent selection receipts disagree")
    sources = {name: digest(folder/name) for name in ("complete.json", "scores.npz", "encoding-status.json", "selection-status.json")}
    sources["request_manifest_sha256"] = method_receipt["request_sha256"]
    return receipt, rows, sources, set(choices)


def aggregate(rows):
    eligible = [r for r in rows if not r["fallback"]]
    return {"pools": len(rows), "full_scoring_eligible_pools": len(eligible),
            "shared_uniform_fallback_pools": len(rows)-len(eligible),
            "spatial_unique_image_count_histogram": {str(k): v for k, v in sorted(Counter(r["spatial_unique_images"] for r in rows).items())},
            "spatial_unique_image_count": describe(r["spatial_unique_images"] for r in rows),
            "target_probability_range_within_full_K8_pool": {v: describe(r["probability_range"][v] for r in rows) for v in ("S", "V")},
            "floored_logscore_range_within_full_K8_pool": {v: describe(r["floored_logscore_range"][v] for r in rows) for v in ("S", "V")},
            "exactly_constant_floored_logscore_pools": {v: sum(r["floored_logscore_range"][v] == 0 for r in rows) for v in ("S", "V")},
            "selection_change_from_Uniform_all_pools": {m: frequency(r["changed_from_uniform"][m] for r in rows) for m in METHODS},
            "selection_change_from_Uniform_eligible_pools": {m: frequency(r["changed_from_uniform"][m] for r in eligible) for m in METHODS},
            "S_V_select_same_candidate_all_pools": frequency(r["S_V_same_candidate"] for r in rows),
            "S_V_select_same_candidate_eligible_pools": frequency(r["S_V_same_candidate"] for r in eligible),
            "selected_target_probability_auxiliary_only": {m: {v: describe(r["selected_probability"][m][v] for r in rows)
                                                                for v in ("S", "V")} for m in METHODS}}


def run(folders, output):
    if output.exists():
        raise FileExistsError("Preserve the previous diagnostic")
    runs, merged, request_hashes, request_sets = {}, [], set(), []
    for folder in folders:
        receipt, rows, sources, request_ids = load(folder)
        seed = int(receipt["seed"])
        if seed not in (42, 43) or str(seed) in runs:
            raise ValueError("Exactly one completed run per native seed42/43 required")
        runs[str(seed)] = {"seed": seed, "all": aggregate(rows),
                          "by_class": {name: aggregate([r for r in rows if r["label"] == c]) for c, name in enumerate(CLASSES)},
                          "source_sha256": sources}
        merged.extend(rows)
        request_hashes.add(sources["request_manifest_sha256"])
        request_sets.append(request_ids)
    if set(runs) != {"42", "43"} or len(request_hashes) != 1 or request_sets[0] != request_sets[1]:
        raise ValueError("Both fixed native seeds must share the same frozen request support")
    result = {"complete": True, "scope": "read_only_posthoc_mechanism_diagnostic_no_fit_or_selection_change",
              "classes": CLASSES, "runs": runs,
              "pooled_request_seed_pairs": {"all": aggregate(merged),
                  "by_class": {name: aggregate([r for r in merged if r["label"] == c]) for c, name in enumerate(CLASSES)}},
              "script_sha256": digest(Path(__file__)), "reference_trajectories_opened": False,
              "raw_ids_or_probabilities_exported": False,
              "interpretation_limits": ["GAF classification across observations does not prove within-fixed-route candidate ranking.",
                "Spatial RGB hashes detect identical encoded images; different images can reflect sampling density and cell skipping on the same route.",
                "Probability ranges describe frozen-teacher sensitivity, not correctness or physical quality.",
                "Selected target probabilities are the same teachers used for selection and are auxiliary only.",
                "Shared failures force Uniform for all methods and mechanically increase agreement; eligible-pool summaries are separate.",
                "Floored logscore uses the already fixed1e-8 floor; no threshold or weight is selected from this diagnostic.",
                "Pooled rows are request-by-model-seed pairs, not independent traffic realizations; no uncertainty intervals or scientific benefit claim."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def self_test():
    row = {"label": 0, "spatial_unique_images": 1, "fallback": False,
           "probability_range": {"S": 0., "V": .2}, "floored_logscore_range": {"S": 0., "V": 1.},
           "changed_from_uniform": {m: m in ("V", "SV") for m in METHODS}, "S_V_same_candidate": False,
           "selected_probability": {m: {"S": .3, "V": .2 if m in ("Uniform", "S") else .4} for m in METHODS}}
    summary = aggregate([row])
    assert summary["spatial_unique_image_count_histogram"] == {"1": 1}
    assert summary["selection_change_from_Uniform_all_pools"]["S"]["fraction"] == 0
    assert summary["selection_change_from_Uniform_all_pools"]["V"]["fraction"] == 1
    assert summary["selected_target_probability_auxiliary_only"]["SV"]["V"]["mean"] == .4
    assert aggregate([])["S_V_select_same_candidate_all_pools"]["fraction"] is None
    assert describe([None, 0.])["unsupported_n"] == 1
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder)
        (p/"complete.json").write_text('{"complete":false}')
        try:
            completed(p)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete run accepted")
    print("g33_ranking_diagnostic_checks_passed:no_partial_reads,summaries,unsupported_support", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.selected or args.output is None:
            parser.error("Two completed --selected folders and fresh --output required")
        result = run(args.selected, args.output)
        print(json.dumps({"complete": result["complete"], "seeds": sorted(result["runs"])}))
