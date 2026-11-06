"""Physical source fidelity on continuous native observations, without classifiers.

The first seven features reproduce G23/G28's one-second arc-length definitions.
Each trajectory contributes one feature value. Joint Bayesian-bootstrap units
are base entity IDs, not native samples. Shared CNR/native demand identities
receive the same weight, and both comparisons reuse the CNR reference weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CLASSES = ["pedestrian", "bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
DOMAINS = ("cnr", "sumo", "native")
SPLITS = ("train", "val")
METRICS = ["longest_steady_s", "duration_s", "mean_absolute_lag4_speed_change_mps",
           "mean_speed_mps", "mean_absolute_acceleration_mps2", "stopped_fraction",
           "speed_p95_mps", "longest_stop_s", "stop_episodes_per_minute", "arc_length_m"]
UNITS = ["s", "s", "m/s", "m/s", "m/s^2", "fraction", "m/s", "s", "episodes/min", "m"]
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 30055


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runs(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    start, stop = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = stop - start
    return int(lengths.max()) if len(lengths) else 0, int(len(lengths))


def physical(txy):
    """Return all physical features and the common integer-second speed series."""
    txy = np.asarray(txy, dtype=np.float64)
    if txy.ndim != 2 or txy.shape[1] != 3 or len(txy) < 2 or not np.isfinite(txy).all():
        raise ValueError("Invalid native observation array")
    t = txy[:, 0]-txy[0, 0]
    if not np.allclose(np.diff(t), .25, atol=1e-7, rtol=0):
        raise ValueError("Quality input must be a continuous native .25-second segment")
    if t[-1] < 5.:
        raise ValueError("Physical metric support requires at least five seconds; no silent drop")
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(txy[:, 1:], axis=0), axis=1))]
    grid = np.arange(int(np.floor(t[-1]))+1)
    speed = np.diff(np.interp(grid, t, distance))
    steady = (speed[1:] >= .3) & (np.abs(np.diff(speed)) < .5)
    longest_steady, _ = runs(steady)
    longest_stop, stop_count = runs(speed < .3)
    features = np.asarray([longest_steady, t[-1], np.abs(speed[4:]-speed[:-4]).mean(),
                           speed.mean(), np.abs(np.diff(speed)).mean(), (speed < .3).mean(),
                           np.quantile(speed, .95), longest_stop,
                           60.*stop_count/len(speed), distance[-1]], dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Physical metric became nonfinite; preserve the input and diagnose")
    return features, speed, t, grid


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite values in a reported distribution")
    if not len(values):
        return {"n": 0, "supported": False}
    return {"n": len(values), "supported": True, "mean": float(values.mean()),
            "std": float(values.std()), "min": float(values.min()),
            "p05": float(np.quantile(values, .05)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def load(path):
    with np.load(path, allow_pickle=False) as z:
        ids = z["ids"].astype(str)
        labels = z["labels"].astype(np.int64)
        xy, offset = z["txy"], z["offset"]
        if len(ids) != len(labels) or len(set(ids)) != len(ids):
            raise ValueError("Participant IDs must be unique and aligned with labels")
        if offset.dtype.kind not in "iu" or offset.shape != (len(ids)+1,) or offset[0] != 0 or offset[-1] != len(xy):
            raise ValueError("Native offsets do not match observation rows")
        if np.any(np.diff(offset) <= 0) or np.any((labels < 0) | (labels >= 6)):
            raise ValueError("Unsupported classes or empty entity segments")
        base_ids_present = "base_ids" in z
        base = z["base_ids"].astype(str) if base_ids_present else ids
        tags = z["entity_tags"].astype(str) if "entity_tags" in z else np.repeat("", len(ids))
        if len(base) != len(ids) or len(tags) != len(ids):
            raise ValueError("Base entity metadata is not aligned")
        groups = base.tolist()
        if len(set(groups)) != len(groups) or any(not x for x in groups):
            raise ValueError("More than one row per base entity; entity bootstrap would be invalid")
        native = z["native_speed"] if "native_speed" in z else None
        duration = z["duration"] if "duration" in z else None
        if duration is not None and duration.shape != (len(ids),):
            raise ValueError("Saved duration metadata is not aligned")
        if native is not None and native.shape != (len(xy),):
            raise ValueError("Native speed must use exactly the same core offsets as txy")
        features, speed_audit = [], {str(c): [] for c in range(6)}
        invalid_native = np.zeros(6, dtype=int)
        covered_seconds = []
        for i, (start, stop) in enumerate(zip(offset[:-1], offset[1:])):
            feature, speed, t, grid = physical(xy[start:stop])
            if duration is not None and not np.isclose(feature[1], duration[i], atol=1e-7, rtol=0):
                raise ValueError("Saved duration is not the observed continuous-segment span")
            features.append(feature)
            covered_seconds.append(len(speed))
            if native is not None:
                nv = np.asarray(native[start:stop], dtype=np.float64)
                if not np.isfinite(nv).all() or np.any(nv < 0):
                    invalid_native[labels[i]] += 1
                    continue  # Only the optional native-speed consistency audit is unsupported.
                # Integrate native instantaneous speeds over the same [0,floor(T)]
                # seconds as coordinate arc-length speed, using trapezoidal .25s bins.
                native_arc = np.r_[0., np.cumsum((nv[:-1]+nv[1:])*.5*np.diff(t))]
                native_seconds = np.diff(np.interp(grid, t, native_arc))
                error = speed-native_seconds
                speed_audit[str(labels[i])].append([np.abs(error).mean(), error.mean(),
                                                    np.abs(error).max()])
    values = np.asarray(features, dtype=np.float64).reshape(len(ids), len(METRICS))
    table = {}
    class_means = []
    for c, name in enumerate(CLASSES):
        selected = values[labels == c]
        table[name] = {name_: statistics(selected[:, j]) for j, name_ in enumerate(METRICS)}
        if len(selected):
            class_means.append(selected.mean(0))
    macro = (dict(zip(METRICS, np.mean(class_means, axis=0).tolist())) if len(class_means) == 6 else None)
    audit = {"available": native is not None,
             "definition": "per-track coordinate versus trapezoid-integrated native speed, identical one-second bins",
             "interpretation": "includes rounding, chord geometry and integration effects; not a pure coordinate quantization estimate",
             "invalid_or_missing_native_tracks_by_class": invalid_native.tolist(),
             "by_class": {}}
    if native is None:
        audit["reason"] = "raw cohort does not contain an aligned native_speed field; no substitute clock or source is used"
    else:
        for c, name in enumerate(CLASSES):
            a = np.asarray(speed_audit[str(c)], dtype=float).reshape(-1, 3)
            audit["by_class"][name] = {key: statistics(a[:, j]) for j, key in enumerate(
                ["mean_absolute_error_mps", "mean_signed_error_mps", "maximum_absolute_error_mps"])}
    return {"ids": ids, "base_ids": base, "base_ids_present": base_ids_present,
            "groups": set(groups), "labels": labels, "values": values,
            "report": {"rows": len(ids), "class_counts": np.bincount(labels, minlength=6).tolist(),
                       "by_class": table, "macro_of_class_means": macro,
                       "macro_supported_class_count": len(class_means),
                       "whole_second_speed_coverage_s": statistics(covered_seconds),
                       "native_speed_consistency": audit, "sha256": digest(path)}}


def bayesian_weights(keys_by_domain, rng, repetitions):
    """One Exp(1) weight per union entity, shared wherever that identity occurs."""
    union = sorted(set().union(*[set(keys) for keys in keys_by_domain.values()]))
    indices = {key: i for i, key in enumerate(union)}
    draws = np.vstack([np.ones(len(union)), rng.exponential(1., (repetitions, len(union)))])
    return {domain: draws[:, [indices[key] for key in keys]] for domain, keys in keys_by_domain.items()}, len(union)


def wasserstein_curve(left, right, left_weight, right_weight):
    """Exact 1D empirical W1 for original and bootstrap multiplicities."""
    left, right = np.asarray(left), np.asarray(right)
    if not len(left) or not len(right) or not np.isfinite(np.r_[left, right]).all():
        raise ValueError("W1 requires finite nonempty source and reference")
    if left_weight.shape[1] != len(left) or right_weight.shape[1] != len(right) or left_weight.shape[0] != right_weight.shape[0]:
        raise ValueError("W1 weight dimensions do not align")
    support = np.r_[left, right]
    order = np.argsort(support, kind="stable")
    deltas = np.diff(support[order])
    signed = np.concatenate([left_weight, -right_weight], axis=1)[:, order]
    return (np.abs(np.cumsum(signed, axis=1)[:, :-1])*deltas).sum(1)


def interval(curve):
    return {"estimate": float(curve[0]),
            "bayesian_bootstrap_interval95": np.quantile(curve[1:], [.025, .975]).tolist()}


def evaluate(root, output, repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED):
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing physical diagnosis")
    data = {domain: {split: load(root/"raw"/f"{domain}-{split}.npz") for split in SPLITS}
            for domain in DOMAINS}
    for domain in DOMAINS:
        if data[domain]["train"]["groups"] & data[domain]["val"]["groups"]:
            raise ValueError(f"Base entities cross training/validation in {domain}")
    if not all(data["native"][split]["base_ids_present"] for split in SPLITS):
        raise ValueError("Native cohorts must preserve base_ids for shared-identity weighting")
    cross_split_counts = {
        "cnr_train_native_val": len(set(data["cnr"]["train"]["ids"]) & set(data["native"]["val"]["base_ids"])),
        "cnr_val_native_train": len(set(data["cnr"]["val"]["ids"]) & set(data["native"]["train"]["base_ids"])),
    }
    if any(cross_split_counts.values()):
        raise ValueError("Shared CNR/native base identities must inherit the same data partition")
    rng = np.random.default_rng(seed)
    keys = {
        "cnr": ["cnr-native:"+ident for ident in data["cnr"]["val"]["ids"]],
        "native": ["cnr-native:"+ident for ident in data["native"]["val"]["base_ids"]],
        "sumo": ["random-sumo:"+ident for ident in data["sumo"]["val"]["ids"]],
    }
    joint_weights, union_count = bayesian_weights(keys, rng, repetitions)
    pair_counts = {a+"--"+b: len(set(keys[a]) & set(keys[b]))
                   for i, a in enumerate(DOMAINS) for b in DOMAINS[i+1:]}
    cnr_labels = dict(zip(data["cnr"]["val"]["ids"], data["cnr"]["val"]["labels"]))
    native_labels = dict(zip(data["native"]["val"]["base_ids"], data["native"]["val"]["labels"]))
    shared_ids = set(cnr_labels) & set(native_labels)
    changed_taxonomy = sum(int(cnr_labels[i] != native_labels[i]) for i in shared_ids)
    shared_class_matrix = np.bincount([6*int(cnr_labels[i])+int(native_labels[i]) for i in shared_ids],
                                      minlength=36).reshape(6, 6).tolist()
    source_curves = {name: {} for name in ("sumo", "native")}
    differences, unsupported = {}, []
    for c, name in enumerate(CLASSES):
        samples = {domain: data[domain]["val"]["values"][data[domain]["val"]["labels"] == c]
                   for domain in DOMAINS}
        missing = [domain for domain, values in samples.items() if len(values) == 0]
        if missing:
            unsupported.append({"class": name, "missing_domains": missing})
        weights = {}
        for domain in DOMAINS:
            if len(samples[domain]):
                w = joint_weights[domain][:, data[domain]["val"]["labels"] == c]
                weights[domain] = w/w.sum(axis=1, keepdims=True)
        for domain in ("sumo", "native"):
            if domain not in weights or "cnr" not in weights:
                continue
            source_curves[domain][name] = np.stack([
                wasserstein_curve(samples[domain][:, j], samples["cnr"][:, j], weights[domain], weights["cnr"])
                for j in range(len(METRICS))], axis=1)
        if all(name in source_curves[domain] for domain in ("sumo", "native")):
            differences[name] = source_curves["native"][name]-source_curves["sumo"][name]
    def summarize_curves(curves):
        macro = np.mean(list(curves.values()), axis=0) if len(curves) == 6 else None
        return {"by_class": {name: {metric: interval(curve[:, j]) for j, metric in enumerate(METRICS)}
                             for name, curve in curves.items()},
                "macro_equal_class": ({metric: interval(macro[:, j]) for j, metric in enumerate(METRICS)}
                                      if macro is not None else None),
                "supported_classes": list(curves),
                "unsupported_classes": [name for name in CLASSES if name not in curves],
                "source_support_gaps": unsupported}
    report = {"complete": True, "scope": "source-distribution diagnosis on existing training/validation cohorts; no generation or classifier input",
              "classes": CLASSES, "metrics": METRICS, "units": dict(zip(METRICS, UNITS)),
              "definitions": {
                  "speed": "One-second increments of interpolated cumulative native coordinate arc length; bins cover [0,floor(observed span)].",
                  "longest_steady_s": "Longest true run of (v[1:]>=.3)&(abs(diff(v))<.5), following G23/G28. A constant N-second speed series has N-1 stable transitions.",
                  "duration_s": "Observed continuous-segment span, including any fractional final second; not declared full journey duration.",
                  "mean_absolute_lag4_speed_change_mps": "Mean abs(v[t+4]-v[t]); not divided by four.",
                  "mean_absolute_acceleration_mps2": "Mean absolute difference of one-second speed bins, divided by one second.",
                  "stopped_fraction": "Fraction of one-second speed bins below .3 m/s.",
                  "longest_stop_s": "Longest contiguous run of bins below .3 m/s.",
                  "stop_episodes_per_minute": "Number of stopped runs (including a left-censored first run) *60 / number of whole-second speed bins.",
                  "arc_length_m": "Native coordinate arc length over the entire continuous segment, including a fractional final second.",
                  "feature_scope": "The first seven features reproduce G23 phys/G28 physical; all are scalar trajectory-level distributions."},
              "domains": {domain: {split: data[domain][split]["report"] for split in SPLITS} for domain in DOMAINS},
              "validation_w1_to_cnr": {domain: summarize_curves(curves) for domain, curves in source_curves.items()},
              "native_minus_random_w1_to_cnr": summarize_curves(differences),
              "difference_direction": "negative means native MoST demand is closer to CNR for that metric; no sum across different physical units",
              "bootstrap": {"method": "joint-identity weighted Bayesian bootstrap", "repetitions": repetitions,
                            "seed": seed, "unit": "one union base entity across source domains",
                            "draw": "independent Exp(1) weight per union entity; shared CNR/native identity has the same weight",
                            "normalization": "weights separately normalized within each source and class",
                            "reference": "same CNR weights reused for native and random comparisons",
                            "sources": "CNR uses original IDs, native uses base_ids; random SUMO IDs are domain-prefixed to prevent accidental identity matching",
                            "union_entity_count": union_count, "validation_shared_identity_counts": pair_counts,
                            "shared_cnr_native_label_disagreements": changed_taxonomy,
                            "shared_cnr_native_label_matrix": shared_class_matrix,
                            "cross_source_train_val_shared_identity_counts": cross_split_counts,
                            "macro": "equal mean across all six classes within one physical metric; absent if any class unsupported",
                            "limits": "Bayesian-bootstrap credible intervals, not frequency-guaranteed confidence intervals; conditional on observed demand templates and a few simulation seeds, not new episodes/lines/maps; no multiple-comparison adjustment"},
              "checks": {"one_row_per_base_entity": True, "train_val_base_entity_disjoint": True,
                         "all_input_rows_have_finite_metrics": True, "native_clock_point25_seconds": True,
                         "no_short_rows_silently_dropped": True, "frozen_test_read": False},
              "script_sha256": digest(Path(__file__))}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(output)
    print(json.dumps({"complete": True, "output": str(output), "supported_classes": len(differences),
                      "bootstrap_repetitions": repetitions}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("g30-data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.root, args.output or args.root/"quality.json")
