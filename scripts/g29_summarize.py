"""Summarize the fixed G29 development audit; never open frozen test data.

Intervals resample validation vehicles within class, keeping the two fitted
seeds fixed. All arms on a cohort share the same draws. They are conditional
development diagnostics, not scene-level or model-training uncertainty.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (42, 43)
BOOTSTRAP_SEED = 29055
REPLICATES = 2000
CLASSES = ("pedestrian", "bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber")
ARMS = ("xy", "grid_xy", "native_h", "squeezed_h", "unit_h", "gaf")
METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "per_class_f1",
           "per_class_recall", "per_class_precision")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def cohort(path, prediction=False):
    with np.load(path, allow_pickle=False) as archive:
        ids, labels = archive["ids"].astype(str), archive["labels"]
        require(ids.ndim == labels.ndim == 1 and len(ids) == len(labels) > 0,
                f"Invalid ID/label dimensions: {path}")
        require(len(set(ids)) == len(ids), f"Duplicate IDs: {path}")
        require(np.issubdtype(labels.dtype, np.integer) and np.isin(labels, range(6)).all(),
                f"Invalid labels: {path}")
        result = {"ids": ids, "labels": labels.astype(np.int64)}
        if prediction:
            probability = archive["probabilities"]
            require(probability.shape == (len(ids), 6) and np.isfinite(probability).all(),
                    f"Invalid probabilities: {path}")
            require((probability >= 0).all() and (probability <= 1).all()
                    and np.allclose(probability.sum(1), 1, rtol=0, atol=1e-5),
                    f"Non-probability predictions: {path}")
            result["predicted"] = probability.argmax(1)
    return result


def same_cohort(left, right, description):
    require(np.array_equal(left["ids"], right["ids"])
            and np.array_equal(left["labels"], right["labels"]),
            f"Ordered cohort mismatch: {description}")


def class_counts(rows):
    return np.bincount(rows["labels"], minlength=6)


def confusion_metrics(confusion):
    """Accept either one matrix or an arbitrary leading batch dimension."""
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
    actual, predicted = confusion.sum(-1), confusion.sum(-2)
    f1 = 2 * diagonal / np.maximum(actual + predicted, 1)
    recall = diagonal / np.maximum(actual, 1)
    return {"accuracy": diagonal.sum(-1) / np.maximum(actual.sum(-1), 1),
            "macro_f1": f1.mean(-1), "balanced_accuracy": recall.mean(-1),
            "per_class_f1": f1, "per_class_recall": recall,
            "per_class_precision": diagonal / np.maximum(predicted, 1)}


def metrics(rows):
    confusion = np.bincount(6 * rows["labels"] + rows["predicted"], minlength=36).reshape(6, 6)
    result = {key: np.asarray(value).tolist() for key, value in confusion_metrics(confusion).items()}
    result.update(confusion=confusion.tolist(), rows=len(rows["ids"]),
                  class_counts=class_counts(rows).tolist())
    return result


def check_saved_score(saved, actual, description):
    for key in ("accuracy", "macro_f1", "per_class_f1", "per_class_recall", "confusion", "rows"):
        require(key in saved and np.allclose(saved[key], actual[key], rtol=0, atol=1e-7),
                f"Saved score disagrees with predictions: {description}/{key}")


def stratified_draws(labels, rng):
    """One row per vehicle; preserve each class's observed validation count."""
    parts = []
    for label in range(6):
        indices = np.flatnonzero(labels == label)
        require(len(indices) > 0, "Bootstrap cohort must contain all six classes")
        parts.append(rng.choice(indices, size=(REPLICATES, len(indices)), replace=True))
    return np.concatenate(parts, axis=1)


def bootstrap_metrics(rows, draws):
    codes = 6 * rows["labels"][draws] + rows["predicted"][draws]
    codes += 36 * np.arange(REPLICATES)[:, None]
    confusion = np.bincount(codes.ravel(), minlength=36 * REPLICATES).reshape(REPLICATES, 6, 6)
    return confusion_metrics(confusion)


def intervals(distributions):
    return {key: {"ci95": np.quantile(values, [.025, .975], axis=0).T.tolist()}
            for key, values in distributions.items()}


def aggregate(predictions, draws=None):
    require(set(predictions) == set(SEEDS), "Every result must have exactly seeds 42 and 43")
    same_cohort(predictions[42], predictions[43], "model seeds")
    scores = {seed: metrics(predictions[seed]) for seed in SEEDS}
    mean = {key: np.asarray([scores[seed][key] for seed in SEEDS]).mean(0).tolist()
            for key in (*METRICS, "confusion")}
    result = {"rows": scores[42]["rows"], "class_counts": scores[42]["class_counts"],
              "by_seed": {str(seed): scores[seed] for seed in SEEDS}, "mean": mean}
    distribution = None
    if draws is not None:
        by_seed = {seed: bootstrap_metrics(predictions[seed], draws) for seed in SEEDS}
        distribution = {key: np.mean([by_seed[seed][key] for seed in SEEDS], axis=0)
                        for key in METRICS}
        result["bootstrap_mean_of_seed_scores"] = intervals(distribution)
        result["bootstrap_macro_f1_by_seed"] = {
            str(seed): intervals({"macro_f1": by_seed[seed]["macro_f1"]})["macro_f1"]
            for seed in SEEDS}
    return result, distribution


def difference(left, right, left_distribution=None, right_distribution=None):
    """Left minus right; never pool predictions or treat seeds as vehicles."""
    result = {"by_seed": {
        str(seed): {key: (np.asarray(left["by_seed"][str(seed)][key])
                         - np.asarray(right["by_seed"][str(seed)][key])).tolist()
                    for key in METRICS} for seed in SEEDS},
        "mean": {key: (np.asarray(left["mean"][key]) - np.asarray(right["mean"][key])).tolist()
                 for key in METRICS}}
    if left_distribution is not None:
        require(right_distribution is not None, "Both bootstrap distributions are required")
        result["bootstrap_mean_of_seed_differences"] = intervals({
            key: left_distribution[key] - right_distribution[key] for key in METRICS})
    return result


def expected_models():
    cases = [(domain, "full", arm) for domain in ("cnr", "sumo") for arm in ARMS]
    cases += [(domain, "window512", "gaf") for domain in ("cnr", "sumo")]
    cases += [("sumo-duration", "full", "gaf")]
    return [(domain, view, arm, seed) for domain, view, arm in cases for seed in SEEDS]


def validate_data(root):
    for marker in ("training-complete.txt", "grid-training-complete.txt"):
        require((root / marker).is_file() and (root / marker).read_text().strip() == "complete",
                f"Training completion marker missing or invalid: {marker}")
    read_json(root / "raw/prepared.json")
    manifest = read_json(root / "encoded/complete.json")
    grid = read_json(root / "encoded/grid-control-complete.json")
    require(grid.get("boundary_policy") == "inherited recursion, including reflected dyadic-boundary ties",
            "Grid-control completion must certify the corrected inherited boundary policy")
    require(np.isclose(manifest["frozen_metadata"]["hilbert_eps"], .0009), "Expected frozen eps=.0009")
    for name, expected in manifest["raw_sha256"].items():
        require(digest(root / "raw" / (name + ".npz")) == expected, f"Raw cohort changed: {name}")
    cohorts = {}
    cases = sorted({(domain, view) for domain, view, _, _ in expected_models()})
    for domain, view in cases:
        for split in ("train", "val"):
            name = f"{domain}/{view}/{split}"
            directory = root / "encoded" / name
            completion = read_json(directory / "complete.json")
            rows = cohort(directory / "features.npz")
            require(np.all(class_counts(rows) > 0), f"Missing class: {name}")
            expected_hash = hashlib.sha256("\n".join(rows["ids"]).encode()).hexdigest()
            require(completion["ids_sha256"] == expected_hash, f"Encoded IDs changed: {name}")
            require(completion == manifest["reports"][name], f"Encoded completion changed: {name}")
            require(completion["class_counts"] == class_counts(rows).tolist()
                    and completion["all"]["rows"] == len(rows["ids"]), f"Encoded counts disagree: {name}")
            if domain in ("cnr", "sumo") and view == "full":
                report = grid["reports"][name]
                require(report["class_counts"] == class_counts(rows).tolist()
                        and report["rows"] == len(rows["ids"]), f"Grid cohort mismatch: {name}")
            cohorts[domain, view, split] = rows
        require(not set(cohorts[domain, view, "train"]["ids"]) & set(cohorts[domain, view, "val"]["ids"]),
                f"Train/validation overlap: {domain}/{view}")
    for view in ("full", "window512"):
        for split in ("train", "val"):
            require(np.array_equal(class_counts(cohorts["cnr", view, split]),
                                   class_counts(cohorts["sumo", view, split])),
                    f"Source class priors differ: {view}/{split}")
    for split in ("train", "val"):
        require(np.array_equal(class_counts(cohorts["cnr", "full", split]),
                               class_counts(cohorts["sumo-duration", "full", split])),
                f"Duration cohort class priors differ: {split}")
    return cohorts


def load_models(root, cohorts):
    expected = expected_models()
    names = {f"{domain}-{view}-{arm}-{seed}" for domain, view, arm, seed in expected}
    actual = {path.parent.name for path in (root / "models").glob("*/complete.json")}
    require(actual == names, f"Expected exactly 30 completed prescribed fits; missing={sorted(names-actual)}, extra={sorted(actual-names)}")
    models = {}
    for domain, view, arm, seed in expected:
        name = f"{domain}-{view}-{arm}-{seed}"
        directory = root / "models" / name
        completion, config = read_json(directory / "complete.json"), read_json(directory / "config.json")
        require(completion.get("complete") is True, f"Model not complete: {name}")
        for key, value in (("source", domain), ("view", view), ("arm", arm), ("seed", seed), ("epochs", 50)):
            require(completion.get(key) == value, f"Unexpected model completion {name}/{key}")
        for key, value in (("domain", domain), ("view", view), ("arm", arm), ("seed", seed),
                           ("epochs", 50), ("batch_size", 64), ("learning_rate", .0002),
                           ("weight_decay", .0001), ("selection", "fixed final epoch50, no monitor-based selection")):
            require(config.get(key) == value, f"Unexpected model configuration {name}/{key}")
        train_counts = class_counts(cohorts[domain, view, "train"])
        require(config["train_counts"] == train_counts.tolist(), f"Training counts changed: {name}")
        weights = train_counts.astype(float) ** -.5
        require(np.allclose(config["class_weights"], weights / weights.mean(), atol=1e-6, rtol=0),
                f"Class weights changed: {name}")
        predictions = {}
        for split, score_key in (("train", "train"), ("val", "validation")):
            rows = cohort(directory / (split + ".npz"), prediction=True)
            same_cohort(rows, cohorts[domain, view, split], f"{name}/{split}")
            check_saved_score(completion[score_key], metrics(rows), f"{name}/{split}")
            predictions[split] = rows
        cross_domains = [target for target in ("cnr", "sumo") if target != domain]
        require(set(completion["cross_domain_validation"]) == set(cross_domains), f"Cross sources changed: {name}")
        for target in cross_domains:
            rows = cohort(directory / f"cross-{target}.npz", prediction=True)
            same_cohort(rows, cohorts[target, view, "val"], f"{name}/cross-{target}")
            check_saved_score(completion["cross_domain_validation"][target], metrics(rows), f"{name}/cross-{target}")
            predictions[f"cross-{target}"] = rows
        models[domain, view, arm, seed] = predictions
    return models


def subset(rows, indices):
    return {key: values[indices] for key, values in rows.items()}


def window_comparison(domain, cohorts, models, rng):
    full, window = cohorts[domain, "full", "val"], cohorts[domain, "window512", "val"]
    index = {ident: i for i, ident in enumerate(full["ids"])}
    window_indices = np.array([i for i, ident in enumerate(window["ids"]) if ident in index], dtype=np.int64)
    full_indices = np.array([index[window["ids"][i]] for i in window_indices], dtype=np.int64)
    full_rows, window_rows = subset(full, full_indices), subset(window, window_indices)
    same_cohort(full_rows, window_rows, f"{domain}/full-window intersection")
    missing = np.ones(len(window["ids"]), dtype=bool)
    missing[window_indices] = False
    counts = class_counts(window_rows)
    result = {"scope": "Full-trained/full-input versus window-trained/window-input GAF on common validation IDs; training populations and input spans both change, not a pure duration effect.",
              "coverage": {"full_validation_rows": len(full["ids"]), "window_validation_rows": len(window["ids"]),
                           "intersection_rows": len(window_indices),
                           "window_coverage_fraction": len(window_indices) / len(window["ids"]),
                           "window_rows_missing_from_full": int(missing.sum()),
                           "window_missing_class_counts": np.bincount(window["labels"][missing], minlength=6).tolist(),
                           "intersection_class_counts": counts.tolist(),
                           "window_subset_of_full": bool(not missing.any())},
              "training_population": {view: {"rows": len(cohorts[domain, view, "train"]["ids"]),
                                                "class_counts": class_counts(cohorts[domain, view, "train"]).tolist()}
                                      for view in ("full", "window512")}}
    if not len(window_indices):
        result["status"] = "no common validation IDs; no comparison computed"
        return result
    full_predictions = {seed: subset(models[domain, "full", "gaf", seed]["val"], full_indices) for seed in SEEDS}
    window_predictions = {seed: subset(models[domain, "window512", "gaf", seed]["val"], window_indices) for seed in SEEDS}
    draws = stratified_draws(window_rows["labels"], rng) if np.all(counts > 0) else None
    a, da = aggregate(full_predictions, draws)
    b, db = aggregate(window_predictions, draws)
    result.update(status="complete" if draws is not None else "point estimates only: intersection lacks a class",
                  full_model_on_common_ids=a, window_model_on_common_ids=b,
                  window_minus_full_on_common_ids=difference(b, a, db, da))
    return result


def summarize(args):
    cohorts = validate_data(args.root)
    models = load_models(args.root, cohorts)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Distinct source/cohort keys get independent draws; every arm on a key uses
    # exactly the same draws, including cross-source predictions on that cohort.
    draws = {key[:2]: stratified_draws(rows["labels"], rng)
             for key, rows in sorted(cohorts.items()) if key[2] == "val" and key[0] != "sumo-duration"}
    results, distributions = {}, {}
    cases = list(dict.fromkeys((domain, view, arm) for domain, view, arm, _ in expected_models()))
    for domain, view, arm in cases:
        key = f"{domain}/{view}/{arm}"
        predictions = {seed: models[domain, view, arm, seed] for seed in SEEDS}
        train, _ = aggregate({seed: predictions[seed]["train"] for seed in SEEDS})
        validation, dist = aggregate({seed: predictions[seed]["val"] for seed in SEEDS}, draws.get((domain, view)))
        result = {"train": train, "validation": validation, "cross_domain_validation": {}}
        distributions[key] = dist
        for target in (target for target in ("cnr", "sumo") if target != domain):
            cross, _ = aggregate({seed: predictions[seed][f"cross-{target}"] for seed in SEEDS},
                                 None if domain == "sumo-duration" else draws[target, view])
            result["cross_domain_validation"][target] = cross
        if domain == "sumo-duration":
            result["scope"] = "Duration-matched SUMO sensitivity cohort: point estimates only; not paired to random SUMO or CNR vehicle IDs."
        results[key] = result
    paired = {}
    for domain in ("cnr", "sumo"):
        for left, right in zip(ARMS[1:], ARMS[:-1]):
            left_key, right_key = f"{domain}/full/{left}", f"{domain}/full/{right}"
            item = difference(results[left_key]["validation"], results[right_key]["validation"],
                              distributions[left_key], distributions[right_key])
            item.update(paired_validation_ids=True, rows=len(cohorts[domain, "full", "val"]["ids"]),
                        architecture_changes=left == "gaf")
            item["interpretation"] = ("Different readout architectures: ImageNet ResNet18 versus SequenceClassifier; not pure encoding loss."
                                      if left == "gaf" else "Same readout architecture and paired IDs; representation/interpolation and readout difficulty can all affect scores.")
            paired[f"{domain}/{left}-minus-{right}"] = item
    source_differences = {}
    for view in ("full", "window512"):
        left, right = f"cnr/{view}/gaf", f"sumo/{view}/gaf"
        item = difference(results[left]["validation"], results[right]["validation"],
                          distributions[left], distributions[right])
        item.update(independent_source_vehicle_bootstrap=True,
                    scope="CNR minus SUMO, matched class counts; different source populations, not a paired or causal source effect.")
        source_differences[view] = item
    duration = {}
    left = results["sumo-duration/full/gaf"]["validation"]
    for reference in ("sumo", "cnr"):
        duration[f"sumo-duration-minus-{reference}"] = difference(left, results[f"{reference}/full/gaf"]["validation"])
    result = {"complete": True, "expected_model_count": 30, "validated_model_count": len(models),
              "classes": list(CLASSES), "model_seeds": list(SEEDS),
              "units": "All scores and differences are fractions, not percentages or percentage points.",
              "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": REPLICATES, "interval": "95% percentile",
                            "unit": "validation participant ID, stratified within each of six classes",
                            "seed_aggregation": "Compute F1 separately for each fixed model seed, then average; no probability ensemble or pooled-seed pseudoreplication.",
                            "scope": "Conditional on fitted models and fixed development partitions. Preserves observed class priors. CNR is one export and SUMO has three monitor episodes; no scene-level or training-uncertainty inference. Exploratory comparisons, no multiplicity correction."},
              "validation_checks": {"completion_markers": True, "raw_hashes_match_encoded_manifest": True,
                                    "fixed_fit_configurations": True, "all_arm_and_seed_cohorts_identical": True,
                                    "source_class_counts_matched_within_view": True,
                                    "saved_scores_reproduced_from_predictions": True,
                                    "frozen_test_data_read": False},
              "cohorts": {"/".join(key): {"rows": len(rows["ids"]), "class_counts": class_counts(rows).tolist()}
                          for key, rows in sorted(cohorts.items())},
              "models": results, "paired_encoding_ladder": paired,
              "cnr_minus_sumo_gaf": source_differences,
              "duration_sensitivity_point_differences": duration,
              "full_window_common_validation_ids": {
                  domain: window_comparison(domain, cohorts, models, rng) for domain in ("cnr", "sumo")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"complete": True, "models": len(models), "output": str(args.output),
                      "full_gaf_mean_macro_f1": {domain: results[f"{domain}/full/gaf"]["validation"]["mean"]["macro_f1"]
                                                 for domain in ("cnr", "sumo")}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("g29-data"))
    parser.add_argument("--output", type=Path, default=Path("g29-data/summary.json"))
    summarize(parser.parse_args())
