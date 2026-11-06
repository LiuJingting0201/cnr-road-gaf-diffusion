"""Read-only fixed-teacher classification audit on the same fresh G35 val legs.

No fitting, checkpoint selection, old CNR targets, or generation quality claims.
Old G33 teachers run on G35's verified inputs; new predictions are reused only
after artifact and row alignment checks. Original teacher/cache provenance stays
separate from the new input-cache provenance.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from g33_prepare import CLASSES, digest, save
from g35_teachers import BudgetGuard, load_cache, validate_records


REPLICATES = 2000
BOOTSTRAP_SEED = 35055
VIEWS = ("spatial", "speed")


def assert_alignment(data, labels, ids, parents, selected):
    for actual, expected in ((labels, data["labels"][selected]), (ids, data["ids"][selected]),
                             (parents, data["parent_ids"][selected])):
        if not np.array_equal(actual, expected):
            raise ValueError("Prediction IDs, parent IDs or labels differ from the declared cache rows")


def weighted_metrics(labels, logits, weights):
    labels = np.asarray(labels)
    logits, weights = np.asarray(logits, float), np.asarray(weights, float)
    if (labels.ndim != 1 or labels.dtype.kind not in "iu" or not np.isin(labels, np.arange(5)).all()
            or logits.shape != (len(labels), 5) or weights.shape != labels.shape
            or not np.isfinite(logits).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Invalid finite five-class weighted predictions")
    predicted = logits.argmax(1)
    confusion = np.bincount(5 * labels + predicted, weights=weights, minlength=25).reshape(5, 5)
    support, predicted_mass = confusion.sum(1), confusion.sum(0)
    diagonal = confusion.diagonal()
    f1 = np.divide(2 * diagonal, support + predicted_mass, out=np.zeros(5), where=support + predicted_mass > 0)
    precision = np.divide(diagonal, predicted_mass, out=np.zeros(5), where=predicted_mass > 0)
    recall = np.divide(diagonal, support, out=np.zeros(5), where=support > 0)
    shifted = logits - logits.max(1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    probability = np.exp(log_prob)
    confidence, correct = probability.max(1), predicted == labels
    bins = np.minimum((confidence * 15).astype(int), 14)
    total = weights.sum()
    ece = 0.
    for k in range(15):
        select = bins == k
        if weights[select].sum() > 0:
            ece += weights[select].sum() / total * abs(
                np.average(confidence[select], weights=weights[select]) -
                np.average(correct[select], weights=weights[select]))
    return {"n_legs": len(labels), "total_weight": float(total),
        "accuracy": float(diagonal.sum() / total), "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.sum(f1 * support) / total),
        "per_class_f1": f1.tolist(), "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
        "weighted_class_support": support.tolist(), "confusion": confusion.tolist(),
        "nll": float(np.average(-log_prob[np.arange(len(labels)), labels], weights=weights)),
        "brier_multiclass_sum": float(np.average(np.square(probability - np.eye(5)[labels]).sum(1), weights=weights)),
        "ece_15_equal_width": float(ece)}


def parent_structure(labels, parents):
    names, inverse, counts = np.unique(np.asarray(parents).astype(str), return_inverse=True, return_counts=True)
    if not len(names) or "" in names:
        raise ValueError("Nonempty parent identities required")
    parent_labels = np.full(len(names), -1, dtype=np.int64)
    for label, index in zip(labels, inverse):
        if parent_labels[index] not in (-1, int(label)):
            raise ValueError("One parent has inconsistent participant labels")
        parent_labels[index] = int(label)
    return inverse, counts, parent_labels


def metric_vector(result):
    return np.asarray([result["accuracy"], result["macro_f1"], result["weighted_f1"],
                       *result["per_class_f1"], *result["per_class_recall"]], dtype=float)


def confusion_vector(confusion):
    """Fast bootstrap scoring; only confusion-derived metrics need intervals."""
    support, predicted = confusion.sum(1), confusion.sum(0)
    diag = confusion.diagonal()
    f1 = np.divide(2 * diag, support + predicted, out=np.zeros(5), where=support + predicted > 0)
    recall = np.divide(diag, support, out=np.zeros(5), where=support > 0)
    return np.r_[diag.sum() / support.sum(), f1.mean(), np.sum(f1 * support) / support.sum(), f1, recall]


def paired_summary(labels, parents, logits_by_model, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    labels = np.asarray(labels)
    if set(logits_by_model) != {f"{era}_{view}" for era in ("old", "new") for view in VIEWS}:
        raise ValueError("Exactly the four predeclared old/new spatial/speed models are required")
    inverse, counts, parent_labels = parent_structure(labels, parents)
    groups = [np.flatnonzero(parent_labels == c) for c in range(5)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Every participant class requires held-out parents")
    points, boot = {}, {}
    # Aggregate all legs into their parent's confusion contribution once.
    matrices = {}
    for model, logits in logits_by_model.items():
        weighted_metrics(labels, logits, np.ones(len(labels)))
        value = np.zeros((len(counts), 25), dtype=np.float64)
        predicted = np.asarray(logits).argmax(1)
        np.add.at(value, (inverse, 5 * labels + predicted), 1.)
        matrices[model] = value
    rng = np.random.default_rng(seed)
    multiplicities = np.zeros((replicates, len(counts)), dtype=np.int64)
    for replicate in range(replicates):
        for group in groups:
            sampled = rng.choice(group, size=len(group), replace=True)
            multiplicities[replicate] += np.bincount(sampled, minlength=len(counts))
    names = ["accuracy", "macro_f1", "weighted_f1",
             *[f"f1:{name}" for name in CLASSES], *[f"recall:{name}" for name in CLASSES]]
    for weighting in ("leg", "equal_parent"):
        points[weighting], boot[weighting] = {}, {}
        row_weights = np.ones(len(labels)) if weighting == "leg" else 1. / counts[inverse]
        for model, logits in logits_by_model.items():
            points[weighting][model] = weighted_metrics(labels, logits, row_weights)
            contribution = matrices[model] if weighting == "leg" else matrices[model] / counts[:, None]
            # Dot product in bounded chunks keeps bootstrap memory independent of leg count.
            values = []
            for start in range(0, replicates, 100):
                confusion = multiplicities[start:start+100] @ contribution
                values.extend(confusion_vector(c.reshape(5, 5)) for c in confusion)
            boot[weighting][model] = np.stack(values)
    comparisons = {}
    for weighting in points:
        comparisons[weighting] = {}
        for view in VIEWS:
            old, new = f"old_{view}", f"new_{view}"
            delta = boot[weighting][new] - boot[weighting][old]
            point_delta = metric_vector(points[weighting][new]) - metric_vector(points[weighting][old])
            comparisons[weighting][f"new_minus_old_{view}"] = {
                metric: {"delta": float(point_delta[j]), "percentile_95": np.quantile(delta[:, j], [.025, .975]).tolist()}
                for j, metric in enumerate(names)}
        # Representation contrast uses the same participants and resampling draws.
        for era in ("old", "new"):
            spatial, speed = f"{era}_spatial", f"{era}_speed"
            delta = boot[weighting][speed] - boot[weighting][spatial]
            point_delta = metric_vector(points[weighting][speed]) - metric_vector(points[weighting][spatial])
            comparisons[weighting][f"{era}_speed_minus_spatial"] = {
                metric: {"delta": float(point_delta[j]), "percentile_95": np.quantile(delta[:, j], [.025, .975]).tolist()}
                for j, metric in enumerate(names)}
    intervals = {weighting: {model: {metric: np.quantile(values[:, j], [.025, .975]).tolist()
                  for j, metric in enumerate(names)} for model, values in models.items()}
                 for weighting, models in boot.items()}
    return {"point": points, "intervals_95": intervals, "paired_comparisons": comparisons,
        "support": {"legs": len(labels), "parents": len(counts),
            "class_legs": np.bincount(labels, minlength=5).tolist(),
            "class_parents": [len(group) for group in groups],
            "parents_with_multiple_legs": int(np.sum(counts > 1)), "max_legs_per_parent": int(counts.max())},
        "bootstrap": {"replicates": replicates, "seed": seed, "unit": "parent_base_id",
            "scheme": "Resample parents within true participant class; fixed observed class-parent counts; all legs carried together",
            "pairing": "Identical parent multiplicities for both teacher generations and both representations",
            "primary": "leg-weighted metrics", "sensitivity": "Each parent has total weight one",
            "calibration": "Point estimates only; no posthoc calibration or confidence-based selection",
            "limits": "Conditional on fixed teachers, one new validation episode, observed class mix, same city/public demand and known bus lines; no training-seed or simulation-episode uncertainty; descriptive intervals, no multiplicity correction"}}


def read_index_metadata(cache):
    receipt = json.loads((cache / "complete.json").read_text())
    if receipt.get("complete") is not True or receipt.get("classes") != CLASSES or (cache / "failed.json").exists():
        raise ValueError("Completed old five-class G33 cache required")
    spec = receipt["artifacts"]["index.npz"]
    path = cache / "index.npz"
    if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
        raise ValueError("Old cache identity metadata hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        # Intentionally only identity, partition and label arrays; no old images or targets.
        records = {key: archive[key] for key in ("ids", "parent_ids", "partition", "labels")}
    validate_records(records)
    return receipt, records


def assert_encoder_compatibility(old, new):
    if old["spatial"]["encoding_metadata_sha256"] != new["spatial"]["encoding_metadata_sha256"]:
        raise ValueError("Old/new spatial encoders have different frozen metadata")
    before, after = old["sources"]["code_sha256"], new["sources"]["code_sha256"]
    required = {"g29_encode.py", "original_gaf.py", "hilbert.py", "sequence.py",
                "representations.py", "g23_modes.py", "g28_teachers.py"}
    if not required.issubset(before) or not required.issubset(after):
        raise ValueError("Encoder provenance is missing a required source file")
    if any(before[name] != after[name] for name in required):
        raise ValueError("Shared original spatial/speed encoder implementation changed")


def verify_teacher(folder, view, cache):
    config = json.loads((folder / "config.json").read_text())
    receipt = json.loads((folder / "complete.json").read_text())
    if ((folder / "failed.json").exists() or receipt.get("complete") is not True
            or receipt.get("selected_epoch") != 50 or config.get("epochs") != 50
            or config.get("checkpoint") != "final_epoch_50_only" or config.get("seed") != 42
            or receipt.get("view") != view or config.get("view") != view
            or receipt.get("classes") != CLASSES or config.get("classes") != CLASSES
            or receipt.get("config_sha256") != digest(folder / "config.json")
            or config.get("cache_receipt_sha256") != digest(cache / "complete.json")):
        raise ValueError("Teacher provenance or fixed-final-checkpoint contract mismatch")
    for name in ("model.pt", "train-predictions.npz", "val-predictions.npz"):
        spec = receipt["artifacts"][name]
        path = folder / name
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError("Teacher model/prediction artifact hash mismatch: " + name)
    return config, receipt


def read_predictions(folder, split, data):
    selected = np.flatnonzero(data["partition"] == split)
    with np.load(folder / f"{split}-predictions.npz", allow_pickle=False) as archive:
        values = {key: archive[key] for key in ("ids", "parent_ids", "labels", "logits")}
    assert_alignment(data, values["labels"], values["ids"], values["parent_ids"], selected)
    # Also rejects nonfinite or wrong-shaped logits.
    weighted_metrics(values["labels"], values["logits"], np.ones(len(selected)))
    return values


def own_train_gap(folder, data, fresh_val):
    train = read_predictions(folder, "train", data)
    old_val = read_predictions(folder, "val", data)
    inverse, counts, _ = parent_structure(train["labels"], train["parent_ids"])
    val_inverse, val_counts, _ = parent_structure(old_val["labels"], old_val["parent_ids"])
    result = {}
    for weighting in ("leg", "equal_parent"):
        weights = np.ones(len(train["ids"])) if weighting == "leg" else 1. / counts[inverse]
        val_weights = np.ones(len(old_val["ids"])) if weighting == "leg" else 1. / val_counts[val_inverse]
        fit = weighted_metrics(train["labels"], train["logits"], weights)
        original_val = weighted_metrics(old_val["labels"], old_val["logits"], val_weights)
        result[weighting] = {"own_training": fit, "own_original_validation": original_val,
            "train_minus_own_val_accuracy": fit["accuracy"] - original_val["accuracy"],
            "train_minus_own_val_macro_f1": fit["macro_f1"] - original_val["macro_f1"],
            "train_minus_fresh_G35_val_accuracy": fit["accuracy"] - fresh_val[weighting]["accuracy"],
            "train_minus_fresh_G35_val_macro_f1": fit["macro_f1"] - fresh_val[weighting]["macro_f1"],
            "scope": "Training and fresh-val populations differ; gap is diagnostic, not a paired data-scale effect"}
    return result


def old_inference(folder, view, values, selected, guard):
    import torch
    from torch import nn
    from torchvision.models import resnet18
    from g28_teachers import image_inputs
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 5)
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True), strict=True)
    model = model.cuda().eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(selected), 64):
            guard.check()
            batch = selected[start:start+64]
            logits.append(model(image_inputs(view, values[batch])).cpu().numpy())
    result = np.concatenate(logits)
    del model
    torch.cuda.empty_cache()
    return result


def run(args):
    output = args.output.resolve()
    guard = BudgetGuard(args.budget_root, output)
    for source in (args.cache, args.old_cache, args.old_teachers, args.new_teachers):
        if output.is_relative_to(source.resolve()) or source.resolve().is_relative_to(output):
            raise ValueError("New audit output must be separate from all immutable inputs")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        guard.check(force=True)
        old_cache_receipt, old_data = read_index_metadata(args.old_cache)
        predictions, provenance, configs, identities = {}, {}, {}, None
        for view in VIEWS:
            prepared, data, values = load_cache(args.cache, view)
            selected = np.flatnonzero(data["partition"] == "val")
            if identities is not None:
                for key in ("ids", "parent_ids", "labels", "partition"):
                    if not np.array_equal(data[key], identities[key]):
                        raise ValueError("Spatial/speed input cohorts differ")
            identities = {key: data[key] for key in ("ids", "parent_ids", "labels", "partition")}
            assert_encoder_compatibility(old_cache_receipt, prepared)
            old_train = set(old_data["parent_ids"][old_data["partition"] == "train"].astype(str))
            new_val = set(data["parent_ids"][selected].astype(str))
            if old_train & new_val:
                raise ValueError("Old teacher training parent overlaps the fresh G35 evaluation cohort")
            for era, base, cache in (("old", args.old_teachers, args.old_cache), ("new", args.new_teachers, args.cache)):
                folder = base / f"{view}-42"
                config, receipt = verify_teacher(folder, view, cache)
                train_records = old_data if era == "old" else data
                train_ix = np.flatnonzero(train_records["partition"] == "train")
                if np.bincount(train_records["labels"][train_ix], minlength=5).tolist() != config["class_counts"]:
                    raise ValueError("Teacher training class counts disagree with its own cache")
                configs[f"{era}_{view}"] = config
                provenance[f"{era}_{view}"] = {"config_sha256": digest(folder / "config.json"),
                    "receipt_sha256": digest(folder / "complete.json"), "model_sha256": receipt["artifacts"]["model.pt"]["sha256"],
                    "training_cache_receipt_sha256": config["cache_receipt_sha256"]}
                if era == "new":
                    prediction = read_predictions(folder, "val", data)
                    predictions[f"{era}_{view}"] = prediction["logits"]
                else:
                    predictions[f"{era}_{view}"] = old_inference(folder, view, values, selected, guard)
                    guard.check(force=True, reserve=1024**2 + len(selected) * 2048)
                    np.savez_compressed(output / f"old-{view}-fresh-val.npz", ids=data["ids"][selected],
                        parent_ids=data["parent_ids"][selected], labels=data["labels"][selected],
                        logits=predictions[f"{era}_{view}"])
            del values
        labels, parents = data["labels"][selected], data["parent_ids"][selected]
        recipe_keys = ("epochs", "batch_size", "optimizer", "learning_rate", "weight_decay",
                       "scheduler", "class_weight", "seed", "pretrained_sha256")
        for view in VIEWS:
            if any(configs[f"old_{view}"].get(key) != configs[f"new_{view}"].get(key) for key in recipe_keys):
                raise ValueError("Old/new fixed teacher recipe or ImageNet initialization differs")
        report = paired_summary(labels, parents, predictions)
        gaps = {}
        for view in VIEWS:
            for era, base, records in (("old", args.old_teachers, old_data), ("new", args.new_teachers, identities)):
                key = f"{era}_{view}"
                fresh = {weighting: values[key] for weighting, values in report["point"].items()}
                gaps[key] = own_train_gap(base / f"{view}-42", records, fresh)
        names = ("old-spatial-fresh-val.npz", "old-speed-fresh-val.npz")
        report.update({"complete": True, "classes": CLASSES, "generation_quality_evaluated": False,
            "selection": "Fixed final epoch50 seed42 teachers; no fitting or score-based model selection",
            "train_validation_diagnostics": gaps, "model_provenance": provenance,
            "input_cache_receipt_sha256": digest(args.cache / "complete.json"),
            "old_cache_identity_receipt_sha256": digest(args.old_cache / "complete.json"),
            "fresh_cache_groups": prepared["groups"], "speed_input_definition": prepared["speed"],
            "parent_isolation": {"old_train_new_val_overlap": 0, "new_train_new_val_overlap": 0,
                "old_val_new_val_overlap": len(set(old_data["parent_ids"][old_data["partition"] == "val"].astype(str)) & set(parents.astype(str))),
                "meaning": "Shared old-val identities are fresh random realizations, not unseen demand templates"},
            "training_budget": {key: {"epochs": value["epochs"], "batch_size": value["batch_size"],
                "train_legs": int(sum(value["class_counts"])), "class_counts": value["class_counts"],
                "optimizer_updates": int(value["epochs"] * np.ceil(sum(value["class_counts"]) / value["batch_size"]))}
                for key, value in configs.items()},
            "scale_claim_limit": "G35 versus G33 changes data volume, native demand/time coverage and CNN optimizer update count at fixed50epochs; paired common-val improvement is a corpus-plus-training-recipe result, not isolated sample-size causality",
            "GAF_claim_limit": "Real SUMO observation classification measures label information and reliability; it does not prove improved synthetic trajectories or guidance benefit",
            "code_sha256": digest(Path(__file__)), "seconds": time.monotonic() - started,
            "artifacts": {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size} for name in names}})
        guard.check(force=True, reserve=4 * 1024**2)
        save(output / "complete.json", report)
        print(json.dumps({"complete": True, "support": report["support"],
            "leg_accuracy_macro_f1": {key: [row["accuracy"], row["macro_f1"]] for key, row in report["point"]["leg"].items()}}), flush=True)
    except BaseException as error:
        save(output / "failed.json", {"complete": False, "error": repr(error), "seconds": time.monotonic() - started})
        raise


def self_test():
    import unittest

    class Checks(unittest.TestCase):
        def cohort(self):
            labels = np.repeat(np.arange(5), 3)
            parents = np.asarray([f"p{c}-{j // 2}" for c in range(5) for j in range(3)])
            perfect = np.eye(5)[labels] * 15.
            imperfect = perfect.copy()
            imperfect[np.arange(0, 15, 3)] = np.roll(imperfect[np.arange(0, 15, 3)], 1, axis=1)
            return labels, parents, perfect, imperfect

        def test_perfect_metrics_and_uniform_calibration(self):
            labels = np.arange(5)
            result = weighted_metrics(labels, np.eye(5) * 30, np.ones(5))
            self.assertEqual(result["accuracy"], 1)
            self.assertEqual(result["macro_f1"], 1)
            self.assertLess(result["nll"], 1e-10)
            uniform = weighted_metrics(labels, np.zeros((5, 5)), np.ones(5))
            self.assertAlmostEqual(uniform["nll"], np.log(5))
            self.assertAlmostEqual(uniform["brier_multiclass_sum"], .8)

        def test_multiple_legs_parent_weighting(self):
            labels, parents, _, imperfect = self.cohort()
            inverse, counts, parent_labels = parent_structure(labels, parents)
            weighted = weighted_metrics(labels, imperfect, 1. / counts[inverse])
            self.assertEqual(len(parent_labels), 10)
            self.assertAlmostEqual(weighted["total_weight"], 10)
            self.assertAlmostEqual(weighted["accuracy"], .75)
            ordinary = weighted_metrics(labels, imperfect, np.ones(len(labels)))
            self.assertAlmostEqual(ordinary["accuracy"], 2 / 3)

        def test_confusion_metric_vector_matches_full_metrics(self):
            labels, parents, _, imperfect = self.cohort()
            inverse, counts, _ = parent_structure(labels, parents)
            result = weighted_metrics(labels, imperfect, 1. / counts[inverse])
            np.testing.assert_allclose(confusion_vector(np.asarray(result["confusion"])), metric_vector(result))

        def test_invalid_nonfinite_and_parent_label(self):
            with self.assertRaises(ValueError):
                weighted_metrics(np.arange(5), np.full((5, 5), np.nan), np.ones(5))
            with self.assertRaises(ValueError):
                parent_structure(np.arange(5), ["same"] * 5)

        def test_prediction_alignment_rejects_reordering(self):
            data = {"labels": np.arange(5), "ids": np.asarray(list("abcde")), "parent_ids": np.asarray(list("ABCDE"))}
            selected = np.arange(5)
            assert_alignment(data, data["labels"], data["ids"], data["parent_ids"], selected)
            with self.assertRaises(ValueError):
                assert_alignment(data, data["labels"], data["ids"][::-1], data["parent_ids"], selected)

        def test_identical_models_have_exact_zero_paired_intervals(self):
            labels, parents, _, imperfect = self.cohort()
            predictions = {f"{era}_{view}": imperfect for era in ("old", "new") for view in VIEWS}
            report = paired_summary(labels, parents, predictions, replicates=80)
            for weighting in ("leg", "equal_parent"):
                for comparison in report["paired_comparisons"][weighting].values():
                    for value in comparison.values():
                        self.assertEqual(value["delta"], 0)
                        self.assertEqual(value["percentile_95"], [0, 0])

        def test_paired_gain_and_deterministic_bootstrap(self):
            labels, parents, perfect, imperfect = self.cohort()
            predictions = {f"{era}_{view}": perfect if era == "new" else imperfect
                           for era in ("old", "new") for view in VIEWS}
            first = paired_summary(labels, parents, predictions, replicates=80)
            second = paired_summary(labels, parents, predictions, replicates=80)
            self.assertEqual(first, second)
            gain = first["paired_comparisons"]["equal_parent"]["new_minus_old_spatial"]["accuracy"]
            self.assertAlmostEqual(gain["delta"], .25)
            self.assertGreaterEqual(gain["percentile_95"][0], 0)

        def test_missing_five_class_support_rejected(self):
            labels = np.zeros(3, dtype=np.int64)
            predictions = {f"{era}_{view}": np.zeros((3, 5)) for era in ("old", "new") for view in VIEWS}
            with self.assertRaises(ValueError):
                paired_summary(labels, ["a", "b", "c"], predictions, replicates=5)

        def test_encoder_provenance_not_rebound_to_new_teacher(self):
            required = {"g29_encode.py", "original_gaf.py", "hilbert.py", "sequence.py",
                        "representations.py", "g23_modes.py", "g28_teachers.py"}
            old = {"spatial": {"encoding_metadata_sha256": "metadata"},
                   "sources": {"code_sha256": {name: "old-shared-hash" for name in required}}}
            new = json.loads(json.dumps(old))
            new["sources"]["code_sha256"]["g35_teachers.py"] = "new-cache-code"
            assert_encoder_compatibility(old, new)
            new["sources"]["code_sha256"]["g23_modes.py"] = "changed-speed-encoding"
            with self.assertRaises(ValueError):
                assert_encoder_compatibility(old, new)

        def test_fixed_checkpoint_and_cache_provenance(self):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model, cache = root / "model", root / "cache"
                model.mkdir(); cache.mkdir()
                save(cache / "complete.json", {"fixture": True})
                config = {"epochs": 50, "checkpoint": "final_epoch_50_only", "seed": 42,
                    "view": "spatial", "classes": CLASSES,
                    "cache_receipt_sha256": digest(cache / "complete.json")}
                save(model / "config.json", config)
                names = ("model.pt", "train-predictions.npz", "val-predictions.npz")
                for name in names:
                    (model / name).write_bytes(b"fixture-only")
                receipt = {"complete": True, "selected_epoch": 50, "view": "spatial", "classes": CLASSES,
                    "config_sha256": digest(model / "config.json"),
                    "artifacts": {name: {"sha256": digest(model / name), "bytes": (model / name).stat().st_size} for name in names}}
                save(model / "complete.json", receipt)
                verify_teacher(model, "spatial", cache)
                save(cache / "complete.json", {"fixture": "new unrelated input cache"})
                with self.assertRaises(ValueError):
                    verify_teacher(model, "spatial", cache)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({"complete": True, "component": "G35 paired teacher audit checks",
                      "checks": result.testsRun, "real_model_inference_tested": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "self-test"))
    parser.add_argument("--cache", type=Path, default=Path("g35-data/gaf-cache"))
    parser.add_argument("--old-cache", type=Path, default=Path("../g33-job/g33-data/gaf-cache"))
    parser.add_argument("--old-teachers", type=Path, default=Path("../g33-job/g33-data/teachers"))
    parser.add_argument("--new-teachers", type=Path, default=Path("g35-data/teachers"))
    parser.add_argument("--output", type=Path, default=Path("g35-data/teacher-audit"))
    parser.add_argument("--budget-root", type=Path, default=Path("g35-data"))
    args = parser.parse_args()
    self_test() if args.action == "self-test" else run(args)
