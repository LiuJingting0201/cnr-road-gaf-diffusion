"""Aggregate fixed G30 fits; share resampling weights for shared source entities."""
import json
from pathlib import Path
import numpy as np

ROOT = Path("g30-data")
DOMAINS = ("cnr", "sumo", "native")


def metrics(cm):
    return {"macro_f1": np.mean(2*np.diagonal(cm, axis1=-2, axis2=-1) /
                                np.maximum(cm.sum(-1)+cm.sum(-2), 1e-30), axis=-1),
            "accuracy": np.trace(cm, axis1=-2, axis2=-1)/cm.sum(axis=(-1, -2))}


def run():
    transfer = json.loads((ROOT/"transfer.json").read_text())
    assert transfer["complete"]
    rows, keys = {}, {}
    for domain in DOMAINS:
        with np.load(ROOT/"raw"/f"{domain}-val.npz") as data:
            rows[domain] = {"ids": data["ids"], "labels": data["labels"]}
            base = data["base_ids"] if domain == "native" else data["ids"]
            keys[domain] = [("random:" if domain == "sumo" else "source:")+str(x) for x in base]
        assert len(set(keys[domain])) == len(keys[domain])
    expected = {f"{domain}-{seed}" for domain in DOMAINS for seed in (42,43)} | {
        f"frozen-g29-cnr-{seed}" for seed in (42,43)}
    assert set(transfer["models"]) == expected
    for model in expected:
        assert set(transfer["models"][model]) == set(DOMAINS)
        for domain in DOMAINS:
            with np.load(ROOT/"predictions"/f"{model}-on-{domain}.npz") as p:
                assert np.array_equal(p["ids"],rows[domain]["ids"])
                assert np.array_equal(p["labels"],rows[domain]["labels"])
                assert np.isfinite(p["probabilities"]).all()
                cm = np.bincount(6*p["labels"]+p["probabilities"].argmax(1), minlength=36).reshape(6,6)
                saved = transfer["models"][model][domain]
                assert np.array_equal(cm,saved["confusion"])
                assert all(np.isclose(v,saved[k],atol=1e-7) for k,v in metrics(cm).items())
    union = sorted(set().union(*map(set, keys.values())))
    lookup = {key: i for i, key in enumerate(union)}
    weights = np.random.default_rng(30055).exponential(size=(2000, len(union)))
    domain_weights = {d: weights[:, [lookup[k] for k in keys[d]]] for d in DOMAINS}
    reused = len(list((ROOT/"models").glob("reused-*.json")))
    result = {"complete": True, "model_slots": 6, "new_fitted_models": 6-reused,
              "reused_identical_cohort_models": reused, "frozen_teacher_models": 2,
              "scope": "Fixed final50 epochs, two seeds; no generation fit or thesis update",
              "validation_checks": {"eight_transfer_models_and_24_prediction_sets": True,
                  "fixed_fit_configs": True, "train_validation_prediction_metrics": True},
              "uncertainty": {"method": "shared-entity weighted Bayesian bootstrap, 2000 Exp(1) draws, seed30055",
                  "shared_cnr_native_validation_base_ids": len(set(keys["cnr"]) & set(keys["native"])),
                  "limitation": "Conditional on observed entities, fixed models and public demand; not confidence coverage for new routes, cities or training randomness"},
              "models": {}, "source_comparisons": {}, "transfer": transfer["models"]}
    boot = {}
    for source in DOMAINS:
        reports = [json.loads((ROOT/"models"/f"{source}-full-gaf-{s}"/"complete.json").read_text()) for s in (42, 43)]
        for seed, r in zip((42,43), reports):
            folder = ROOT/"models"/f"{source}-full-gaf-{seed}"
            config = json.loads((folder/"config.json").read_text())
            assert r["complete"] and r["epochs"] == 50 and r["source"] == source
            assert r["seed"] == seed and r["arm"] == "gaf" and r["view"] == "full"
            assert config["epochs"] == 50 and config["seed"] == seed and config["batch_size"] == 64
            assert config["learning_rate"] == 2e-4 and config["weight_decay"] == 1e-4
            for split, part in (("train", "train"), ("val", "validation")):
                with np.load(folder/f"{split}.npz") as p, np.load(ROOT/"raw"/f"{source}-{split}.npz") as raw:
                    assert np.array_equal(p["ids"],raw["ids"]) and np.array_equal(p["labels"],raw["labels"])
                    cm = np.bincount(6*p["labels"]+p["probabilities"].argmax(1),minlength=36).reshape(6,6)
                    assert np.array_equal(cm, r[part]["confusion"])
                    computed = metrics(cm)
                    assert all(np.isclose(computed[k], r[part][k], atol=1e-7) for k in computed)
        means = {part: {field: np.mean([r[part][field] for r in reports], axis=0).tolist()
                        for field in ("accuracy", "macro_f1", "per_class_f1", "per_class_recall", "confusion")}
                 for part in ("train", "validation")}
        result["models"][source] = {"mean": means, "by_seed": {str(s): r for s, r in zip((42, 43), reports)}}
        seed_boot = []
        for s, report in zip((42, 43), reports):
            with np.load(ROOT/"models"/f"{source}-full-gaf-{s}"/"val.npz") as prediction:
                assert np.array_equal(prediction["ids"], rows[source]["ids"])
                assert np.array_equal(prediction["labels"], rows[source]["labels"])
                category = 6*prediction["labels"] + prediction["probabilities"].argmax(1)
                design = np.eye(36)[category]
                cm = design.sum(0).reshape(6, 6)
                assert np.allclose(cm, report["validation"]["confusion"])
                seed_boot.append(metrics((domain_weights[source] @ design).reshape(-1, 6, 6)))
        boot[source] = {field: np.mean([b[field] for b in seed_boot], axis=0) for field in ("macro_f1", "accuracy")}
        result["models"][source]["bayesian_interval_95"] = {field: np.quantile(value, [.025,.975]).tolist()
                                                               for field, value in boot[source].items()}
    for left, right in (("native", "sumo"), ("cnr", "native"), ("cnr", "sumo")):
        key = f"{left}-minus-{right}"
        result["source_comparisons"][key] = {field: {
            "point": result["models"][left]["mean"]["validation"][field]-result["models"][right]["mean"]["validation"][field],
            "bayesian_interval_95": np.quantile(boot[left][field]-boot[right][field], [.025,.975]).tolist()}
            for field in ("macro_f1", "accuracy")}
    (ROOT/"summary.tmp.json").write_text(json.dumps(result, indent=2)+"\n")
    (ROOT/"summary.tmp.json").replace(ROOT/"summary.json")
    print(json.dumps({domain: result["models"][domain]["mean"]["validation"]["macro_f1"] for domain in DOMAINS}), flush=True)


if __name__ == "__main__":
    run()
