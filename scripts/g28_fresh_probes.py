"""Evaluate fixed matched-label probes on independent G28 SUMO references.

Must run after synthesis evaluation is complete. No fitting, selection or tuning
uses the fresh references. Teacher features, PCA, scaling and probe coefficients
are fitted exclusively on the original training journeys.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from torch import nn

from g19_gaf import images
from g23_modes import speed_series
from g28_probes import MOTOR, fit_group, paired_inputs, route_features
from g28_teachers import classifier, image_inputs


def encode_batch(tracks):
    rgb, valid = images([track[:, 1:] for track in tracks])
    speed = [speed_series(track[:, 0], track[:, 1:]) for track in tracks]
    return rgb, valid, np.stack([row[0] for row in speed]), np.stack([row[1] for row in speed])


@torch.no_grad()
def extract(tracks, teachers, workers):
    models = {}
    for view in ("spatial", "speed"):
        path = teachers / f"{view}-42"
        if not (path / "complete.json").exists():
            raise FileNotFoundError(f"Teacher incomplete: {path}")
        model = classifier(42)
        model.load_state_dict(torch.load(path / "model.pt", weights_only=True, map_location="cuda"))
        model.fc = nn.Identity()
        models[view] = model.eval()
    output = {key: [] for key in ("spatial", "speed", "physical")}
    flags = []
    # Spawn avoids inheriting an initialized CUDA runtime into CPU image workers.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for start in range(0, len(tracks), 16 * workers):
            batches = [tracks[i:i + 16] for i in range(start, min(start + 16 * workers, len(tracks)), 16)]
            for rgb, valid, speed, stats in pool.map(encode_batch, batches):
                flags.extend(valid)
                output["spatial"].append(models["spatial"](image_inputs("spatial", rgb)).cpu().numpy())
                output["speed"].append(models["speed"](image_inputs("speed", speed)).cpu().numpy())
                output["physical"].append(stats)
    return {key: np.concatenate(rows) for key, rows in output.items()}, np.asarray(flags)


def evaluate_group(train_ids, labels, train, ids, truth, fresh, classes):
    result = fit_group(train_ids, labels, train, ids, truth, fresh, classes)
    result["evaluation_ids"] = result.pop("monitor_ids")
    result["evaluation_episodes"] = result.pop("monitor_episodes")
    if len(result["evaluation_episodes"]) != 8:
        raise ValueError("Expected eight independent simulation episodes")
    return result


def main(root, workers):
    if not (root / "evaluation-complete.txt").exists():
        raise RuntimeError("Synthesis evaluation must finish before fresh-reference probing")
    with np.load(root / "requests.npz") as source:
        requests = {key: source[key] for key in source.files}
    with np.load(root / "references.npz") as source:
        reference = {key: source[key] for key in source.files}
    if not np.array_equal(requests["ids"], reference["ids"]):
        raise ValueError("Request and reference IDs differ")
    train_ids, labels, train = paired_inputs(root, "train")
    if set(train_ids).intersection(requests["ids"]):
        raise ValueError("Fresh reference and training IDs overlap")
    tracks = [reference["txy"][a:b] for a, b in zip(reference["offset"][:-1], reference["offset"][1:])]
    fresh, valid = extract(tracks, root / "teachers", workers)
    if not valid.all():
        result = {"status": "encoding_failure", "requested": len(valid), "valid": int(valid.sum()),
                  "failed_ids": requests["ids"][~valid].tolist(),
                  "reason": "Matched classification must retain all preselected requests; no silent exclusions"}
    else:
        fresh["route"] = route_features(requests)
        result = {
            "status": "complete",
            "scope": "Independent same-map SUMO episodes 279001–279008; first six arrival-qualified requests per class and episode",
            "protocol": "Seed-42 frozen teachers, original training-only PCA64, scaling and balanced C=1 logistic regression; no fresh-data fitting or tuning",
            "interpretation": "Eight-episode paired bootstrap conditional on fitted teachers; joint input has twice each single latent dimension; readout improvement alone does not demonstrate better synthesis",
            "requests": len(valid), "encoding_valid": int(valid.sum()),
            "full8": evaluate_group(train_ids, labels, train, requests["ids"], requests["labels"], fresh, np.arange(8)),
            "motor6": evaluate_group(train_ids, labels, train, requests["ids"], requests["labels"], fresh, MOTOR),
        }
    path = root / "fresh-probes.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(path)
    print(json.dumps({"output": str(path), "status": result["status"]}), flush=True)
    if result["status"] != "complete":
        raise RuntimeError("Fresh-reference encoding audit failed; see fresh-probes.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("g28-data"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    torch.set_num_threads(2)
    main(args.root, args.workers)
