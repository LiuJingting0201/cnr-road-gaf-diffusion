"""Check roads and secondary seed-43 teacher readouts after all pools exist."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import shapely
import torch

from cnr.topology import LaneConsistency
from g16_data import NAMES
from g19_gaf import images
from g22_diffusion import decode
from g23_modes import speed_series
from g28_teachers import classifier, image_inputs, metrics
from g28_train import ARMS

ROOT = Path("g28-data")


@lru_cache(maxsize=1)
def matcher():
    return LaneConsistency("data/maps/most.net.xml", radius=5, max_candidates=64)


def encode_and_check(job):
    name, indices, labels, paths, tracks = job
    spatial, spatial_valid = images([track[:, 1:] for track in tracks])
    velocities, speed_valid, passes = [], [], []
    for label, path, track in zip(labels, paths, tracks):
        try:
            velocity, _ = speed_series(track[:, 0], track[:, 1:])
            valid = bool(np.isfinite(velocity).all())
        except ValueError:
            velocity, valid = np.zeros(128, dtype=np.float32), False
        velocities.append(velocity)
        speed_valid.append(valid)
        if name == "reference":
            passes.append(False)
        elif label == 0:
            distance = shapely.distance(shapely.points(track[:, 1:]), shapely.LineString(path))
            passes.append(bool(np.max(distance) < .02))
        else:
            result = matcher().match(track[:, 1:], NAMES[label])
            passes.append(bool(result["graph_consistent"] and not result["candidate_truncation_warning"]))
    return {"name": name, "indices": indices, "labels": labels,
            "spatial": spatial, "spatial_valid": spatial_valid,
            "speed": np.asarray(velocities), "speed_valid": np.asarray(speed_valid),
            "road_pass": np.asarray(passes)}


def jobs(request, references):
    count = len(request["ids"])
    offsets = references["offset"]
    observed = [references["txy"][start:end] for start, end in zip(offsets[:-1], offsets[1:])]
    for base in range(0, count, 16):
        indices = np.arange(base, min(base + 16, count))
        yield ("reference", indices, request["labels"][indices], request["paths"][indices],
               observed[base:base + 16])
    for arm in ARMS:
        with np.load(ROOT / "models" / f"{arm}-42" / "samples.npz") as data:
            if not np.array_equal(data["ids"], request["ids"]):
                raise ValueError(f"Sample IDs differ for {arm}")
            duration, progress = data["duration"][:, 0], data["progress"][:, 0]
        for base in range(0, count, 16):
            indices = np.arange(base, min(base + 16, count))
            tracks = []
            for index in indices:
                times, xy = decode(request["paths"][index], duration[index], progress[index])
                tracks.append(np.column_stack((times, xy)))
            yield arm, indices, request["labels"][indices], request["paths"][indices], tracks


def summarize_readout(labels, logits, valid):
    if not valid.any():
        return {"attempts": len(labels), "encoded": 0, "failure_count": len(labels)}
    score = metrics(labels[valid], logits[valid])
    probability = torch.from_numpy(logits).softmax(1).numpy()
    requested = probability[np.arange(len(labels)), labels]
    byclass = {}
    for label, name in enumerate(NAMES):
        use = (labels == label) & valid
        byclass[name] = {"attempts": int(np.sum(labels == label)), "encoded": int(use.sum()),
                         "target_probability": float(requested[use].mean()) if use.any() else None,
                         "recall": float(np.mean(logits[use].argmax(1) == label)) if use.any() else None}
    score.update(attempts=len(labels), encoded=int(valid.sum()), failure_count=int((~valid).sum()),
                 target_probability_macro=float(np.mean([row["target_probability"] for row in byclass.values()
                                                         if row["target_probability"] is not None])),
                 byclass=byclass)
    return score


@torch.no_grad()
def run():
    # Check every pool before opening any observed evaluation trajectory.
    for arm in ARMS:
        for seed in (42, 43):
            directory = ROOT / "models" / f"{arm}-{seed}"
            if not (directory / "complete.json").exists() or not (directory / "samples.npz").exists():
                raise RuntimeError(f"All frozen sample pools must exist before readouts: {directory}")
    for view in ("spatial", "speed"):
        if not (ROOT / "teachers" / f"{view}-43" / "complete.json").exists():
            raise RuntimeError(f"Seed-43 {view} teacher incomplete")
    with np.load(ROOT / "requests.npz") as data:
        request = {key: data[key] for key in data.files}
    with np.load(ROOT / "references.npz") as data:
        references = {key: data[key] for key in data.files}
    if not np.array_equal(request["ids"], references["ids"]):
        raise ValueError("Reference IDs differ from requests")
    torch.set_num_threads(1)
    networks = {}
    for view in ("spatial", "speed"):
        net = classifier(43)
        net.load_state_dict(torch.load(ROOT / "teachers" / f"{view}-43" / "model.pt",
                                       map_location="cuda", weights_only=True))
        networks[view] = net.eval()
    count = len(request["ids"])
    tables = {name: {"spatial_logits": np.empty((count, 8), dtype=np.float32),
                     "speed_logits": np.empty((count, 8), dtype=np.float32),
                     "spatial_valid": np.zeros(count, dtype=bool),
                     "speed_valid": np.zeros(count, dtype=bool),
                     "road_pass": np.zeros(count, dtype=bool),
                     "seen": np.zeros(count, dtype=bool)} for name in ("reference", *ARMS)}
    with ProcessPoolExecutor(max_workers=8, mp_context=get_context("spawn")) as pool:
        for batch in pool.map(encode_and_check, jobs(request, references)):
            table, indices = tables[batch["name"]], batch["indices"]
            if table["seen"][indices].any():
                raise ValueError("Duplicate scoring batch")
            table["seen"][indices] = True
            table["road_pass"][indices] = batch["road_pass"]
            for view, network in networks.items():
                table[f"{view}_logits"][indices] = network(image_inputs(view, batch[view])).cpu().numpy()
                table[f"{view}_valid"][indices] = batch[f"{view}_valid"]
    output = {"scope": "one predeclared seed-42 slot-0 candidate per request",
              "reader_independence": "separate seed-43 fits, same architecture and training corpus as seed-42 teachers",
              "reference_spatial_input": "raw native XY observations, no prescribed endpoint appended",
              "generated_spatial_input": "decoded delivered XY including exact sampled arrival",
              "road_check": "directed lane graph for vehicles; prescribed-sidewalk proximity for pedestrians",
              "requests": count, "arms": {}}
    labels = request["labels"]
    for name, table in tables.items():
        if not table["seen"].all():
            raise ValueError(f"Incomplete scoring for {name}")
        record = {view: summarize_readout(labels, table[f"{view}_logits"], table[f"{view}_valid"])
                  for view in ("spatial", "speed")}
        if name != "reference":
            record["road"] = {"attempts": count, "passed": int(table["road_pass"].sum()),
                              "byclass": {label: {"attempts": int(np.sum(labels == index)),
                                                   "passed": int(table["road_pass"][labels == index].sum())}
                                          for index, label in enumerate(NAMES)}}
        output["arms"][name] = record
        np.savez_compressed(ROOT / f"readout-{name}.npz", ids=request["ids"], labels=labels,
                            **{key: value for key, value in table.items() if key != "seen"})
    temporary = ROOT / "checks.tmp.json"
    temporary.write_text(json.dumps(output, indent=2))
    temporary.replace(ROOT / "checks.json")


if __name__ == "__main__":
    run()
