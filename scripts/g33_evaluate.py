"""Prospective G33 generation, then independent simulator comparison.

Three separate actions enforce ordering: prepare_maps reads only frozen public
requests; generate has no reference argument; evaluate opens reference targets
only after verifying a completed generated pool. Raw trajectories stay private.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import torch

from g32_routes import VehicleRouteNetwork, UnsupportedRoute
from g33_model import CLASSES, MotionDenoiser, clock, sample, sample_duration
from g31_roundtrip import signals, stats, UNITS


K = 8
DDIM_STEPS = 100
NEAREST_TRAIN = 8


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def artifact(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def completed(folder, names):
    record = json.loads((folder/"complete.json").read_text())
    if record.get("complete") is not True:
        raise ValueError("Completed input required: "+str(folder))
    for name in names:
        if digest(folder/name) != record["artifacts"][name]["sha256"]:
            raise ValueError("Immutable input artifact changed: "+str(folder/name))
    return record


def curve_xy(vertices, progress):
    vertices, progress = np.asarray(vertices, float), np.asarray(progress, float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))]
    if (len(vertices) < 2 or not np.isfinite(vertices).all() or np.any(np.diff(arc) <= 0)
            or not np.isfinite(progress).all() or np.any(progress < 0) or np.any(progress > 1)):
        raise ValueError("Invalid exact path/progress")
    return np.stack([np.interp(progress*arc[-1], arc, v) for v in vertices.T], axis=1)


def prepare_maps(prepared, network_path, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        preparation = json.loads((prepared/"prepare-complete.json").read_text())
        manifest = json.loads((prepared/"requests.json").read_text())
        if (not preparation["complete"] or digest(prepared/"requests.json") != preparation["requests_sha256"]
                or manifest["classes"] != CLASSES or len(manifest["requests"]) != 100):
            raise ValueError("Expected frozen 100-request five-class manifest")
        network = VehicleRouteNetwork(network_path)
        if network.network_sha256 != manifest["network_sha256"]:
            raise ValueError("Request map changed")
        rows, kept, vertices, offsets, path_conditions = [], [], [], [0], []
        for request in manifest["requests"]:
            row = {"id": request["id"], "label": request["label"], "class": request["class"]}
            if request["status"] != "request_ready":
                row.update(status="source_request_unresolved", reason=request.get("reason", "unknown"))
            else:
                try:
                    geom = network.construct(request)
                    route = geom.manifest()
                    if len(route["stop_progress_m"]) != len(request["stops"]):
                        raise ValueError("Planned service stops were lost")
                    xy = np.asarray(geom.vertices, np.float64)
                    length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
                    if abs(length-geom.length_m) > 1e-7:
                        raise ValueError("Map decoder lengths disagree")
                    row.update(status="map_ready", length_m=float(length), lane_changes=geom.lane_changes,
                               stop_count=len(route["stop_progress_m"]), vertex_count=len(xy))
                    kept.append(row); vertices.append(xy); offsets.append(offsets[-1]+len(xy))
                    path_conditions.append(curve_xy(xy, np.linspace(0, 1, 129)))
                except UnsupportedRoute as error:
                    row.update(status="map_unsupported", reason=str(error))
            rows.append(row)
        if not kept:
            raise ValueError("No prospective routes could be constructed")
        np.savez_compressed(output/"maps.npz", ids=np.asarray([r["id"] for r in kept]),
                            labels=np.asarray([r["label"] for r in kept], np.int64),
                            length=np.asarray([r["length_m"] for r in kept]),
                            paths=np.asarray(path_conditions, np.float32),
                            offsets=np.asarray(offsets, np.int64), vertices=np.concatenate(vertices))
        save(output/"request-status.json", {"rows": rows})
        result = {"complete": True, "phase": "prospective_maps", "classes": CLASSES, "denominator": 100,
                  "ready": len(kept), "status_counts": dict(Counter(r["status"] for r in rows)),
                  "by_class": {c: dict(Counter(r["status"] for r in rows if r["class"] == c)) for c in CLASSES},
                  "request_sha256": digest(prepared/"requests.json"), "network_sha256": network.network_sha256,
                  "constructor_sha256": digest(Path(inspect.getsourcefile(VehicleRouteNetwork))),
                  "script_sha256": digest(__file__), "reference_data_opened": False,
                  "scope": "Static permitted route geometry only; no traffic/signal/collision or generated dynamics claim",
                  "artifacts": {n: artifact(output/n) for n in ("maps.npz", "request-status.json")}}
        save(output/"complete.json", result)
        return result
    except Exception as error:
        save(output/"failed.json", {"complete": False, "error": repr(error)})
        raise


def training_records(folder):
    completed(folder, ["dataset.npz"])
    with np.load(folder/"dataset.npz", allow_pickle=False) as archive:
        partitions = archive["partition"].astype(str)
        if not set(partitions).issubset({"train", "val"}):
            raise ValueError("Frozen tests forbidden")
        ix = np.flatnonzero(partitions == "train")
        arrays = {k: archive[k][ix].copy() for k in ("ids", "labels", "length", "duration", "paths")}
        offsets, t, p = archive["offset"], archive["time"], archive["progress"]
        arrays["time"] = [t[offsets[i]:offsets[i+1]].copy() for i in ix]
        arrays["progress"] = [p[offsets[i]:offsets[i+1]].copy() for i in ix]
    return arrays


def transfer(request, training, prior, seed):
    key = int.from_bytes(hashlib.sha256(f"G33:{request['id']}:{seed}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(key)
    duration, bounded = sample_duration(prior, [request["length"]], [request["label"]], rng)
    eligible = np.flatnonzero(training["labels"] == request["label"])
    candidates = sorted(eligible, key=lambda i: (abs(np.log(training["length"][i]/request["length"])), str(training["ids"][i])))[:NEAREST_TRAIN]
    chosen = int(rng.choice(candidates))
    times = clock(float(duration[0]), "native_0_25s")
    p = np.interp(times/duration[0], training["time"][chosen]/training["duration"][chosen], training["progress"][chosen])
    return {"id": request["id"], "duration": float(duration[0]), "time": times, "progress": p,
            "duration_bounded": bool(bounded[0]), "negative_bins": 0, "upper_clipped_bins": 0,
            "constant_speed_fallback": False, "donor_training_id": str(training["ids"][chosen])}


def generate(maps, model_folder, training_folder, output, seed, family, device):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        map_receipt = completed(maps, ["maps.npz", "request-status.json"])
        model_receipt = json.loads((model_folder/"complete.json").read_text())
        if not model_receipt.get("complete") or digest(model_folder/"model.pt") != model_receipt["model_sha256"]:
            raise ValueError("Completed fixed model required")
        if digest(model_folder/"stats.json") != model_receipt["stats_sha256"]:
            raise ValueError("Training prior changed")
        if digest(model_folder/"config.json") != model_receipt["config_sha256"]:
            raise ValueError("Frozen training configuration changed")
        training_config = json.loads((model_folder/"config.json").read_text())
        if training_config["model_source_sha256"] != digest(Path(__file__).with_name("g33_model.py")):
            raise ValueError("Sampler model implementation differs from fixed training code")
        state = torch.load(model_folder/"model.pt", weights_only=False, map_location="cpu")
        if state["classes"] != CLASSES or (family == "diffusion" and state["seed"] != seed):
            raise ValueError("Model classes/seed differ from generation protocol")
        normalizer = state["stats"]
        if normalizer != json.loads((model_folder/"stats.json").read_text()):
            raise ValueError("Checkpoint and standalone training statistics disagree")
        training = training_records(training_folder) if family == "empirical" else None
        model = None
        forward_calls = {"n": 0}
        if family == "diffusion":
            model = MotionDenoiser(base=state["width"]).to(device).eval()
            model.load_state_dict(state["state_dict"])
            def count_forward(module, inputs, result):
                forward_calls["n"] += 1
            model.register_forward_hook(count_forward)
        elif digest(training_folder/"dataset.npz") != training_config["data_sha256"]:
            raise ValueError("Empirical donor data differs from diffusion training data")
        with np.load(maps/"maps.npz", allow_pickle=False) as f:
            geometry = {k: f[k].copy() for k in f.files}
        rows, offsets, trajectories = [], [0], []
        candidates = []
        with gzip.open(output/"generated.csv.gz", "wt", newline="") as stream, (output/"generation.jsonl").open("w", buffering=1) as log:
            writer = csv.writer(stream); writer.writerow(["time", "id", "x", "y", "type"])
            for i, ident in enumerate(geometry["ids"].astype(str)):
                vertices = geometry["vertices"][geometry["offsets"][i]:geometry["offsets"][i+1]]
                for candidate in range(K):
                    name = f"{ident}/candidate-{candidate:02d}"
                    request = {"id": name, "label": int(geometry["labels"][i]), "length": float(geometry["length"][i]), "path": geometry["paths"][i]}
                    row = {"id": name, "request_id": ident, "candidate": candidate, "label": request["label"]}
                    try:
                        draw = (sample(model, [request], normalizer, state["arm"], seed, device=device, ddim_steps=DDIM_STEPS)[0]
                                if family == "diffusion" else transfer(request, training, normalizer["duration_prior"], seed))
                        times = clock(draw["duration"], "native_0_25s")
                        progress = np.interp(times, draw["time"], draw["progress"])
                        xy = curve_xy(vertices, progress)
                        if not np.isfinite(xy).all() or np.any(np.diff(progress) < 0):
                            raise ValueError("Invalid decoded trajectory")
                        if not np.allclose(xy[[0, -1]], vertices[[0, -1]], atol=1e-8, rtol=0):
                            raise ValueError("Planned endpoint not preserved")
                        txy = np.c_[times, xy]
                        trajectories.append(txy); offsets.append(offsets[-1]+len(txy)); candidates.append(row)
                        row.update(status="generated", duration=draw["duration"], samples=len(times),
                                   duration_bounded=draw["duration_bounded"], negative_bins=draw["negative_bins"],
                                   upper_clipped_bins=draw["upper_clipped_bins"], constant_speed_fallback=draw["constant_speed_fallback"])
                        if "donor_training_id" in draw:
                            row["donor_training_id"] = draw["donor_training_id"]
                        for stamp, (x, y) in zip(times, xy):
                            writer.writerow([f"{stamp:.9f}", name, f"{x:.9f}", f"{y:.9f}", CLASSES[request["label"]]])
                    except ValueError as error:
                        row.update(status="candidate_failed", error=str(error))
                    rows.append(row)
                log.write(json.dumps({"requests_processed": i+1, "candidates_generated": len(trajectories), "elapsed_s": time.monotonic()-started})+"\n")
                print(json.dumps({"family": family, "seed": seed, "request": i+1, "total": len(geometry["ids"])}), flush=True)
        if not trajectories:
            raise ValueError("No candidates generated")
        np.savez_compressed(output/"samples.npz", ids=np.asarray([r["id"] for r in candidates]),
                            request_ids=np.asarray([r["request_id"] for r in candidates]),
                            labels=np.asarray([r["label"] for r in candidates], np.int64),
                            offsets=np.asarray(offsets, np.int64), txy=np.concatenate(trajectories))
        save(output/"candidate-status.json", {"rows": rows})
        result = {"complete": True, "phase": "prospective_generation", "family": family,
                  "arm": state["arm"] if family == "diffusion" else "training_motion_transfer", "seed": seed,
                  "classes": CLASSES, "requested_conditions": map_receipt["denominator"],
                  "map_ready_conditions": map_receipt["ready"], "requested_candidates_per_condition": K,
                  "candidate_denominator_all_requests": map_receipt["denominator"]*K,
                  "attempted_candidates": len(rows), "generated_candidates": len(trajectories),
                  "status_counts": dict(Counter(r["status"] for r in rows)),
                  "ddim_steps": DDIM_STEPS if family == "diffusion" else 0,
                  "network_forward_calls": forward_calls["n"],
                  "teacher_calls": 0, "nearest_training_pool": NEAREST_TRAIN if family == "empirical" else None,
                  "elapsed_s": time.monotonic()-started, "reference_data_opened": False,
                  "request_sha256": map_receipt["request_sha256"], "map_receipt_sha256": digest(maps/"complete.json"),
                  "training_model_receipt_sha256": digest(model_folder/"complete.json"),
                  "duration_prior_sha256": json_hash(normalizer["duration_prior"]), "script_sha256": digest(__file__),
                  "model_code_sha256": digest(Path(__file__).with_name("g33_model.py")),
                  "artifacts": {n: artifact(output/n) for n in ("samples.npz", "generated.csv.gz", "candidate-status.json")},
                  "limits": ["Route legality is inherited from a static map constructor, not learned collision/signal compliance.",
                             "Both diffusion representations and empirical transfer share the same sampled duration prior; duration differences are not a diffusion benefit.",
                             "All outputs use independent pre-simulation route requests; no observed reference trajectory/time is passed to generation."]}
        save(output/"complete.json", result)
        return result
    except Exception as error:
        save(output/"failed.json", {"complete": False, "error": repr(error), "elapsed_s": time.monotonic()-started})
        raise


def read_pool(folder, name):
    receipt = completed(folder, [name])
    with np.load(folder/name, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    return receipt, data


def extract_record(txy):
    time_axis = txy[:, 0]-txy[0, 0]
    xy = txy[:, 1:]
    if len(txy) < 2 or not np.isfinite(txy).all() or np.any(np.diff(time_axis) <= 0):
        raise ValueError("Bad trajectory in completed archive")
    physical, _, _, _, _, arc = signals(time_axis, xy)
    interval = np.diff(time_axis)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)/interval
    acceleration_interval = (interval[1:]+interval[:-1])/2
    acceleration = np.diff(speed)/acceleration_interval
    profile = np.interp(np.linspace(0, time_axis[-1], 257), time_axis, arc/max(arc[-1], 1e-12))
    aligned_xy = np.stack([np.interp(np.linspace(0, time_axis[-1], 129), time_axis, values) for values in xy.T], axis=1)
    return {"physical": physical, "speed": speed, "speed_weight": interval/interval.sum(),
            "acceleration": acceleration, "acceleration_weight": acceleration_interval/max(acceleration_interval.sum(), 1e-12),
            "profile": profile, "aligned_xy": aligned_xy}


def request_weights(rows):
    """Equal request mass; surviving candidates divide that request's mass."""
    keys = [r.get("request_id", str(i)) for i, r in enumerate(rows)]
    count = Counter(keys)
    return np.asarray([1./(len(count)*count[key]) for key in keys])


def matched_signal_rows(generated, reference, signal):
    a = [r for r in generated if len(r[signal])]
    b = [r for r in reference if len(r[signal])]
    if all("request_id" in r for r in a+b):
        common = {r["request_id"] for r in a} & {r["request_id"] for r in b}
        a = [r for r in a if r["request_id"] in common]
        b = [r for r in b if r["request_id"] in common]
    return a, b


def weighted_signal_distance(generated, reference, signal):
    a, b = matched_signal_rows(generated, reference, signal)
    if not a or not b:
        return None
    return float(wasserstein_distance(np.concatenate([r[signal] for r in a]), np.concatenate([r[signal] for r in b]),
                 np.concatenate([r[signal+"_weight"]*weight for r, weight in zip(a, request_weights(a))]),
                 np.concatenate([r[signal+"_weight"]*weight for r, weight in zip(b, request_weights(b))])))


def evaluate(generated_folder, reference_folder, maps, training_folder, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        # The generator receipt is mandatory and checked before any reference read.
        generation, pool = read_pool(generated_folder, "samples.npz")
        if generation["reference_data_opened"] is not False:
            raise ValueError("Generation data permissions were not separated")
        map_receipt = completed(maps, ["maps.npz", "request-status.json"])
        reference_receipt, reference = read_pool(reference_folder, "references.npz")
        if (reference_receipt["source_sha256"]["requests"] != generation["request_sha256"]
                or map_receipt["request_sha256"] != generation["request_sha256"]):
            raise ValueError("Reference and generated requests differ")
        train = training_records(training_folder)
        ref_rows = {str(ident): dict(extract_record(reference["txy"][reference["offsets"][i]:reference["offsets"][i+1]]),
                                   label=int(reference["labels"][i]), request_id=str(ident)) for i, ident in enumerate(reference["ids"])}
        gen_rows = []
        for i, ident in enumerate(pool["ids"].astype(str)):
            r = extract_record(pool["txy"][pool["offsets"][i]:pool["offsets"][i+1]])
            r.update(id=ident, request_id=str(pool["request_ids"][i]), label=int(pool["labels"][i]))
            gen_rows.append(r)
        reference_ids = set(ref_rows)
        generator_ids = {r["request_id"] for r in gen_rows}
        common = reference_ids & generator_ids
        class_results = {}
        for c, name in enumerate(CLASSES):
            g = [r for r in gen_rows if r["label"] == c and r["request_id"] in common]
            rr = [ref_rows[i] for i in sorted(common) if ref_rows[i]["label"] == c]
            for r in g:
                if ref_rows[r["request_id"]]["label"] != c:
                    raise ValueError("Class mismatch")
            metrics = {}
            for key, unit in UNITS.items():
                supported_g = [r for r in g if r["physical"][key] is not None]
                supported_r = [r for r in rr if r["physical"][key] is not None]
                supported_requests = {r["request_id"] for r in supported_g} & {r["request_id"] for r in supported_r}
                ag = [r for r in supported_g if r["request_id"] in supported_requests]
                br = [r for r in supported_r if r["request_id"] in supported_requests]
                a, b = [r["physical"][key] for r in ag], [r["physical"][key] for r in br]
                per_request_error = [float(np.mean([abs(r["physical"][key]-ref_rows[ident]["physical"][key])
                                                    for r in ag if r["request_id"] == ident])) for ident in sorted(supported_requests)]
                metrics[key] = {"unit": unit, "generated_supported": len(a), "reference_supported": len(b),
                    "generated_supported_before_request_matching": len(supported_g), "reference_supported_before_request_matching": len(supported_r),
                    "paired_supported_requests": len(supported_requests), "requests_without_paired_support": len(rr)-len(supported_requests),
                    "w1": float(wasserstein_distance(a, b, request_weights(ag), request_weights(br))) if a and b else None,
                    "generated_mean": float(np.average(a, weights=request_weights(ag))) if a else None,
                    "reference_mean": float(np.average(b, weights=request_weights(br))) if b else None,
                    "matched_request_mean_absolute_error": stats(per_request_error),
                    "matched_request_candidate_absolute_error": stats(abs(r["physical"][key]-ref_rows[r["request_id"]]["physical"][key])
                        if r["physical"][key] is not None and ref_rows[r["request_id"]]["physical"][key] is not None else None for r in g)}
            class_results[name] = {"reference_requests": len(rr), "generated_candidates": len(g), "physical": metrics,
                "native_speed_w1_mps": weighted_signal_distance(g, rr, "speed"),
                "native_acceleration_w1_mps2": weighted_signal_distance(g, rr, "acceleration"),
                "native_speed_supported_requests": len({r["request_id"] for r in matched_signal_rows(g, rr, "speed")[0]}),
                "native_acceleration_supported_requests": len({r["request_id"] for r in matched_signal_rows(g, rr, "acceleration")[0]})}
        diversity, diversity_requests, nearest = [], [], []
        diversity_keys = ("position_mean_separation_m", "relative_progress_mean_separation", "duration_difference_s")
        for ident in sorted(generator_ids):
            group = [r for r in gen_rows if r["request_id"] == ident]
            request_pairs = []
            for i, a in enumerate(group):
                for b in group[i+1:]:
                    request_pairs.append({"position_mean_separation_m": float(np.linalg.norm(a["aligned_xy"]-b["aligned_xy"], axis=1).mean()),
                                      "relative_progress_mean_separation": float(np.abs(a["profile"]-b["profile"]).mean()),
                                      "duration_difference_s": abs(a["physical"]["duration_s"]-b["physical"]["duration_s"])})
            diversity.extend(request_pairs)
            diversity_requests.append({key: float(np.mean([r[key] for r in request_pairs])) if request_pairs else None for key in diversity_keys})
        for c in range(len(CLASSES)):
            gi = [r for r in gen_rows if r["label"] == c]
            ti = np.flatnonzero(train["labels"] == c)
            tp = np.stack([np.interp(np.linspace(0, train["duration"][i], 257), train["time"][i], train["progress"][i]) for i in ti])
            if not gi:
                continue
            distance = cdist(np.stack([r["profile"] for r in gi]), tp, metric="cityblock")/257
            ni = distance.argmin(1)
            nearest.extend({"relative_progress_mae": float(distance[j, k]),
                            "duration_difference_s_to_shape_nearest": abs(r["physical"]["duration_s"]-float(train["duration"][ti[k]]))}
                           for j, (r, k) in enumerate(zip(gi, ni)))
        with np.load(maps/"maps.npz", allow_pickle=False) as f:
            route_overlap = []
            for p, label in zip(f["paths"], f["labels"]):
                candidates = train["paths"][train["labels"] == label]
                route_overlap.append(float(np.linalg.norm(candidates-p[None], axis=-1).mean(1).min()))
        all_summary = {name: stats(r["physical"][name] for r in gen_rows) for name in UNITS}
        supported_classes = {name: [c for c in CLASSES if class_results[c]["physical"][name]["w1"] is not None] for name in UNITS}
        available_macro = {name: float(np.mean([class_results[c]["physical"][name]["w1"] for c in supported_classes[name]]))
                           if supported_classes[name] else None for name in UNITS}
        macro = {name: available_macro[name] if len(supported_classes[name]) == len(CLASSES) else None for name in UNITS}
        result = {"complete": True, "scope": "single_new_SUMO_episode_descriptive_prospective_generation_evaluation",
                  "family": generation["family"], "arm": generation["arm"], "seed": generation["seed"], "classes": CLASSES,
                  "selection_method": generation.get("selection_method"),
                  "coverage": {"requested": 100, "map_ready": map_receipt["ready"], "reference_complete": len(reference_ids),
                               "requests_with_generated_candidates": len(generator_ids), "paired_reference_and_generation": len(common),
                               "generated_candidates_all": len(gen_rows), "generated_candidates_paired": sum(r["request_id"] in common for r in gen_rows),
                               "map_status_counts": map_receipt["status_counts"], "reference_status_counts": reference_receipt["status_counts"]},
                  "all_generated_physical": all_summary, "paired_by_class": class_results, "paired_macro_w1": macro,
                  "paired_macro_supported_classes": supported_classes,
                  "paired_available_class_macro_w1_secondary": available_macro,
                  "within_request_diversity": {key: stats(r[key] for r in diversity_requests) for key in diversity_keys},
                  "candidate_pair_diversity_descriptive": {key: stats(r[key] for r in diversity) for key in diversity_keys},
                  "training_shape_nearest": {key: stats(r[key] for r in nearest) for key in
                        ("relative_progress_mae", "duration_difference_s_to_shape_nearest")},
                  "condition_route_overlap_with_training_mae_m": stats(route_overlap),
                  "method": {"physical": "G31 ten endpoints on one-second cumulative-coordinate-arc clock; each metric matches supported requests, equal request mass then equal surviving candidate mass within request",
                             "native_w1": "Exact weighted W1; matched supported requests equal weight, candidates within request equal weight, then each native interval weighted by duration",
                             "diversity": "Mean over candidate pairs within each request, then summaries over requests;129 normalized-time coordinates and257 progress points; raw pair summaries separately descriptive",
                             "nearcopy": "Nearest same-class train progress MAE at257 relative-time positions, plus duration difference; descriptive, no privacy threshold",
                             "route_overlap": "Conditioning geometry overlap, not evidence that the motion model memorized trajectories",
                             "confidence_intervals": "none; one simulation episode, fixed route templates and two model seeds reported separately"},
                  "sources": {"generation_receipt_sha256": digest(generated_folder/"complete.json"),
                              "reference_receipt_sha256": digest(reference_folder/"complete.json"),
                              "map_receipt_sha256": digest(maps/"complete.json"), "training_dataset_sha256": digest(training_folder/"dataset.npz"),
                              "script_sha256": digest(__file__)},
                  "limits": ["Generation and reference failures remain in the100-request denominator; paired metrics cover only their intersection.",
                             "Static route-constructor legality and exact endpoint decoding do not guarantee collision/signal or vehicle-body safety.",
                             "Same duration prior and candidate IDs are shared; differences in duration alone cannot establish a motion-diffusion benefit.",
                             "Fixed-route generation tests motion, not novel route-choice or spatial-GAF efficacy.",
                             "Five-class macro W1 is undefined when any class lacks paired support; available-class values and class lists are separate secondary summaries.",
                             ("Independent physical evaluation after frozen GAF terminal selection; the selection teacher is not a physical quality metric."
                              if generation.get("phase") == "terminal_gaf_selection" else
                              "No GAF selection or teacher score participates in this baseline evaluation.")]}
        save(output/"complete.json", result)
        return result
    except Exception as error:
        save(output/"failed.json", {"complete": False, "error": repr(error)})
        raise


def self_test():
    vertices = np.array([[0., 0.], [3., 0.], [3., 4.]])
    xy = curve_xy(vertices, [0., 3/7, 1.])
    assert np.array_equal(xy, vertices)
    t = np.array([0., .25, .5, .625])
    a = extract_record(np.c_[t, t*2, np.zeros(len(t))])
    b = extract_record(np.c_[t, t*3, np.zeros(len(t))])
    assert np.isclose(weighted_signal_distance([a], [b], "speed"), 1.)
    assert weighted_signal_distance([a], [a], "speed") == 0.
    assert a["physical"]["mean_speed_mps"] is None  # Short trips remain explicitly unsupported at1s.
    assert a["profile"][0] == 0 and a["profile"][-1] == 1
    # Eight surviving candidates for request A must not outweigh one for B.
    a.update(request_id="A"); b.update(request_id="B")
    assert weighted_signal_distance([a]*8+[b], [a, b], "speed") < 1e-10
    assert wasserstein_distance([2.]*8+[3.], [2., 3.]) > .3
    assert abs(wasserstein_distance([2.]*8+[3.], [2., 3.], request_weights([a]*8+[b]), request_weights([a, b]))) < 1e-10
    print("g33_evaluation_checks_passed:exact_polyline,physical_fractional_clock,timeweighted_W1,shortsupport,unequal_candidate_request_weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare_maps", "generate", "evaluate", "checks"))
    parser.add_argument("--prepared", type=Path, default=Path("g33-data/requests"))
    parser.add_argument("--network", type=Path, default=Path("../g30-job/g30-data/source/scenario/in/most.net.xml"))
    parser.add_argument("--maps", type=Path, default=Path("g33-data/maps"))
    parser.add_argument("--training", type=Path, default=Path("g33-data/direct-legs"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("g33-data/reference-extracted"))
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42)
    parser.add_argument("--family", choices=("diffusion", "empirical"), default="diffusion")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action != "checks" and args.output is None:
        parser.error("A fresh output directory is required")
    if args.action == "generate" and args.model is None:
        parser.error("Fixed trained model folder required")
    if args.action == "evaluate" and args.generated is None:
        parser.error("Completed generation folder required")
    result = (self_test() if args.action == "checks" else
              prepare_maps(args.prepared, args.network, args.output) if args.action == "prepare_maps" else
              generate(args.maps, args.model, args.training, args.output, args.seed, args.family, args.device) if args.action == "generate" else
              evaluate(args.generated, args.reference, args.maps, args.training, args.output))
    if result is not None:
        print(json.dumps({k: result[k] for k in ("complete", "phase", "scope") if k in result}), flush=True)
