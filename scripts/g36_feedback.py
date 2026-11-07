"""Frozen terminal Uniform/S/V/SV feedback on exactly the same G33 pools.

Selection never opens a reference trajectory. Evaluation is a separate action.
All requested slots remain in each pool. Any missing/failed encoding causes all
four selectors to use the same predetermined Uniform slot; no rescoring,
resampling, alternate teacher, or candidate removal is permitted.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from g33_prepare import CLASSES, digest, save


METHODS = ("Uniform", "S", "V", "SV")
UNIFORM_SEED = 33073
PROBABILITY_FLOOR = 1e-8


def permutation(request_id, size):
    key = hashlib.sha256(f"G33-terminal:{UNIFORM_SEED}:{request_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "little")).permutation(size)


def choose(spatial, speed, order, complete=True):
    spatial, speed = np.asarray(spatial, float), np.asarray(speed, float)
    order = np.asarray(order, np.int64)
    if (spatial.ndim != 1 or speed.shape != spatial.shape or len(order) != len(spatial)
            or set(order.tolist()) != set(range(len(spatial))) or not len(order)):
        raise ValueError("Candidate vectors/permutation differ")
    if not complete:
        return {name: int(order[0]) for name in METHODS}
    if (not np.isfinite(spatial).all() or not np.isfinite(speed).all()
            or np.any(spatial < 0) or np.any(speed < 0) or np.any(spatial > 1) or np.any(speed > 1)):
        raise ValueError("Nonfinite/invalid complete-pool probabilities")
    s, v = np.log(np.maximum(spatial, PROBABILITY_FLOOR)), np.log(np.maximum(speed, PROBABILITY_FLOOR))
    result = {"Uniform": int(order[0])}
    for name, values in (("S", s), ("V", v), ("SV", .5*s+.5*v)):
        # Exact ties follow the same random order, without a tuned tolerance.
        result[name] = int(order[int(np.argmax(values[order]))])
    return result


def spatial_sequence(xy, metadata):
    """Same original spatial transform; timestamps are not image features.

    Unlike raw-observation validation in G29, a generated trajectory may end
    with a fractional final .25s interval. Keep that real endpoint, and encode
    its coordinates directly rather than fabricating regular timestamps.
    """
    from cnr.representations import hilbert_numpy
    from cnr.original_gaf import squeeze, duplicate
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError("Finite generated XY sequence required")
    low, high = np.asarray(metadata["xy_min"]), np.asarray(metadata["xy_max"])
    unit = (xy-low)/(high-low)
    clipped = int(np.any((unit < 0) | (unit > 1), axis=1).sum())
    native = hilbert_numpy(unit.clip(0, 1), metadata["hilbert_eps"])
    sequence = np.asarray(duplicate(squeeze(native.tolist(), 224), 224), dtype=np.float64)
    if len(sequence) != 224:
        raise ValueError("Original inherited squeeze cannot yield224 samples")
    return sequence, clipped


def code_hashes():
    from g33_teachers import source_hashes
    from g33_evaluate import extract_record
    return {**source_hashes(), "g36_feedback.py": digest(Path(__file__)),
            "g33_evaluate.py": digest(Path(inspect.getsourcefile(extract_record)))}


def completed(folder, names):
    receipt = json.loads((folder/"complete.json").read_text())
    if receipt.get("complete") is not True:
        raise ValueError("Completed input required: "+str(folder))
    for name in names:
        if digest(folder/name) != receipt["artifacts"][name]["sha256"]:
            raise ValueError("Frozen artifact hash differs: "+str(folder/name))
    return receipt


def load_reader(folder, cache_receipt_hash, expected_view):
    import torch
    from torch import nn
    from torchvision.models import resnet18
    receipt = completed(folder, ["model.pt"])
    config = json.loads((folder/"config.json").read_text())
    if (receipt["classes"] != CLASSES or config["classes"] != CLASSES or receipt["selected_epoch"] != 50
            or receipt["view"] != expected_view or config["view"] != expected_view
            or digest(folder/"config.json") != receipt["config_sha256"]
            or config["cache_receipt_sha256"] != cache_receipt_hash
            or config["seed"] != 42 or config["checkpoint"] != "final_epoch_50_only"):
        raise ValueError("Fixed matched final teacher contract differs")
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, len(CLASSES))
    model.load_state_dict(torch.load(folder/"model.pt", weights_only=True, map_location="cpu"))
    return model.cuda().eval().requires_grad_(False), receipt


def score(args):
    import torch
    from g23_modes import speed_series
    from g28_teachers import image_inputs
    from cnr.original_gaf import encode_frozen
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        generation = completed(args.generated, ["samples.npz", "candidate-status.json"])
        cache = completed(args.cache, ["index.npz"])
        if (generation["phase"] != "prospective_generation" or generation["reference_data_opened"] is not False
                or generation["classes"] != CLASSES or cache["classes"] != CLASSES):
            raise ValueError("Prospective complete candidate pool and matched teachers required")
        from g33_teachers import source_hashes
        required = {"g29_encode.py", "original_gaf.py", "hilbert.py", "sequence.py", "representations.py", "g23_modes.py", "g28_teachers.py"}
        current = source_hashes()
        if any(cache["sources"]["code_sha256"].get(n) != current.get(n) or n not in current for n in required):
            raise ValueError("Frozen inherited image/normalization code changed")
        metadata = cache["spatial"]["frozen_parameters"]
        with np.load(args.generated/"samples.npz", allow_pickle=False) as f:
            pool = {key: f[key] for key in f.files}
        status = json.loads((args.generated/"candidate-status.json").read_text())["rows"]
        K = int(generation["requested_candidates_per_condition"])
        groups = defaultdict(list)
        for row in status:
            groups[row["request_id"]].append(row)
        if len(groups) != generation["map_ready_conditions"] or K != 8:
            raise ValueError("Fixed G33 K=8/map request denominator changed")
        for group in groups.values():
            group.sort(key=lambda r: r["candidate"])
            if len(group) != K or [r["candidate"] for r in group] != list(range(K)) or len({r["label"] for r in group}) != 1:
                raise ValueError("Requested candidate slots or class inconsistent")
        lookup = {str(ident): i for i, ident in enumerate(pool["ids"])}
        if len(lookup) != len(pool["ids"]):
            raise ValueError("Duplicate generated candidate IDs")
        if set(lookup) != {r["id"] for r in status if r["status"] == "generated"}:
            raise ValueError("Status receipt and saved candidate archive disagree")
        for row in status:
            if row["id"] in lookup:
                i = lookup[row["id"]]
                if str(pool["request_ids"][i]) != row["request_id"] or int(pool["labels"][i]) != row["label"]:
                    raise ValueError("Candidate request/class differs from status receipt")
        models, teacher_receipts = {}, {}
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        for view in ("spatial", "speed"):
            models[view], teacher_receipts[view] = load_reader(args.teachers/f"{view}-42", digest(args.cache/"complete.json"), view)
        scores = np.full((len(lookup), 2, len(CLASSES)), np.nan, dtype=np.float64)
        encoding = []
        for base in range(0, len(pool["ids"]), 32):
            spatial, speed, si, vi = [], [], [], []
            batch_rows = []
            for i in range(base, min(base+32, len(pool["ids"]))):
                txy = pool["txy"][pool["offsets"][i]:pool["offsets"][i+1]]
                if (not np.isfinite(txy).all() or len(txy) < 2 or np.any(np.diff(txy[:, 0]) <= 0)):
                    raise ValueError("Completed candidate trajectory invalid")
                row = {"id": str(pool["ids"][i]), "request_id": str(pool["request_ids"][i]), "label": int(pool["labels"][i]),
                       "samples": len(txy), "fractional_final_interval": not np.isclose(np.diff(txy[:, 0])[-1], .25)}
                try:
                    sequence, clipped = spatial_sequence(txy[:, 1:], metadata)
                    rgb, channel_clipped = encode_frozen(sequence[None], np.asarray(metadata["channel_min"]), np.asarray(metadata["channel_max"]))
                    spatial.append(rgb[0]); si.append(i)
                    row.update(spatial_status="encoded", spatial_coordinate_clipped_points=clipped,
                               spatial_channel_clipped_values=int(channel_clipped),
                               spatial_image_sha256=hashlib.sha256(rgb[0].tobytes()).hexdigest())
                except ValueError as error:
                    row.update(spatial_status="encoding_failed", spatial_error=str(error))
                try:
                    profile, _ = speed_series(txy[:, 0], txy[:, 1:])
                    speed.append(profile); vi.append(i)
                    row.update(speed_status="encoded", speed_values_above_35=int(np.sum(profile > 35.)), speed_values=len(profile))
                except ValueError as error:
                    row.update(speed_status="encoding_failed", speed_error=str(error))
                batch_rows.append(row)
            with torch.no_grad():
                for axis, view, indices, values in ((0, "spatial", si, spatial), (1, "speed", vi, speed)):
                    if indices:
                        logits = models[view](image_inputs(view, np.stack(values)))
                        if not torch.isfinite(logits).all():
                            raise ValueError("Nonfinite frozen-teacher logits")
                        scores[indices, axis] = logits.softmax(1).cpu().numpy()
            encoding.extend(batch_rows)
            print(json.dumps({"scored_candidates": min(base+32, len(pool["ids"])), "total": len(pool["ids"]),
                              "seconds": time.monotonic()-started}), flush=True)
        save(output/"encoding-status.json", {"rows": encoding})
        np.savez_compressed(output/"scores.npz", ids=pool["ids"], request_ids=pool["request_ids"], labels=pool["labels"], probabilities=scores)
        selections, selected_indices = [], {method: [] for method in METHODS}
        source_by_id = {row["id"]: row for row in encoding}
        for request_id, group in sorted(groups.items()):
            order = permutation(request_id, K)
            label = group[0]["label"]
            s = np.asarray([scores[lookup[r["id"]], 0, label] if r["id"] in lookup else np.nan for r in group])
            v = np.asarray([scores[lookup[r["id"]], 1, label] if r["id"] in lookup else np.nan for r in group])
            all_valid = bool(np.isfinite(s).all() and np.isfinite(v).all())
            choices = choose(s, v, order, all_valid)
            unique_images = len({source_by_id[r["id"]].get("spatial_image_sha256") for r in group
                                 if r["id"] in source_by_id and source_by_id[r["id"]].get("spatial_image_sha256")})
            for method, index in choices.items():
                chosen = group[index]
                available = chosen["id"] in lookup
                row = {"method": method, "request_id": request_id, "label": label,
                       "candidate": int(index), "id": chosen["id"], "shared_uniform_fallback": not all_valid,
                       "status": "selected" if available else "uniform_selected_failed_slot",
                       "spatial_unique_images_in_original_pool": unique_images,
                       "spatial_target_probability_auxiliary": float(s[index]) if np.isfinite(s[index]) else None,
                       "speed_target_probability_auxiliary": float(v[index]) if np.isfinite(v[index]) else None}
                selections.append(row)
                if available:
                    selected_indices[method].append(lookup[chosen["id"]])
        save(output/"selection-status.json", {"rows": selections})
        selected_request_sets = [{str(pool["request_ids"][i]) for i in selected_indices[m]} for m in METHODS]
        if any(ids != selected_request_sets[0] for ids in selected_request_sets):
            raise ValueError("Terminal strategies have different successful request coverage")
        for method in METHODS:
            folder = output/method
            folder.mkdir()
            indices = selected_indices[method]
            if not indices:
                raise ValueError("No selected candidate in entire fixed pool")
            tracks = [pool["txy"][pool["offsets"][i]:pool["offsets"][i+1]] for i in indices]
            np.savez_compressed(folder/"samples.npz", ids=pool["ids"][indices], request_ids=pool["request_ids"][indices],
                                labels=pool["labels"][indices], offsets=np.r_[0, np.cumsum([len(t) for t in tracks])], txy=np.concatenate(tracks))
            rows = [r for r in selections if r["method"] == method]
            save(folder/"selection-status.json", {"rows": rows})
            receipt = {"complete": True, "phase": "terminal_gaf_selection", "selection_method": method,
                       "family": generation["family"], "arm": generation["arm"], "seed": generation["seed"], "classes": CLASSES,
                       "request_sha256": generation["request_sha256"], "reference_data_opened": False,
                       "requested_conditions": generation["requested_conditions"], "map_ready_conditions": len(groups),
                       "requested_candidates_per_condition": K, "generated_candidates": len(indices),
                       "source_pool_candidates": len(pool["ids"]), "status_counts": dict(Counter(r["status"] for r in rows)),
                       "shared_uniform_fallback_requests": sum(r["shared_uniform_fallback"] for r in rows),
                       "source_pool_receipt_sha256": digest(args.generated/"complete.json"),
                       "selection_receipt_sha256": digest(output/"selection-status.json"),
                       "artifacts": {n: {"sha256": digest(folder/n), "bytes": (folder/n).stat().st_size}
                                     for n in ("samples.npz", "selection-status.json")}}
            save(folder/"complete.json", receipt)
        result = {"complete": True, "phase": "terminal_gaf_selection", "methods": list(METHODS), "classes": CLASSES,
                  "family": generation["family"], "arm": generation["arm"], "seed": generation["seed"],
                  "requested_conditions": generation["requested_conditions"], "pools": len(groups), "K": K,
                  "source_candidates": len(pool["ids"]), "source_candidate_status_counts": generation["status_counts"],
                  "shared_uniform_fallback_pools": sum(r["shared_uniform_fallback"] for r in selections if r["method"] == "Uniform"),
                  "selected_counts": {m: len(selected_indices[m]) for m in METHODS},
                  "space_encoding_failures": sum(r["spatial_status"] != "encoded" for r in encoding),
                  "speed_encoding_failures": sum(r["speed_status"] != "encoded" for r in encoding),
                  "spatial_unique_image_count_by_pool": dict(Counter(r["spatial_unique_images_in_original_pool"] for r in selections if r["method"] == "Uniform")),
                  "selection_changes_from_uniform": {m: sum(a["id"] != b["id"] for a, b in zip(
                      [r for r in selections if r["method"] == m], [r for r in selections if r["method"] == "Uniform"])) for m in METHODS},
                  "policy": {"uniform_seed": UNIFORM_SEED, "uniform": "first slot of request-ID-keyed fixed random permutation over all requested K slots",
                             "scores": "S=log(target_p_spatial),V=log(target_p_speed),SV=.5*S+.5*V", "probability_floor": PROBABILITY_FLOOR,
                             "ties": "exact score ties use same permutation as Uniform; no tolerance",
                             "encoding_failure": "any missing/failed candidate/view makes all four selectors use same Uniform slot; no replacement",
                             "teacher_score_role": "auxiliary; independently evaluated physical quality determines synthesis benefit"},
                  "teacher_candidate_evaluations": {v: sum(r[v+"_status"] == "encoded" for r in encoding) for v in ("spatial", "speed")},
                  "seconds": time.monotonic()-started, "reference_data_opened": False,
                  "sources": {"generation_receipt_sha256": digest(args.generated/"complete.json"),
                              "cache_receipt_sha256": digest(args.cache/"complete.json"),
                              "teacher_receipts": {v: digest(args.teachers/f"{v}-42"/"complete.json") for v in ("spatial", "speed")},
                              "code_sha256": code_hashes()},
                  "artifacts": {n: {"sha256": digest(output/n), "bytes": (output/n).stat().st_size}
                                for n in ("scores.npz", "encoding-status.json", "selection-status.json")},
                  "limits": ["This is terminal selection, not in-loop guidance; no denoising or extra candidate generation occurs.",
                             "Same-route candidates may have identical spatial images; exact ties keep Uniform selection.",
                             "Both selection teachers are self-trained; their own scores cannot establish independent quality gain.",
                             "One retained candidate per request has no within-request diversity estimate; retain all-pool baseline and across-request dispersion."]}
        save(output/"complete.json", result)
        return result
    except Exception as error:
        save(output/"failed.json", {"complete": False, "error": repr(error), "seconds": time.monotonic()-started})
        raise


def evaluate(args):
    from g33_evaluate import evaluate as physical_evaluate, extract_record, stats
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    try:
        selection = completed(args.selected, ["scores.npz", "encoding-status.json", "selection-status.json"])
        if selection["reference_data_opened"] is not False:
            raise ValueError("Selection used reference data")
        reports, dispersion = {}, {}
        for method in METHODS:
            reports[method] = physical_evaluate(args.selected/method, args.reference, args.maps, args.training, output/method)
            with np.load(args.selected/method/"samples.npz", allow_pickle=False) as archive:
                rows = [extract_record(archive["txy"][archive["offsets"][i]:archive["offsets"][i+1]]) for i in range(len(archive["ids"]))]
                labels = archive["labels"].copy()
            dispersion[method] = {}
            for c, name in enumerate(CLASSES):
                group = [r for r, label in zip(rows, labels) if label == c]
                dispersion[method][name] = {"requests": len(group),
                    "mean_relative_progress_pointwise_sd": float(np.std(np.stack([r["profile"] for r in group]), axis=0).mean()) if group else None,
                    "duration_s": stats(r["physical"]["duration_s"] for r in group)}
        baseline = reports["Uniform"]["paired_macro_w1"]
        delta = {m: {key: reports[m]["paired_macro_w1"][key]-baseline[key]
                     if baseline[key] is not None and reports[m]["paired_macro_w1"][key] is not None else None for key in baseline} for m in METHODS}
        result = {"complete": True, "scope": "paired_terminal_selection_single_episode_descriptive_quality_comparison",
                  "methods": list(METHODS), "selection_receipt_sha256": digest(args.selected/"complete.json"),
                  "macro_w1": {m: reports[m]["paired_macro_w1"] for m in METHODS},
                  "macro_w1_difference_from_uniform_lower_is_better": delta,
                  "coverage": {m: reports[m]["coverage"] for m in METHODS},
                  "across_request_same_class_dispersion": dispersion,
                  "source_hashes": code_hashes(),
                  "limits": ["Selected rows share the original candidate pool; report full-pool generation result alongside this selection result.",
                             "Across-request dispersion includes different route conditions and is not conditional diversity.",
                             "No new noise samples, teacher fitting or parameter selection occur here.",
                             "No confidence intervals from one simulation episode; two generator seeds remain separate.",
                             "The selection teacher scores are not used as physical-quality truth."]}
        save(output/"complete.json", result)
        return result
    except Exception as error:
        save(output/"failed.json", {"complete": False, "error": repr(error)})
        raise


def self_test():
    order = permutation("synthetic-request", 8)
    assert np.array_equal(order, permutation("synthetic-request", 8))
    equal = np.full(8, .2)
    assert len(set(choose(equal, equal, order).values())) == 1
    x, y = np.full(8, .1), np.full(8, .1)
    x[0], y[1] = .9, .9
    choices = choose(x, y, order)
    assert choices["S"] == 0 and choices["V"] == 1
    assert choices["SV"] == next(int(i) for i in order if i in (0, 1))
    x[2] = np.nan
    assert set(choose(x, y, order, False).values()) == {int(order[0])}
    try:
        choose(x, y, order, True)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid complete-pool scores accepted")
    from g29_encode import sequence_job
    from cnr.original_gaf import encode_frozen
    from g23_modes import speed_series
    metadata = {"xy_min": [-20., -20.], "xy_max": [20., 20.], "hilbert_eps": .0009,
                "channel_min": [-1., -1., 0.], "channel_max": [1., 1., 1.]}
    actual_time = np.r_[np.arange(0., 8.25, .25), 8.125]
    for xy in (np.c_[10*np.cos(actual_time), 10*np.sin(actual_time)],
               np.full((len(actual_time), 2), 5.)):
        sequence, clipped = spatial_sequence(xy, metadata)
        # A regular dummy clock only tests the inherited spatial formula;
        # actual generated times remain untouched in the scoring path.
        inherited = sequence_job(("synthetic", 0, np.c_[np.arange(len(xy))*.25, xy], metadata))
        assert np.array_equal(sequence, inherited["sequence"])
        assert clipped == inherited["audit"]["clipped_points"]
        a, _ = encode_frozen(sequence[None], np.array(metadata["channel_min"]), np.array(metadata["channel_max"]))
        b, _ = encode_frozen(inherited["sequence"][None], np.array(metadata["channel_min"]), np.array(metadata["channel_max"]))
        assert np.array_equal(a, b)
        profile, _ = speed_series(actual_time, xy)
        assert profile.shape == (128,) and np.isfinite(profile).all()
    print("g33_terminal_selection_checks_passed:uniform,ties,fallback,invalid_scores,original_spatial_bytes,constant,fractional_clock", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("score", "evaluate", "checks"))
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--teachers", type=Path, default=Path("g33-data/teachers"))
    parser.add_argument("--cache", type=Path, default=Path("g33-data/gaf-cache"))
    parser.add_argument("--reference", type=Path, default=Path("g33-data/reference-extracted"))
    parser.add_argument("--maps", type=Path, default=Path("g33-data/maps"))
    parser.add_argument("--training", type=Path, default=Path("g33-data/direct-legs"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action != "checks" and args.output is None:
        parser.error("Fresh output directory required")
    if args.action == "score" and args.generated is None:
        parser.error("score requires completed --generated pool")
    if args.action == "evaluate" and args.selected is None:
        parser.error("evaluate requires completed --selected folder")
    result = self_test() if args.action == "checks" else score(args) if args.action == "score" else evaluate(args)
    if result is not None:
        print(json.dumps({"complete": result["complete"], "scope": result.get("scope", result.get("phase"))}), flush=True)
