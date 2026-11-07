"""G36: fixed-duration sampling for completed old/new native diffusion models.

New public requests are frozen before independent SUMO reference simulation.
No fitting, reference access or teacher scoring occurs here. The inherited
Uniform slot supplies the training-prior duration RNG key, shared across K8.
Old and expanded native models use identical sampling mathematics. No old
candidate pool is needed or opened. Frozen inherited modules are unchanged.
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
import torch

from g33_evaluate import artifact, completed, curve_xy, digest, json_hash, save
from g33_feedback import permutation, UNIFORM_SEED
from g33_model import (CLASSES, MotionDenoiser, clock, decode, pack_motion,
                       predict, sample, sample_duration)


K = 8
DDIM_STEPS = 100
ARM = "native_0_25s"
EXPERIMENT_PHASE = "G36_new_SUMO_fixed_duration_replication"
DURATION_SOURCE = "G33_train_prior_draw_keyed_by_predetermined_Uniform_candidate_ID"
FORBIDDEN = {"duration", "true_duration", "progress", "target_xy", "reference", "score"}


def candidate_key(candidate_id, seed):
    """Exactly the G33 sample RNG derivation, including model seed."""
    return int.from_bytes(hashlib.sha256(f"G33:{candidate_id}:{seed}".encode()).digest()[:8], "little")


def validate_request(request):
    # Fail closed: this generator accepts no target, score or supplied duration.
    if set(request) != {"id", "label", "length", "path"}:
        raise ValueError("Only declared id/label/length/path are accepted")
    path = np.asarray(request["path"])
    if (int(request["label"]) not in range(len(CLASSES)) or path.shape != (129, 2)
            or not np.isfinite(path).all() or not np.isfinite(request["length"]) or request["length"] <= 0):
        raise ValueError("Invalid public route request")


def duration_plan(request, prior, seed):
    """Select a prior draw by an outcome-independent slot; never by score."""
    validate_request(request)
    slot = int(permutation(request["id"], K)[0])
    name = f"{request['id']}/candidate-{slot:02d}"
    key = candidate_key(name, seed)
    duration, bounded = sample_duration(prior, [request["length"]], [request["label"]],
                                        np.random.default_rng(key))
    return {"request_id": request["id"], "label": int(request["label"]),
            "uniform_slot": slot, "duration_source_candidate_id": name,
            "duration_prior_rng_key": key, "model_seed": seed,
            "duration": float(duration[0]), "duration_bounded": bool(bounded[0])}


@torch.no_grad()
def sample_fixed(model, request, stats, plan, seed, device="cuda", ddim_steps=DDIM_STEPS):
    """Unchanged G33 DDIM equations; replace only the per-candidate prior draw.

    The supplied plan is produced by duration_plan from the training-side prior,
    not from a reference trajectory. Validate its request/class/seed association.
    """
    validate_request(request)
    if (seed != plan["model_seed"] or request["label"] != plan["label"]
            or request["id"].rsplit("/candidate-", 1)[0] != plan["request_id"]):
        raise ValueError("Duration plan belongs to another request/class/model seed")
    duration = np.asarray([plan["duration"]], dtype=np.float64)
    times = clock(float(duration[0]), ARM)
    n = len(times)-1
    batch = pack_motion([np.zeros(n, np.float32)], [times], [request["label"]],
                        [request["path"]], [request["length"]], duration, stats, device)
    key = candidate_key(request["id"], seed)
    generator = torch.Generator(device=device).manual_seed(key)
    x = torch.randn(batch["x"].shape, generator=generator, device=device)*batch["mask"]
    schedule = torch.cumprod(1-torch.linspace(.0001, .02, 1000, device=device), 0)
    timesteps = torch.linspace(999, 0, ddim_steps).long().tolist()
    for j, t in enumerate(timesteps):
        epsilon = predict(model, batch, x, torch.full((1,), t, dtype=torch.long, device=device))
        clean = (x-(1-schedule[t]).sqrt()*epsilon)/schedule[t].sqrt()
        previous = schedule[timesteps[j+1]] if j+1 < len(timesteps) else 1.
        x = (previous**.5*clean+(1-previous)**.5*epsilon)*batch["mask"]
    profile, flags = decode(x[0, 0, :n].cpu().numpy(), times, stats["motion_mean"], stats["motion_sd"])
    return {"id": request["id"], "duration": float(duration[0]), "time": times, "progress": profile,
            "duration_bounded": plan["duration_bounded"], "motion_noise_rng_key": key, **flags}


def source_hashes():
    return {"g36_generate.py": digest(__file__),
            "g33_model.py": digest(inspect.getsourcefile(MotionDenoiser)),
            "g33_evaluate.py": digest(inspect.getsourcefile(curve_xy)),
            "g33_feedback.py": digest(inspect.getsourcefile(permutation))}


def generate(maps, model_folder, output, seed, device):
    if seed not in (42, 43) or UNIFORM_SEED != 33073:
        raise ValueError("G34 fixes model seeds42/43 and Uniform seed33073")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        map_receipt = completed(maps, ["maps.npz", "request-status.json"])
        if (map_receipt.get("phase") != "prospective_maps" or map_receipt.get("classes") != CLASSES
                or map_receipt.get("denominator") != 100 or map_receipt.get("reference_data_opened") is not False):
            raise ValueError("Original prospective100-request maps required")
        model_receipt = json.loads((model_folder/"complete.json").read_text())
        if (not model_receipt.get("complete") or model_receipt.get("arm") != ARM or model_receipt.get("seed") != seed
                or digest(model_folder/"model.pt") != model_receipt["model_sha256"]
                or digest(model_folder/"stats.json") != model_receipt["stats_sha256"]
                or digest(model_folder/"config.json") != model_receipt["config_sha256"]):
            raise ValueError("Unchanged completed G33 native model required")
        config = json.loads((model_folder/"config.json").read_text())
        if config["model_source_sha256"] != digest(inspect.getsourcefile(MotionDenoiser)):
            raise ValueError("G33 model implementation changed")
        state = torch.load(model_folder/"model.pt", weights_only=False, map_location="cpu")
        stats = json.loads((model_folder/"stats.json").read_text())
        if state["classes"] != CLASSES or state["seed"] != seed or state["arm"] != ARM or state["stats"] != stats:
            raise ValueError("Checkpoint class/seed/arm/statistics differ")
        model = MotionDenoiser(base=state["width"]).to(device).eval()
        model.load_state_dict(state["state_dict"])
        forward_calls = {"n": 0}
        def count_forward(module, inputs, result):
            forward_calls["n"] += 1
        model.register_forward_hook(count_forward)
        with np.load(maps/"maps.npz", allow_pickle=False) as archive:
            geometry = {k: archive[k].copy() for k in archive.files}
        if len(geometry["ids"]) != map_receipt["ready"] or len(set(geometry["ids"].astype(str))) != map_receipt["ready"]:
            raise ValueError("Map-ready identity/support mismatch")
        # Freeze every duration before producing any motion output.
        plans = [duration_plan({"id": str(ident), "label": int(geometry["labels"][i]),
                                "length": float(geometry["length"][i]), "path": geometry["paths"][i]},
                               stats["duration_prior"], seed) for i, ident in enumerate(geometry["ids"])]
        save(output/"duration-plan.json", {"sampled_duration_source": DURATION_SOURCE,
                                          "uniform_seed": UNIFORM_SEED, "rows": plans})
        rows, candidates, trajectories, offsets = [], [], [], [0]
        with gzip.open(output/"generated.csv.gz", "wt", newline="") as stream, (output/"generation.jsonl").open("w", buffering=1) as log:
            writer = csv.writer(stream); writer.writerow(["time", "id", "x", "y", "type"])
            for i, plan in enumerate(plans):
                ident = plan["request_id"]
                vertices = geometry["vertices"][geometry["offsets"][i]:geometry["offsets"][i+1]]
                for candidate in range(K):
                    name = f"{ident}/candidate-{candidate:02d}"
                    request = {"id": name, "label": int(geometry["labels"][i]),
                               "length": float(geometry["length"][i]), "path": geometry["paths"][i]}
                    row = {"id": name, "request_id": ident, "candidate": candidate, "label": request["label"],
                           "planned_duration": plan["duration"], "duration_source_candidate_id": plan["duration_source_candidate_id"],
                           "uniform_slot": plan["uniform_slot"], "motion_noise_rng_key": candidate_key(name, seed)}
                    try:
                        draw = sample_fixed(model, request, stats, plan, seed, device=device)
                        times = clock(draw["duration"], ARM)
                        progress = np.interp(times, draw["time"], draw["progress"])
                        xy = curve_xy(vertices, progress)
                        if (not np.isfinite(xy).all() or np.any(np.diff(progress) < 0)
                                or not np.allclose(xy[[0, -1]], vertices[[0, -1]], atol=1e-8, rtol=0)
                                or draw["duration"] != plan["duration"]):
                            raise ValueError("Invalid decoded trajectory, endpoint or shared duration")
                        txy = np.c_[times, xy]
                        trajectories.append(txy); offsets.append(offsets[-1]+len(txy)); candidates.append(row)
                        row.update(status="generated", duration=draw["duration"], samples=len(times),
                                   duration_bounded=draw["duration_bounded"], negative_bins=draw["negative_bins"],
                                   upper_clipped_bins=draw["upper_clipped_bins"], constant_speed_fallback=draw["constant_speed_fallback"])
                        for stamp, (x, y) in zip(times, xy):
                            writer.writerow([f"{stamp:.9f}", name, f"{x:.9f}", f"{y:.9f}", CLASSES[request["label"]]])
                    except ValueError as error:
                        row.update(status="candidate_failed", error=str(error))
                    rows.append(row)
                log.write(json.dumps({"requests_processed": i+1, "candidates_generated": len(trajectories),
                                      "elapsed_s": time.monotonic()-started})+"\n")
                print(json.dumps({"experiment": "G34", "seed": seed, "request": i+1, "total": len(plans)}), flush=True)
        if not trajectories:
            raise ValueError("No candidates generated")
        np.savez_compressed(output/"samples.npz", ids=np.asarray([r["id"] for r in candidates]),
                            request_ids=np.asarray([r["request_id"] for r in candidates]),
                            labels=np.asarray([r["label"] for r in candidates], np.int64),
                            offsets=np.asarray(offsets, np.int64), txy=np.concatenate(trajectories))
        save(output/"candidate-status.json", {"rows": rows})
        result = {"complete": True, "phase": "prospective_generation", "experiment_phase": EXPERIMENT_PHASE,
                  "sampled_duration_source": DURATION_SOURCE, "duration_fixed_within_request": True,
                  "same_duration_across_model_seeds": False, "duration_source_uniform_seed": UNIFORM_SEED,
                  "family": "diffusion", "arm": ARM, "seed": seed, "classes": CLASSES,
                  "requested_conditions": map_receipt["denominator"], "map_ready_conditions": map_receipt["ready"],
                  "requested_candidates_per_condition": K, "candidate_denominator_all_requests": map_receipt["denominator"]*K,
                  "attempted_candidates": len(rows), "generated_candidates": len(trajectories),
                  "status_counts": dict(Counter(r["status"] for r in rows)), "ddim_steps": DDIM_STEPS,
                  "network_forward_calls": forward_calls["n"], "teacher_calls": 0,
                  "elapsed_s": time.monotonic()-started, "reference_data_opened": False, "teacher_scores_opened": False,
                  "request_sha256": map_receipt["request_sha256"], "map_receipt_sha256": digest(maps/"complete.json"),
                  "training_model_receipt_sha256": digest(model_folder/"complete.json"),
                  
                  "duration_prior_sha256": json_hash(stats["duration_prior"]), "script_sha256": digest(__file__),
                  "model_code_sha256": digest(inspect.getsourcefile(MotionDenoiser)), "source_code_sha256": source_hashes(),
                  "artifacts": {n: artifact(output/n) for n in ("samples.npz", "generated.csv.gz", "candidate-status.json", "duration-plan.json")},
                  "limits": ["New independently seeded SUMO references; shared public source/map/templates, not unseen-city validation.",
                             "One duration is sampled from the unchanged train prior per request and model seed; no reference duration is supplied.",
                             "G33 Uniform candidate ID, duration and noise key are preserved; other candidates retain their keys but change sequence length when duration changes.",
                             "Same number of candidates and DDIM steps does not imply identical runtime or tensor-element compute.",
                             "Fixed durations isolate ranking within a duration-conditioned pool; route geometry and endpoint legality do not imply collision or signal safety.",
                             "No feedback teacher participates in generation; use unchanged G33 terminal scoring and independent physical evaluation separately."]}
        save(output/"complete.json", result)
        return result
    except Exception as error:
        save(output/"failed.json", {"complete": False, "experiment_phase": EXPERIMENT_PHASE,
                                   "error": repr(error), "elapsed_s": time.monotonic()-started})
        raise


def self_test():
    """Synthetic in-memory contracts; never open datasets, models or references."""
    from unittest.mock import patch
    torch.set_num_threads(1)
    # @torch.no_grad wraps sample in a torch context-manager source file.
    # Hash the model class's file, and cross-check the unwrapped sampler.
    model_source = Path(inspect.getsourcefile(MotionDenoiser)).resolve()
    assert model_source.name == "g33_model.py"
    assert model_source == Path(inspect.getsourcefile(inspect.unwrap(sample))).resolve()
    class NoiseReader(torch.nn.Module):
        def forward(self, x, timestep, labels, context, route, clock_features, mask):
            return (.1*x+.01*clock_features[:, :1])*mask
    prior = {"minimum_duration_s": .25, "maximum_mean_speed_mps": 55.,
             "classes": [{"intercept": float(np.log(1.25)), "slope": .1, "center_log_length": 0.,
                          "residuals": [-.15, -.03, .11, .23]} for _ in CLASSES]}
    stats = {"duration_prior": prior, "motion_mean": .65, "motion_sd": .2,
             "start_mean": [0., 0.], "start_sd": [1., 1.]}
    request = {"id": "synthetic-request", "label": 1, "length": 10.,
               "path": np.c_[np.linspace(0., 10., 129), np.zeros(129)]}
    model = NoiseReader()
    checks = []
    # Enforce absence of all filesystem/trajectory/score reads during planning
    # and sampling. Even a future accidental Path.open/read_text fails here.
    with patch("builtins.open", side_effect=AssertionError("No input files in sampling")), \
            patch("pathlib.Path.open", side_effect=AssertionError("No path input files")), \
            patch("numpy.load", side_effect=AssertionError("No NPZ input in sampling")):
        keys = set()
        for seed in (42, 43):
            plan = duration_plan(request, prior, seed)
            assert plan == duration_plan(request, prior, seed)
            assert plan["uniform_slot"] == int(permutation(request["id"], K)[0])
            expected, bounded = sample_duration(prior, [request["length"]], [request["label"]],
                                               np.random.default_rng(candidate_key(plan["duration_source_candidate_id"], seed)))
            assert plan["duration"] == float(expected[0]) and plan["duration_bounded"] == bool(bounded[0])
            for candidate in range(K):
                one = {**request, "id": f"{request['id']}/candidate-{candidate:02d}"}
                draw = sample_fixed(model, one, stats, plan, seed, device="cpu", ddim_steps=4)
                assert draw["duration"] == plan["duration"] and draw["time"][-1] == plan["duration"]
                assert draw["motion_noise_rng_key"] == candidate_key(one["id"], seed)
                keys.add(draw["motion_noise_rng_key"])
                if candidate == plan["uniform_slot"]:
                    old = sample(model, [one], stats, ARM, seed, device="cpu", ddim_steps=4)[0]
                    for field in ("duration", "duration_bounded", "negative_bins", "upper_clipped_bins", "constant_speed_fallback"):
                        assert draw[field] == old[field], field
                    assert np.array_equal(draw["time"], old["time"])
                    assert np.array_equal(draw["progress"], old["progress"])
            bad = dict(plan, model_seed=seed+1)
            try:
                sample_fixed(model, one, stats, bad, seed, device="cpu", ddim_steps=4)
            except ValueError:
                pass
            else:
                raise AssertionError("Mismatched model seed accepted")
        assert len(keys) == 2*K
        for field in FORBIDDEN:
            try:
                duration_plan({**request, field: 1.}, prior, 42)
            except ValueError:
                pass
            else:
                raise AssertionError("Target/score/supplied duration accepted")
    checks.extend(["shared_T_all_eight_slots", "deterministic_Uniform_slot", "two_seed_noise_key_pairing",
                   "G33_prior_draw_exact", "G33_Uniform_motion_bitwise_equal_synthetic_CPU",
                   "no_input_file_reads_in_planning_sampling", "reference_score_and_supplied_duration_rejected",
                   "model_class_and_unwrapped_sampler_source_agree"])
    result = {"complete": True, "scope": "synthetic in-memory tests only; no real generation", "checks": checks}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "checks"))
    parser.add_argument("--maps", type=Path, default=Path("g33-data/maps"))
    parser.add_argument("--model", type=Path)
    
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    if args.action == "checks":
        self_test()
    else:
        if args.output is None or args.model is None:
            parser.error("generate requires --model, and --output")
        generate(args.maps, args.model, args.output, args.seed, args.device)
