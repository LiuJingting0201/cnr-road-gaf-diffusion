"""Audit TRAIN/MONITOR encoding parity and the complete-trip representation floor.

Runs no evaluation cohort, sampling search, or model fitting. Writes only small JSON.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from g22_diffusion import decode
from g23_modes import speed_series


def summarize(values):
    x = np.asarray(values, dtype=float)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def spatial_audit(examples, checkpoint):
    from g12_fit import features
    from g19_gaf import images
    from torchvision.models import resnet18

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 8)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    model.to(device).eval()
    records = []
    for start in range(0, len(examples), 8):
        batch = examples[start:start + 8]
        variants = [track for example in batch for track in example[1:]]
        rgb, valid = images(variants)
        with torch.no_grad():
            probabilities = model(features(torch.from_numpy(rgb).to(device))).softmax(1).cpu().numpy()
        for i, example in enumerate(batch):
            c = example[0]
            raw, padded, oracle = probabilities[3 * i:3 * i + 3]
            a, b, _ = rgb[3 * i:3 * i + 3].astype(float)
            records.append(dict(
                label=c,
                valid=bool(valid[3 * i:3 * i + 3].all()),
                raw_correct=float(raw.argmax() == c),
                padded_correct=float(padded.argmax() == c),
                oracle_correct=float(oracle.argmax() == c),
                target_probability_padded_minus_raw=float(padded[c] - raw[c]),
                target_probability_oracle_minus_padded=float(oracle[c] - padded[c]),
                image_absolute_difference=float(np.abs(a - b).mean() / 255),
                image_changed=float(np.any(a != b)),
            ))
    result = {}
    for c in range(8):
        rows = [r for r in records if r["label"] == c]
        result[str(c)] = {
            "n": len(rows),
            **{k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "label"},
        }
    return result


def run(data, checkpoint, output):
    torch.set_num_threads(2)
    result = {
        "scope": "training and monitor only; oracle reconstruction uses their observed targets",
        "spatial_selection": "first four IDs per class per monitor episode; declared before results",
        "splits": {},
    }
    examples = []
    for split in ("train", "monitor"):
        groups = {c: [] for c in range(8)}
        files = sorted((data / split).glob("*/quarter.npz"))
        if not files:
            raise ValueError(f"No quarter-second observations in {data / split}")
        for file in files:
            with np.load(file) as native, np.load(file.parent / "trips.npz") as trips:
                q = {key: native[key] for key in native.files}
                d = {key: trips[key] for key in trips.files}
                if not np.array_equal(q["ids"], d["ids"]):
                    raise ValueError(f"Mismatched row IDs in {file}")
                selected = np.zeros(8, dtype=int)
                for i, c in enumerate(d["labels"]):
                    c = int(c)
                    a, b = q["offset"][i:i + 2]
                    raw = q["txy"][a:b].astype(float)
                    raw_t, raw_xy = raw[:, 0], raw[:, 1:]
                    path = d["paths"][i]
                    duration = float(d["duration"][i])
                    times, xy = decode(path, duration, d["progress"][i])
                    profile, stats = speed_series(raw_t, raw_xy)
                    recovered, recovered_stats = speed_series(times, xy)
                    begin, end = d["offset"][i:i + 2]
                    one_second = np.vstack([path[0], d["xy"][begin:end], path[-1]])
                    raw_arc = np.linalg.norm(np.diff(raw_xy, axis=0), axis=1).sum()
                    one_second_arc = np.linalg.norm(np.diff(one_second, axis=0), axis=1).sum()
                    length = float(d["length"][i])
                    grid = np.linspace(0, np.floor(duration) - 1, 128)
                    left = np.floor(grid)
                    arc_samples = [length * np.interp(
                        (left + k) / duration, np.linspace(0, 1, 257), d["progress"][i]
                    ) for k in (0, 1, 2)]
                    aa, bb, cc = arc_samples
                    surrogate = bb - aa + (grid - left) * (cc - 2 * bb + aa)
                    groups[c].append(dict(
                        raw_duration_shortfall_s=duration - (raw_t[-1] - raw_t[0]),
                        raw_arc_minus_route_fraction=(raw_arc - length) / length,
                        target_one_second_arc_minus_route_fraction=(one_second_arc - length) / length,
                        start_position_gap_m=float(np.linalg.norm(path[0] - raw_xy[0])),
                        end_position_gap_m=float(np.linalg.norm(path[-1] - raw_xy[-1])),
                        oracle_profile_mae_mps=float(np.abs(recovered - profile).mean()),
                        training_auxiliary_target_mae_mps=float(np.abs(surrogate - profile).mean()),
                        surrogate_vs_emitted_profile_mae_mps=float(np.abs(surrogate - recovered).mean()),
                        oracle_mean_speed_delta_mps=float(recovered_stats[0] - stats[0]),
                        oracle_four_second_change_delta_mps=float(recovered_stats[1] - stats[1]),
                    ))
                    if split == "monitor" and selected[c] < 4:
                        examples.append((c, raw_xy, np.vstack([path[0], raw_xy, path[-1]]), xy))
                        selected[c] += 1
        result["splits"][split] = {
            str(c): {key: summarize([row[key] for row in rows]) for key in rows[0]}
            for c, rows in groups.items()
        }
    result["monitor_spatial"] = spatial_audit(examples, checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(output)
    print(json.dumps({"output": str(output), "spatial_requests": len(examples)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("g22-data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("../g22-job/g22-data/cnn-native-gaf/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("g28-data/roundtrip.json"))
    args = parser.parse_args()
    run(args.data, args.checkpoint, args.output)
