"""Add a grid-coordinate control on the already frozen primary cohort."""
import argparse
import json
from pathlib import Path

import numpy as np

from g29_encode import read_raw, sampled
from cnr.representations import hilbert_numpy


def add_control(args):
    completion = args.encoded / "complete.json"
    if not completion.exists():
        raise RuntimeError("The main encoded cohort must be complete before adding the grid control")
    metadata = json.loads(completion.read_text())["frozen_metadata"]
    low, high = np.asarray(metadata["xy_min"]), np.asarray(metadata["xy_max"])
    # The recursive Hilbert transform reverses boundary tie sides in reflected
    # quadrants. Recover centres from its own integer cell index, not global floor.
    grid_axis = (np.arange(64) + .5) / 64
    x, y = np.meshgrid(grid_axis, grid_axis, indexing="ij")
    grid_centres = np.column_stack((x.ravel(), y.ravel()))
    centre_indices = np.rint(hilbert_numpy(grid_centres, metadata["hilbert_eps"]) * 4096).astype(np.int64)
    if not np.array_equal(np.sort(centre_indices), np.arange(4096)):
        raise ValueError("Inherited Hilbert transform is not bijective on cell centres")
    centre_by_index = np.empty_like(grid_centres)
    centre_by_index[centre_indices] = grid_centres
    reports = {}
    for domain in ("cnr", "sumo"):
        for split in ("train", "val"):
            raw = read_raw(args.raw / f"{domain}-{split}.npz")
            index = {str(ident): i for i, ident in enumerate(raw["ids"])}
            directory = args.encoded / domain / "full" / split
            destination = directory / "features.npz"
            with np.load(destination) as existing:
                features = {key: existing[key] for key in existing.files}
            values = []
            mismatch_points = total_points = mismatch_tracks = 0
            for ident, label in zip(features["ids"], features["labels"]):
                i = index[str(ident)]
                if raw["labels"][i] != label:
                    raise ValueError(f"Label mismatch for {ident}")
                xy = raw["txy"][raw["offset"][i]:raw["offset"][i+1], 1:]
                unit = ((xy-low)/(high-low)).clip(0, 1)
                native_indices = np.rint(hilbert_numpy(unit, metadata["hilbert_eps"]) * 4096).astype(np.int64)
                centres = centre_by_index[native_indices]
                floor_cells = np.floor(unit*64).astype(np.int64).clip(0, 63)
                floor_centres = (floor_cells+.5)/64
                differs = np.any(centres != floor_centres, axis=1)
                mismatch_points += int(differs.sum())
                total_points += len(unit)
                mismatch_tracks += int(differs.any())
                if not np.array_equal(hilbert_numpy(centres, metadata["hilbert_eps"]),
                                      native_indices / 4096):
                    raise ValueError("Inverse-cell lookup changed the inherited Hilbert index")
                values.append(sampled(centres).T)
            grid = np.asarray(values, dtype=np.float32)
            if "grid_xy" in features and not np.array_equal(features["grid_xy"], grid):
                raise ValueError("An existing grid control disagrees with the frozen raw cohort")
            features["grid_xy"] = grid
            temporary = directory / "features-grid.tmp.npz"
            np.savez_compressed(temporary, **features)
            temporary.replace(destination)
            reports[f"{domain}/full/{split}"] = {"rows": len(grid),
                "class_counts": np.bincount(features["labels"], minlength=6).tolist(),
                "native_points": total_points, "global_floor_mismatch_points": mismatch_points,
                "global_floor_mismatch_fraction": mismatch_points / total_points,
                "global_floor_mismatch_tracks": mismatch_tracks}
    result = {"scope": "primary full CNR/SUMO cohorts only, exact existing IDs and labels",
              "transform": "shared unitXY bounds, exact inherited Hilbert-index inverse to 64x64 cell centres, then observation-index interpolation to224",
              "interpretation": "XY versus gridXY holds the two-channel readout architecture fixed and isolates coordinate quantization before interpolation",
              "boundary_policy": "inherited recursion, including reflected dyadic-boundary ties",
              "reports": reports}
    output = args.encoded / "grid-control-complete.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("g29-data/raw"))
    parser.add_argument("--encoded", type=Path, default=Path("g29-data/encoded"))
    add_control(parser.parse_args())
