"""Export verified SUMO observations using complete simulation episodes as split units."""
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from cnr.data import CLASSES


def export(base):
    groups = {"train": range(261001, 261065), "validation": range(261065, 261073),
              "holdout": range(261073, 261081)}
    report = {"splits": {}, "episodes": []}
    ids, traces, demands, fcd_hashes, networks = set(), set(), set(), set(), set()
    temp = base / "sumo-trajectories.partial.zip"
    with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
        for split, seeds in groups.items():
            counts = Counter()
            rejected = Counter()
            with archive.open(split + ".csv", "w") as raw, io.TextIOWrapper(raw, newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time", "id", "x", "y", "type"])
                for seed in seeds:
                    folder = base / str(seed)
                    info = json.loads((folder / "manifest.json").read_text())
                    demands.add(info["demand"])
                    networks.add(info["net"])
                    fcd_hashes.add(info["fcd"])
                    with np.load(folder / "eligible.npz") as data:
                        n = info["eligible"]
                        if data["xy"].shape != (n, 512, 2) or any(len(data[k]) != n for k in ("ids", "labels", "start_time")):
                            raise ValueError(f"Incomplete episode {seed}")
                        if info["seed"] != seed or n + sum(info["rejections"].values()) != 100:
                            raise ValueError(f"Invalid accounting for {seed}")
                        rejected.update(info["rejections"])
                        report["episodes"].append(dict(split=split, **info))
                        for ident, label, xy, t0 in zip(data["ids"], data["labels"], data["xy"], data["start_time"]):
                            digest = hashlib.sha256(xy.tobytes()).digest()
                            if ident in ids or digest in traces or not str(ident).startswith(f"{seed}_"):
                                raise ValueError(f"Duplicate or misassigned track {ident}")
                            if label not in range(1, 6) or not np.isfinite(xy).all() or not np.isfinite(t0):
                                raise ValueError(f"Invalid data {ident}")
                            ids.add(ident)
                            traces.add(digest)
                            name = CLASSES[label]
                            counts[name] += 1
                            writer.writerows((f"{t0+i*.25:.2f}", ident, f"{x:.2f}", f"{y:.2f}", name)
                                             for i, (x, y) in enumerate(xy))
            report["splits"][split] = dict(episodes=len(seeds), target=100 * len(seeds),
                                            counts=dict(counts), tracks=sum(counts.values()),
                                            rows=sum(counts.values()) * 512, rejections=dict(rejected))
        if len(networks) != 1 or len(demands) != 80 or len(fcd_hashes) != 80:
            raise ValueError("Unexpected map or repeated simulation")
        archive.writestr("manifest.json", json.dumps(report, indent=2))
    with ZipFile(temp) as archive:
        if archive.testzip():
            raise ValueError("Invalid archive")
        for split, counts in report["splits"].items():
            with archive.open(split + ".csv") as stream:
                if sum(1 for _ in stream) != counts["rows"] + 1:
                    raise ValueError(f"CSV count differs for {split}")
    package = base / "sumo-trajectories.zip"
    temp.replace(package)
    report["package_sha256"] = hashlib.sha256(package.read_bytes()).hexdigest()
    report["package_bytes"] = package.stat().st_size
    pending = base / "complete.tmp"
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(base / "complete.json")


if __name__ == "__main__":
    export(Path(sys.argv[1]))
