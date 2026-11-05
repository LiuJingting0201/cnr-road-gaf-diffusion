"""Deliver equally represented, globally unique vehicles/people with auditable provenance."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from g15_episode import MODES
from g15_queue import SPLITS


def choose(base):
    seen_ids, seen_xy, demands, networks = set(), set(), set(), set()
    selections = {}
    summary = {}
    episodes = []
    for split, (first, quota) in SPLITS.items():
        choices = defaultdict(list)
        failures = defaultdict(Counter)
        eligible = Counter()
        duplicate = Counter()
        total = Counter()
        folder = base / split
        for file in sorted(folder.glob("*/complete.json")):
            info = json.loads(file.read_text())
            episodes.append(dict(split=split, **info))
            seed = info["seed"]
            if not first <= seed < first + 1000 or info["demand_sha256"] in demands:
                raise ValueError("Episode or demand overlap")
            demands.add(info["demand_sha256"])
            networks.add(info["net_sha256"])
            for label in MODES:
                total[label] += info["requested_per_class"]
                n = info["eligible"].get(label, 0)
                rejected = info["rejected"].get(label, {})
                if n + sum(rejected.values()) != info["requested_per_class"]:
                    raise ValueError(f"Unaccounted IDs: {seed} {label}")
                failures[label].update(rejected)
            with np.load(file.parent / "windows.npz") as data:
                xy = data["xy"]
                if xy.shape != (len(data["ids"]), 512, 2) or not np.isfinite(xy).all():
                    raise ValueError(f"Invalid window file: {seed}")
                for i, (ident, label) in enumerate(zip(data["ids"], data["labels"])):
                    ident, label = str(ident), str(label)
                    if ident in seen_ids or not ident.startswith(f"{seed}_{label}_"):
                        raise ValueError(f"Duplicate or incorrectly namespaced ID: {ident}")
                    seen_ids.add(ident)
                    eligible[label] += 1
                    fingerprint = hashlib.sha256(np.rint(xy[i] * 100).astype("<i4").tobytes()).digest()
                    if fingerprint in seen_xy:
                        duplicate[label] += 1
                        continue
                    seen_xy.add(fingerprint)
                    rank = hashlib.sha256(("G15|" + ident).encode()).digest()
                    choices[label].append((rank, seed, i))
        selected = defaultdict(list)
        for label in MODES:
            ranked = sorted(choices[label])
            if len(ranked) < quota:
                raise ValueError(f"Too few unique windows: {split}, {label}")
            for _, seed, index in ranked[:quota]:
                selected[seed].append(index)
        selections[split] = selected
        summary[split] = dict(episodes=len(list(folder.glob("*/complete.json"))),
                              requested=dict(total), eligible=dict(eligible), rejected=failures,
                              duplicates=dict(duplicate), delivered={label: quota for label in MODES},
                              rows=quota * len(MODES) * 512)
    if len(networks) != 1:
        raise ValueError("Inconsistent network")
    return selections, summary, episodes


def write_episode(task):
    source, out, indices = task
    routes = defaultdict(set)
    with np.load(source) as data, out.open("w", newline="") as stream:
        ids, labels = data["ids"], data["labels"]
        ticks, xy = data["ticks"], data["xy"]
        origins, destinations = data["origins"], data["destinations"]
        writer = csv.writer(stream, lineterminator="\n")
        for i in sorted(indices):
            ident, label = str(ids[i]), str(labels[i])
            routes[label].add((str(origins[i]), str(destinations[i])))
            writer.writerows((f"{(ticks[i] + j) / 4:.2f}", ident,
                              f"{x:.2f}", f"{y:.2f}", label)
                             for j, (x, y) in enumerate(xy[i]))
    return routes


def pack(base):
    selections, summary, episodes = choose(base)
    temp = base / "balanced-trajectories.partial.zip"
    with tempfile.TemporaryDirectory(dir=base, prefix="csv-") as scratch:
        with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            for split, selected in selections.items():
                files = [(base / split / str(seed) / "windows.npz", Path(scratch) / f"{seed}.csv", idx)
                         for seed, idx in sorted(selected.items())]
                od = defaultdict(set)
                with ProcessPoolExecutor(max_workers=12) as pool:
                    for result in pool.map(write_episode, files):
                        for label, pairs in result.items():
                            od[label].update(pairs)
                summary[split]["distinct_delivered_od_pairs"] = {k: len(v) for k, v in od.items()}
                with archive.open(split + ".csv", "w", force_zip64=True) as stream:
                    stream.write(b"time,id,x,y,type\n")
                    for _, file, _ in files:
                        with file.open("rb") as rows:
                            shutil.copyfileobj(rows, stream, length=1024 * 1024)
                        file.unlink()
            manifest = dict(sumo=episodes[0]["sumo"], network_sha256=episodes[0]["net_sha256"],
                            splits=summary, episodes=episodes)
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    with ZipFile(temp) as archive:
        for split, stats in summary.items():
            counts = Counter()
            with archive.open(split + ".csv") as raw:
                reader = csv.reader(io.TextIOWrapper(raw))
                if next(reader) != ["time", "id", "x", "y", "type"]:
                    raise ValueError("Malformed CSV header")
                previous = None
                ids, n, start_tick = set(), 0, 0
                for row in reader:
                    t, ident, _, _, label = row
                    if ident != previous:
                        if previous is not None and n != 512:
                            raise ValueError("Incomplete exported window")
                        if ident in ids or not ident.startswith(ident.split("_")[0] + "_" + label + "_"):
                            raise ValueError(f"Mixed IDs or labels: {ident}")
                        ids.add(ident)
                        counts[label] += 1
                        previous, n, start_tick = ident, 0, round(float(t) * 4)
                    if round(float(t) * 4) != start_tick + n:
                        raise ValueError(f"CSV time gap: {ident}")
                    n += 1
            if n != 512 or counts != stats["delivered"] or len(ids) * 512 != stats["rows"]:
                raise ValueError(f"Incorrect class counts: {split}")
    final = base / "balanced-trajectories.zip"
    temp.replace(final)
    digest = hashlib.sha256()
    with final.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    receipt = dict(sumo=manifest["sumo"], network_sha256=manifest["network_sha256"],
                   splits=summary, package_bytes=final.stat().st_size, package_sha256=digest.hexdigest())
    pending = base / "complete.tmp"
    pending.write_text(json.dumps(receipt, indent=2) + "\n")
    pending.replace(base / "complete.json")


if __name__ == "__main__":
    pack(Path(sys.argv[1]))
