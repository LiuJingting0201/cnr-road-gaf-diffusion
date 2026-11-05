"""Recover simulator-native road references for exactly the delivered, paired XY windows."""
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from g15_pack import choose


def replay(task):
    folder, selected, out = task
    seed = int(folder.name)
    with np.load(folder / "windows.npz") as data:
        ids = data["ids"][selected]
        starts = data["ticks"][selected]
        xy = data["xy"][selected]
        origins, destinations = data["origins"][selected], data["destinations"][selected]
    lookup = {ident: i for i, ident in enumerate(ids)}
    counts = np.zeros(len(ids), dtype=int)
    chains = [[] for _ in ids]
    with tempfile.TemporaryDirectory(dir=folder, prefix="road-replay-") as temporary:
        fcd = Path(temporary) / "fcd.xml"
        cmd = ["g10-env/sumo/bin/sumo", "-n", "data/maps/most.net.xml", "-r", str(folder / "demand.xml"),
               "--seed", str(seed), "--step-length", ".25", "--end", "310",
               "--time-to-teleport", "-1", "--device.fcd.begin", "80",
               "--fcd-output", str(fcd), "--fcd-output.attributes", "x,y,edge,lane",
               "--no-step-log", "true"]
        with (folder / "road-replay.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        for _, step in ET.iterparse(fcd, events=["end"]):
            if step.tag != "timestep":
                continue
            tick = round(float(step.attrib["time"]) * 4)
            for item in step:
                a = item.attrib
                i = lookup.get(a["id"])
                if i is None:
                    continue
                j = tick - starts[i]
                if j < 0 or j >= 512:
                    continue
                if counts[i] != j or abs(float(a["x"]) - xy[i, j, 0]) > .003 or abs(float(a["y"]) - xy[i, j, 1]) > .003:
                    raise ValueError(f"Replay coordinates diverged: {a['id']}, {j}")
                counts[i] += 1
                kind = "lane" if a.get("lane") else "edge"
                ref = a.get(kind)
                if not ref:
                    raise ValueError(f"Missing native road: {a['id']}, {j}")
                chain = chains[i]
                if not chain or chain[-1][1:] != (kind, ref):
                    chain.append((j, kind, ref))
            step.clear()
    if not (counts == 512).all():
        raise ValueError(f"Incomplete native annotation {seed}")
    with out.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for i, ident in enumerate(ids):
            for k, (j, kind, ref) in enumerate(chains[i]):
                stop = chains[i][k + 1][0] if k + 1 < len(chains[i]) else 512
                w.writerow((ident, origins[i], destinations[i], j, stop, kind, ref))
    return int(counts.sum())


def export(base):
    selected, _, _ = choose(base)
    results = {}
    path = base / "road-annotations.partial.zip"
    with tempfile.TemporaryDirectory(dir=base, prefix="roads-") as scratch:
        with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=4) as z:
            for split, groups in selected.items():
                tasks = [(base / split / str(seed), indices, Path(scratch) / f"{seed}.csv")
                         for seed, indices in sorted(groups.items())]
                with ProcessPoolExecutor(max_workers=12) as pool:
                    results[split] = sum(pool.map(replay, tasks))
                with z.open(split + ".csv", "w") as stream:
                    stream.write(b"id,origin_edge,destination_edge,start_index,stop_index,reference_kind,reference_id\n")
                    for _, _, table in tasks:
                        stream.write(table.read_bytes())
                        table.unlink()
            z.writestr("README.txt", "Join to the matching split by participant ID. Index intervals [start,stop) cover 512 positions; internal junctions retain SUMO IDs. Use realized road references only as training targets or audit annotations, never as future route inputs in a predictive evaluation.\n")
    with ZipFile(path) as z:
        if z.testzip():
            raise ValueError("Corrupt road annotation archive")
    main = json.loads((base / "complete.json").read_text())
    if any(n != main["splits"][split]["rows"] for split, n in results.items()):
        raise ValueError("Different trajectory and road coverage")
    final = base / "road-annotations.zip"
    path.replace(final)
    (base / "road-complete.json").write_text(json.dumps(dict(matched_points=results,
        package_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        trajectories_sha256=main["package_sha256"]), indent=2))


if __name__ == "__main__":
    export(Path(sys.argv[1]))
