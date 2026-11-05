"""One episode of SUMO-observed mixed traffic; a single window per independent agent."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import sumolib

# The bicycle label in the CNR tables is spelled bycicle.
MODES = {"pedestrian": "pedestrian", "bus": "bus", "motorcycle": "motorcycle",
         "private_car": "passenger", "bycicle": "bicycle", "taxi_or_uber": "taxi",
         "truck": "truck", "delivery": "delivery"}


def build_demand(network, seed, n, filename):
    rng = np.random.default_rng(seed)
    root = ET.Element("routes")
    jobs = []
    metadata = {}
    for label, vc in MODES.items():
        if label != "pedestrian":
            ET.SubElement(root, "vType", id=label, vClass=vc)
        edges = [e for e in network.getEdges() if not e.isSpecial() and e.allows(vc) and e.getLength() > 20]
        for slot in range(n):
            for _ in range(3000):
                a, b = rng.choice(len(edges), 2, replace=False)
                source, dest = edges[a], edges[b]
                if label == "pedestrian":
                    start = np.array(source.getShape()[0])
                    if 350 <= np.linalg.norm(start - dest.getShape()[-1]) <= 1200:
                        # SUMO's pedestrian router can traverse sidewalks in both directions.
                        path = [source, dest]
                        break
                    continue
                path, length = network.getShortestPath(source, dest, vClass=vc)
                if path and 2500 <= length <= 9000:
                    break
            else:
                raise RuntimeError(f"No reachable OD pair for {label}")
            ident = f"{seed}_{label}_{slot:04}"
            metadata[ident] = (label, source.getID(), dest.getID())
            jobs.append((rng.uniform(0, 100), ident, [e.getID() for e in path]))
    for depart, ident, path in sorted(jobs):
        label, source, dest = metadata[ident]
        if label == "pedestrian":
            person = ET.SubElement(root, "person", id=ident, depart=f"{depart:.2f}")
            ET.SubElement(person, "walk", {"from": source, "to": dest})
        else:
            vehicle = ET.SubElement(root, "vehicle", id=ident, type=label,
                                    depart=f"{depart:.2f}", departLane="best")
            ET.SubElement(vehicle, "route", edges=" ".join(path))
    ET.ElementTree(root).write(filename)
    return metadata


def simulate_reachable(cmd, demand, out):
    """Remove only person ODs rejected by the SUMO walking router and retain its evidence."""
    excluded = set()
    while True:
        with (out / "sumo.log").open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return excluded
        message = (out / "sumo.log").read_text()
        match = re.search(r"Error: No connection found .* for person '([^']+)'", message)
        if not match or match[1] in excluded:
            raise RuntimeError(message[-1800:])
        failed_id = match[1]
        tree = ET.parse(demand)
        person = next((p for p in tree.getroot().findall("person") if p.get("id") == failed_id), None)
        if person is None:
            raise ValueError(f"Missing unroutable pedestrian {failed_id}")
        tree.getroot().remove(person)
        tree.write(demand)
        excluded.add(failed_id)
        with (out / "walking-router-rejections.log").open("a") as stream:
            stream.write(match[0] + "\n")


def run(base, seed, n):
    out = base / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").is_file():
        return
    rng = np.random.default_rng(seed + 1000000)
    netpath = Path("data/maps/most.net.xml")
    network = sumolib.net.readNet(str(netpath))
    demand = out / "demand.xml"
    meta = build_demand(network, seed, n, demand)
    with tempfile.TemporaryDirectory(prefix="scratch-", dir=out) as scratch:
        fcd = Path(scratch) / "fcd.xml"
        binary = "g10-env/sumo/bin/sumo"
        cmd = [binary, "-n", str(netpath), "-r", str(demand), "--seed", str(seed),
               "--step-length", ".25", "--end", "310", "--time-to-teleport", "-1",
               "--fcd-output", str(fcd), "--device.fcd.begin", "80",
               "--fcd-output.attributes", "x,y,edge,lane", "--no-step-log", "true"]
        excluded = simulate_reachable(cmd, demand, out)
        tracks = defaultdict(list)
        for _, step in ET.iterparse(fcd, events=["end"]):
            if step.tag != "timestep":
                continue
            tick = round(4 * float(step.attrib["time"]))
            for item in step:
                a = item.attrib
                tracks[a["id"]].append((tick, float(a["x"]), float(a["y"]),
                                         a.get("lane") or a.get("edge", "")))
            step.clear()
    kept, rejected = [], defaultdict(Counter)
    for ident, (label, origin, dest) in meta.items():
        if ident in excluded:
            rejected[label]["unreachable_walking_od"] += 1
            continue
        rows = tracks.get(ident, [])
        if len(rows) < 512:
            rejected[label]["short_or_absent"] += 1
            continue
        offset = int(rng.integers(0, len(rows) - 511))
        win = rows[offset:offset + 512]
        xy = np.array([(r[1], r[2]) for r in win], dtype=np.float32)
        if any(r[0] != win[0][0] + i or not r[3] for i, r in enumerate(win)) or not np.isfinite(xy).all():
            rejected[label]["time_or_road_reference"] += 1
            continue
        max_step = 1.25 if label == "pedestrian" else 15.0
        if np.linalg.norm(np.diff(xy, axis=0), axis=1).max() > max_step:
            rejected[label]["position_jump"] += 1
            continue
        if np.linalg.norm(xy[-1] - xy[0]) <= 5:
            rejected[label]["stationary"] += 1
            continue
        kept.append((ident, label, origin, dest, win[0][0], xy))
    if not kept:
        raise ValueError("No surviving windows")
    ids, labels, origins, destinations, ticks, xy = zip(*kept)
    np.savez_compressed(out / "windows.npz", ids=ids, labels=labels, origins=origins,
                        destinations=destinations, ticks=ticks, xy=xy)
    receipt = dict(seed=seed, requested_per_class=n, eligible=dict(Counter(labels)),
                   rejected=rejected, net_sha256=hashlib.sha256(netpath.read_bytes()).hexdigest(),
                   demand_sha256=hashlib.sha256(demand.read_bytes()).hexdigest(),
                   sumo=subprocess.check_output([binary, "--version"], text=True).splitlines()[0])
    pending = out / "complete.tmp"
    pending.write_text(json.dumps(receipt, indent=2))
    pending.replace(out / "complete.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("n", type=int)
    opts = parser.parse_args()
    run(opts.base, opts.seed, opts.n)
