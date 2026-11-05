"""Pack spatially sampled training-only path prefixes; hold future references apart."""
import csv
import io
import json
import os
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import shapely
from shapely.ops import substring
from cnr.data import CLASSES
from cnr.route_paths import RoutePaths

NAMES = CLASSES + ["truck", "delivery"]
GRID = np.linspace(0, 64, 65)
BASE = Path(os.getenv("CNR_G16_DATA_DIR", "g16-data"))
ORIGINAL = Path("data/processed/long-path-e3-512")
CORPUS = Path("../g15-job/g15-data")


def csv_tracks(split):
    with ZipFile(CORPUS / "balanced-trajectories.zip") as archive:
        with archive.open(split + ".csv") as raw:
            rows = io.TextIOWrapper(raw)
            next(rows)
            from itertools import islice
            while block := list(islice(rows, 512)):
                fields = [r.rstrip().split(",") for r in block]
                ident, label = fields[0][1], fields[0][4]
                if len(fields) != 512 or fields[-1][1] != ident:
                    raise ValueError("CSV window boundary mismatch")
                xy = np.array([(r[2], r[3]) for r in fields], dtype=np.float32)
                yield ident, NAMES.index(label), xy


def pack(name, records):
    retained, rejected = Counter(), Counter()
    targets, routes, starts, labels = [], [], [], []
    for _, lab, xy in records:
        xy = xy.astype(np.float64)
        inc = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arc = np.r_[0., np.cumsum(inc)]
        if arc[-1] < GRID[-1]:
            rejected[NAMES[lab]] += 1
            continue
        route = np.stack([np.interp(GRID, arc, coord) for coord in xy.T], axis=1)
        routes.append(route - xy[0])
        targets.append(np.log1p(inc))
        starts.append(xy[0])
        labels.append(lab)
        retained[NAMES[lab]] += 1
    np.savez(BASE / (name + ".npz"), route=np.array(routes, np.float32), start=np.array(starts, np.float32),
             target=np.array(targets, np.float32), label=np.array(labels, np.int64))
    return dict(retained=retained, short=rejected)


def walk(matcher, start):
    groups, _ = matcher.candidates(start[None], "pedestrian")
    for lane, s, _ in groups[0]:
        if matcher.road.lanes[lane]["function"] != "normal":
            continue
        shape = matcher.road.shapes[lane]
        end = shape.length if shape.length - s >= s else 0.
        if abs(end - s) < 220:
            continue
        segment = substring(shape, s, end)
        points = shapely.get_coordinates(shapely.line_interpolate_point(segment, np.linspace(0, segment.length, 1025)))
        return points, segment.length, lane
    return None


def requests():
    matcher = RoutePaths("data/maps/most.net.xml", radius=2.)
    prior = json.loads((ORIGINAL / "route_prior.json").read_text())
    counts = {(r["class"], matcher.ids[r["from"]], matcher.ids[r["to"]]):r["count"] for r in prior}
    groups = {}
    for ident, label, xy in csv_tracks("validation"):
        groups.setdefault((int(ident.split("_")[0]), label), []).append((ident, xy))
    selected, rejected = [], Counter()
    for (episode, label), rows in sorted(groups.items()):
        rng = np.random.default_rng(episode * 10 + label)
        n = 0
        for ident, ref in sorted(rows, key=lambda x:x[0]):
            start = ref[0]
            if label == 0:
                plan = walk(matcher, start)
            else:
                plan = matcher.sample(start, NAMES[label], counts, rng, budget=4096.)
                if plan is not None:
                    plan = (*plan, -1)
            if plan is None or plan[1] < 64:
                rejected[NAMES[label]] += 1
                continue
            route, length, lane = plan
            selected.append((ident, label, episode, start, route, length, lane, ref))
            n += 1
            if n == 2:
                break
    ids, labs, episodes, starts, routes, lengths, lanes, refs = map(np.asarray, zip(*selected))
    np.savez(BASE / "requests.npz", ids=ids, labels=labs, episodes=episodes, start=starts,
             route=routes.astype(np.float32), length=lengths.astype(np.float32), lane=lanes)
    np.savez(BASE / "references.npz", ids=ids, xy=refs)
    return dict(selected=Counter(NAMES[l] for l in labs), failed_before_quota=rejected)


if __name__ == "__main__":
    BASE.mkdir(exist_ok=True)
    original = np.load(ORIGINAL / "train.npz")
    stats = {"original":pack("original", zip(original["source_index"],original["labels"],original["xy"]))}
    stats["expanded"] = pack("expanded", csv_tracks("train"))
    stats["requests"] = requests()
    (BASE / "data-complete.json").write_text(json.dumps(stats, indent=2))
