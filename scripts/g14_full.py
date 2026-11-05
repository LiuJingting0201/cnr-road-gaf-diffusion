"""Export full target-vehicle FCD records, linked by ID to the strict 512-point windows."""
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree.ElementTree import iterparse
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from cnr.data import CLASSES


def export_episode(folder, writer, counts):
    tracks = defaultdict(list)
    for _, step in iterparse(folder / "fcd.xml", events=["end"]):
        if step.tag != "timestep":
            continue
        time = step.attrib["time"]
        for vehicle in step.iterfind("vehicle"):
            ident = vehicle.attrib["id"]
            if int(ident.rsplit("_", 1)[1]) < 20:
                tracks[ident].append((time, vehicle.attrib["x"], vehicle.attrib["y"]))
        step.clear()
    gaps = 0
    with np.load(folder / "eligible.npz") as subset:
        for ident, t0, xy in zip(subset["ids"], subset["start_time"], subset["xy"]):
            match = [row[1:] for row in tracks[ident] if t0 <= float(row[0]) <= t0 + 127.75]
            if not np.array_equal(np.asarray(match, dtype=float), xy):
                raise ValueError(f"Window does not match full FCD: {ident}")
    for ident, rows in sorted(tracks.items()):
        sim, label, _ = ident.split("_")
        if sim != folder.name:
            raise ValueError(f"Episode mismatch {ident}")
        coords = np.asarray(rows, dtype=float)
        dt = np.diff(coords[:, 0])
        if not np.isfinite(coords).all() or (dt <= 0).any():
            raise ValueError(f"Invalid vehicle data {ident}")
        gaps += int((dt != 0.25).any())
        name = CLASSES[int(label)]
        counts[name] += 1
        writer.writerows((t, ident, x, y, name) for t, x, y in rows)
    return sum(map(len, tracks.values())), gaps


def export(base):
    manifest = json.loads((base / "complete.json").read_text())
    report = {}
    temp = base / "full-trajectories.partial.zip"
    with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
        for split in manifest["splits"]:
            counts = Counter()
            rows = gaps = 0
            with archive.open(split + ".csv", "w") as raw, io.TextIOWrapper(raw, newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time", "id", "x", "y", "type"])
                for info in manifest["episodes"]:
                    if info["split"] != split:
                        continue
                    n, g = export_episode(base / str(info["seed"]), writer, counts)
                    rows += n
                    gaps += g
            report[split] = dict(classes=dict(counts), tracks=sum(counts.values()), rows=rows,
                                 tracks_with_time_gaps=gaps)
        archive.writestr("manifest.json", json.dumps(report, indent=2))
    with ZipFile(temp) as archive:
        if archive.testzip():
            raise ValueError("Full archive CRC failure")
        for split, totals in report.items():
            with archive.open(split + ".csv") as stream:
                if sum(1 for _ in stream) != totals["rows"] + 1:
                    raise ValueError(f"Full CSV row count differs for {split}")
    final = base / "full-trajectories.zip"
    temp.replace(final)
    summary = dict(splits=report, package_bytes=final.stat().st_size,
                   package_sha256=hashlib.sha256(final.read_bytes()).hexdigest())
    (base / "full-complete.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    export(Path(sys.argv[1]))
