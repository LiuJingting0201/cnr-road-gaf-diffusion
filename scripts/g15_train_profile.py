"""Describe only the delivered training windows, in physical units."""
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np


def profile(base):
    # Read training IDs from the delivered train road table (one or more runs per ID).
    with ZipFile(base / "road-annotations.zip") as archive:
        with archive.open("train.csv") as entries:
            ids = {row["id"] for row in csv.DictReader(io.TextIOWrapper(entries))}
    result = defaultdict(list)
    for file in sorted((base / "train").glob("*/windows.npz")):
        with np.load(file) as data:
            xy, labels = data["xy"], data["labels"]
            for i, ident in enumerate(data["ids"]):
                if ident in ids:
                    steps = np.linalg.norm(np.diff(xy[i].astype(float), axis=0), axis=1)
                    result[str(labels[i])].append((steps.sum(), np.mean(steps * 4 < .1)))
    report = {}
    for label, rows in result.items():
        values = np.asarray(rows)
        if len(rows) != 8000:
            raise ValueError(f"Unexpected training coverage {label}")
        report[label] = dict(windows=len(rows), travel_m_q10_q50_q90=np.quantile(values[:, 0], [.1,.5,.9]).tolist(),
                             mean_near_stop_fraction=float(values[:, 1].mean()))
    (base / "train-profile.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    profile(Path(sys.argv[1]))
