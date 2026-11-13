"""Write the cadence comparison as soon as it is ready, then the long-trip addendum."""
import argparse
import fcntl
import json
import math
import os
import time
import traceback
from pathlib import Path

SEEDS = (42, 43)
ARMS = ("one", "fifteen")
METRICS = ("density_js", "od_js", "length_js", "top20_cell_f1")
TABLE = [
    "| Cohort | Seed | Density JS ↓ | Joint OD JS ↓ | Length JS ↓ | Hotspot F1 ↑ |",
    "|---|---:|---:|---:|---:|---:|",
]


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(value)
    temp.replace(path)


def stamp(path, value):
    atomic(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def metric_values(record):
    values = [record.get(k) for k in METRICS]
    if values[1] is None:
        values[1] = record["trip_js"]
    if any(v is None or not math.isfinite(v) for v in values):
        raise ValueError(f"Incomplete or non-finite metrics: {values}")
    return values


def table_row(label, seed, values, signed=False):
    spec = "+.4f" if signed else ".4f"
    return f"| {label} | {seed} | " + " | ".join(format(v, spec) for v in values) + " |"


def load_native(path):
    record = read(path)
    if record["requests"] != 1000 or record["epoch"] != 200:
        raise ValueError(f"Unexpected sample count or training epoch in {path}")
    return record


def cadence(root):
    common = read(root / "cadence-common.json")
    scores = common["shared_15s_clock"]
    native = {}
    lines = [
        "# SUMO taxi observation cadence", "",
        "The two training representations use the same 10,000 SUMO journeys. The "
        "corrected one-second fits are reused. The 15-second fits are retrained. Each "
        "row evaluates the final checkpoint on 1,000 paired development requests.", "",
        "## Both models scored on the common 15-second observation clock", "", *TABLE,
    ]
    for arm in ARMS:
        native[arm] = {}
        label = "1 s" if arm == "one" else "15 s"
        for seed in SEEDS:
            native[arm][seed] = load_native(root / arm / str(seed) / "development.json")
            values = metric_values(scores[arm][str(seed)])
            lines.append(table_row(label, seed, values))
    for seed in SEEDS:
        if native["one"][seed]["request_index_sha256"] != native["fifteen"][seed]["request_index_sha256"]:
            raise ValueError(f"Unaligned request indices at seed {seed}")
        before, after = [metric_values(scores[arm][str(seed)]) for arm in ARMS]
        lines.append(table_row("15 s minus 1 s", seed, [a-b for a,b in zip(after,before)], signed=True))
    lines.extend(["", "## Each model evaluated on its native observation clock", "", *TABLE])
    for arm, label in zip(ARMS, ("1 s", "15 s")):
        for seed in SEEDS:
            lines.append(table_row(label, seed, metric_values(common["native_clock"][arm][str(seed)])))
    lines.extend(["", "## Physical length of the scored paths", "",
                  "Median lengths and W1 are in kilometres. The observed median can change "
                  "when a coarser coordinate sequence cuts corners on the same trip.", "",
                  "| Cadence | Seed | Observed median | Generated median | Length W1 |",
                  "|---|---:|---:|---:|---:|"])
    for arm, label in zip(ARMS, ("1 s", "15 s")):
        for seed in SEEDS:
            rec = native[arm][seed]
            lengths = [rec[k]/1000 for k in ("reference_median_m","generated_median_m","length_w1_m")]
            lines.append(f"| {label} | {seed} | " + " | ".join(f"{v:.3f}" for v in lengths) + " |")
    lines.extend(["", "The four primary columns use the same 15-second observation clock, training-derived grid and "
                  "length-bin boundaries for both representations. Down arrows favour smaller "
                  "values. For F1, larger overlap is better. A per-request hit rate and an "
                  "aggregate joint OD divergence answer different questions.", ""])
    atomic(root / "CADENCE_REPORT.md", "\n".join(lines))
    stamp(root / "cadence-report.json", {"common": common, "per_arm_scores": native})


def long_results(long_root, porto):
    lines = ["## Long trips under a 15-second observation rule", "",
             "The SUMO training cohort meets the 120-observation threshold after "
             "sampling at 15 seconds. The final comparison uses 1,000 held-out requests per "
             "population. Porto and SUMO each use their own training coordinates for the "
             "grid and length bins. Differences between cities include OD support, spatial "
             "extent and route mix.", "", *TABLE]
    data = {"SUMO": {}, "Porto": {}}
    for label, folder, file in (("SUMO", long_root, "geolife-final.json"), ("Porto", porto, "porto-final.json")):
        for seed in SEEDS:
            rec = load_native(folder / str(seed) / file)
            lines.append(table_row(label, seed, metric_values(rec)))
            data[label][seed] = rec
    for seed in SEEDS:
        sumo, port = (metric_values(data[k][seed]) for k in ("SUMO", "Porto"))
        lines.append(table_row("SUMO minus Porto", seed, [s-p for s,p in zip(sumo,port)], signed=True))
    lines.extend(["", "The previously published Porto row in Table 5 gives 0.0087, "
                  "0.0132, 0.0242 and 0.847. The matched local Porto fits provide the "
                  "10,000-training-trip comparator above.", ""])
    return lines, data


def stage(root, good, bad):
    if (root / bad).exists():
        return "failed"
    if (root / good).exists():
        return "complete"
    return "waiting"


def monitor(root, long_root, porto, interval):
    while True:
        c = stage(root, "CADENCE_COMPLETE", "CADENCE_FAILED")
        l = stage(long_root, "COMPLETE", "FAILED")
        status = {"pid": os.getpid(), "cadence": c, "long15": l,
                  "long_data_ready": (long_root / "DATA_READY").exists(),
                  "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if c == "complete" and not (root / "CADENCE_REPORT_READY").exists():
            cadence(root)
            atomic(root / "CADENCE_REPORT_READY", "ok\n")
        if l == "complete" and not (root / "LONG_REPORT_READY").exists():
            lines, data = long_results(long_root, porto)
            atomic(root / "LONG_REPORT.md", "\n".join(lines))
            stamp(root / "long-report.json", data)
            atomic(root / "LONG_REPORT_READY", "ok\n")
        stamp(root / "report-status.json", status)
        if c != "waiting" and l != "waiting":
            lines = []
            for state, directory, failure, filename in (
                (c, root, "CADENCE_FAILED", "CADENCE_REPORT.md"),
                (l, long_root, "FAILED", "LONG_REPORT.md"),
            ):
                if state == "complete":
                    lines.append((root / filename).read_text())
                else:
                    lines.append(f"Stage failure recorded at {directory / failure}. "
                                 "Its metrics are unavailable.\n")
            atomic(root / "COMBINED_REPORT.md", "\n".join(lines))
            marker = "REPORT_COMPLETE" if c == l == "complete" else "REPORT_INCOMPLETE"
            atomic(root / marker, "See report-status.json and COMBINED_REPORT.md\n")
            return
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--long-root", type=Path)
    parser.add_argument("--porto", type=Path)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    root = args.root.resolve()
    long_root = args.long_root or root.parent / "taxi-long15-v1"
    porto = args.porto or root.parent / "taxi-repro-debug-v1" / "porto"
    with (root / "report-watch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        try:
            monitor(root, long_root, porto, args.interval)
        except Exception:
            atomic(root / "REPORT_FAILED", traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
