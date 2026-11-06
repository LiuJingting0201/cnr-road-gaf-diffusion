"""Read-only post-completion teleport coverage audit; no cohort or model changes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


CLASSES = ["pedestrian", "bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
START = re.compile(r"teleporting vehicle ['\"]([^'\"]+)['\"]", re.IGNORECASE)
WHEN = re.compile(r"\btime[=: ]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(root, output):
    if output.exists():
        raise FileExistsError("Existing aggregate audit is preserved")
    loaded, selected, base_seen = {}, {}, set()
    hashes = {}
    for split in ("train", "val"):
        path = root/"raw"/f"native-{split}.npz"
        with np.load(path, allow_pickle=False) as z:
            # NPZ members are materialized exactly once, never inside the row loop.
            rows = {key: z[key] for key in ("base_ids", "episodes", "labels", "offset", "txy", "break_count")}
        loaded[split] = rows
        hashes[f"native-{split}.npz"] = digest(path)
        for i, (base, episode, label) in enumerate(zip(rows["base_ids"], rows["episodes"], rows["labels"])):
            key = (int(episode), str(base))
            if key in selected or str(base) in base_seen:
                raise ValueError("Frozen cohort repeats a base entity")
            selected[key] = (split, i, int(label))
            base_seen.add(str(base))

    by_episode, coverage, effects = {}, Counter(), defaultdict(Counter)
    event_times, core_rates = defaultdict(list), defaultdict(list)
    for split in ("train", "val"):
        for folder in sorted((root/"native"/split).iterdir()):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            receipt_path = folder/"simulation-complete.json"
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("complete") is not True:
                raise ValueError("An audited simulation is incomplete")
            log_path = folder/"console.log"
            log = log_path.read_text(errors="replace")
            episode = int(folder.name)
            row = Counter()
            row["summary_teleports"] = int(receipt["last_summary"].get("teleports", 0))
            row["teleport_keyword_lines"] = sum("teleport" in line.lower() for line in log.splitlines())
            for line in log.splitlines():
                if "teleporting vehicle" not in line.lower():
                    continue
                row["teleport_start_lines"] += 1
                match = START.search(line)
                if match is None:
                    row["unparsed_start_lines"] += 1
                    continue
                row["parsed_start_events"] += 1
                base = match.group(1)
                if base in base_seen:
                    row["same_base_in_any_selected_realisation"] += 1
                hit = selected.get((episode, base))
                if hit is None:
                    row["start_events_not_in_selected_realisation"] += 1
                    continue
                row["start_events_in_selected_realisation"] += 1
                selected_split, i, label = hit
                rows = loaded[selected_split]
                left, right = rows["offset"][i:i+2]
                txy = rows["txy"][left:right]
                dt = np.diff(txy[:, 0])
                rate = np.linalg.norm(np.diff(txy[:, 1:], axis=0), axis=1)/dt
                if not np.allclose(dt, .25, atol=1e-7, rtol=0) or np.any(rate > 100.):
                    raise ValueError("Frozen core violates inherited continuity criterion")
                group = f"{selected_split}/{CLASSES[label]}"
                effects[group]["start_events"] += 1
                effects[group]["core_has_prior_or_later_continuity_break"] += int(rows["break_count"][i] > 0)
                core_rates[group].append(float(rate.max()))
                tm = WHEN.search(line)
                if tm is None:
                    effects[group]["unparsed_event_time"] += 1
                    continue
                time = float(tm.group(1))
                event_times[group].append(time)
                if txy[0, 0] < time < txy[-1, 0]:
                    effects[group]["event_time_strictly_inside_core"] += 1
                elif np.isclose(time, txy[0, 0], atol=1e-7, rtol=0) or np.isclose(time, txy[-1, 0], atol=1e-7, rtol=0):
                    effects[group]["event_time_at_core_boundary"] += 1
                else:
                    effects[group]["event_time_outside_core"] += 1
            coverage.update(row)
            by_episode[f"{split}/{episode}"] = {**dict(row), "console_sha256": digest(log_path),
                "simulation_receipt_sha256": digest(receipt_path)}
    event_audit = {}
    for group, counts in effects.items():
        event_audit[group] = {**dict(counts),
            "maximum_retained_core_displacement_rate_mps": max(core_rates[group]),
            "parsed_event_time_range_s": [min(event_times[group]), max(event_times[group])] if event_times[group] else None}
    report = {"complete": True, "scope": "post-completion read-only audit of native matched G30 cohorts; no exclusions, refits or metric changes",
        "input_cohort_hashes": hashes, "script_sha256": digest(Path(__file__)),
        "method": "Parse SUMO teleport-start warnings, join exact (episode,base ID) to already selected native train/val rows, inspect retained core interval and displacement rate; no identifiers are exported",
        "continuity_rule": "Inherited g29_prepare.record -> cnr.data.continuous_ranges: split when delta time differs from .25s or displacement/delta-time >100m/s; keep longest segment >=5s",
        "coverage": dict(coverage),
        "parsed_start_count_equals_summary_teleports": coverage["parsed_start_events"] == coverage["summary_teleports"],
        "by_episode": by_episode, "selected_event_groups": event_audit,
        "limits": ["Teleport keyword lines are not teleport event counts.",
                   "No entity-wide exclusion based on teleport warnings was used to form the frozen cohort.",
                   "Time gaps and jumps above 25m per .25s are split; smaller single-step discontinuities are not guaranteed removed.",
                   "This audit does not test graph connectivity or all forms of abnormal motion.",
                   "An event time outside a retained core does not prove the preceding congestion or subsequent reinsertion had no influence.",
                   "The same base entity teleporting under a different seed is separate from the realisation selected for this cohort."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(output)
    print(json.dumps({"complete": True, "coverage": report["coverage"],
                      "selected_event_groups": report["selected_event_groups"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("g30-data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output or args.root/"teleport-audit.json")
