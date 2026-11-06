"""Read-only event audit of the frozen G33 reference targets; no exclusions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(prepared, simulation, extracted, output):
    if output.exists():
        raise FileExistsError("Preserve the existing target-event audit")
    requests_path = prepared / "requests.json"
    receipt_path = simulation / "simulation-complete.json"
    extraction_path = extracted / "complete.json"
    if not json.loads(receipt_path.read_text())["complete"] or not json.loads(extraction_path.read_text())["complete"]:
        raise ValueError("Reference queue has not completed")
    requests = {r["id"]: r for r in json.loads(requests_path.read_text())["requests"]}
    statuses = {r["id"]: r["status"] for r in json.loads((extracted / "reference-metadata.json").read_text())["rows"]}
    log_path = simulation / "console.log"
    counts, by_class, affected, completed = Counter(), defaultdict(Counter), defaultdict(set), defaultdict(set)
    for line in log_path.read_text(errors="replace").splitlines():
        low = line.lower()
        if "teleport" in low:
            counts["teleport_keyword_lines"] += 1
        start = re.search(r"^\s*warning:\s*teleporting vehicle ['\"]([^'\"]+)['\"]", line, re.IGNORECASE)
        kind = ("teleport_start" if start else "emergency_stop" if "performs emergency stop" in low else
                "emergency_braking" if "emergency braking" in low else
                "collision_warning" if "warning:" in low and "collision" in low else None)
        if kind is None:
            continue
        counts[kind + "_all_lines"] += 1
        ids = [start.group(1)] if start else re.findall(r"['\"]([^'\"]+)['\"]", line)
        targets = set(ids) & set(requests)
        counts[kind + "_target_lines"] += bool(targets)
        for ident in targets:
            by_class[requests[ident]["class"]][kind] += 1
            affected[kind].add(ident)
            if statuses[ident] == "reference_complete":
                completed[kind].add(ident)
    last = list(ET.parse(simulation / "summary.xml").getroot())[-1].attrib
    summary_teleports = int(last.get("teleports", 0))
    result = {"complete": True, "scope": "read_only_frozen_target_log_join_no_cohort_changes",
              "request_denominator": len(requests), "counts": dict(counts),
              "summary_teleports_all_traffic": summary_teleports,
              "teleport_start_count_matches_summary": counts["teleport_start_all_lines"] == summary_teleports,
              "affected_target_counts": {k: len(v) for k, v in affected.items()},
              "affected_completed_reference_counts": {k: len(v) for k, v in completed.items()},
              "by_class_target_event_participations": {k: dict(v) for k, v in by_class.items()},
              "hashes": {"requests": sha(requests_path), "simulation_receipt": sha(receipt_path),
                         "extraction_receipt": sha(extraction_path), "console_log": sha(log_path), "script": sha(Path(__file__))},
              "limits": ["No trajectory was removed, replaced, sliced or resimulated using these warnings.",
                         "Completion and continuity do not alone establish physical normality; affected complete references are reported explicitly.",
                         "Absence of direct target warnings does not exclude effects of background congestion or unlogged abnormalities.",
                         "Collision event-participation counts may exceed event-line counts when both participants are targets."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("g33-data/requests"))
    parser.add_argument("--simulation", type=Path, default=Path("g33-data/reference-330201"))
    parser.add_argument("--extracted", type=Path, default=Path("g33-data/reference-extracted"))
    parser.add_argument("--output", type=Path, default=Path("g33-data/reference-extracted/target-events.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.prepared, args.simulation, args.extracted, args.output), indent=2))
