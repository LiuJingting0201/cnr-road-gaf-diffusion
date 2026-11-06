"""Pinned public MoST demand: isolated downloads and bounded native SUMO runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET

COMMIT = "b29b2f65f1096a9c69a601ec62a724815cb4a43f"
NETWORK_SHA = "a06b5c0765fb2d49d4ad33cf54048f15f2ff1592b2b3d0180dbeec4feb73bfc0"
FILES = ["scenario/most.sumocfg", "scenario/in/most.net.xml",
         *[f"scenario/in/route/most.{name}.xml" for name in
           ("buses.flows", "trains.flows", "pedestrian.rou", "commercial.rou", "special.rou", "highway.flows")],
         *[f"scenario/in/add/{name}.xml" for name in
           ("most.poly", "most.busstops.add", "most.trainstops.add", "most.parking.allvisible.add", "basic.vType")]]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def fetch(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/"complete.json").exists():
        manifest = json.loads((root/"complete.json").read_text())
        assert manifest["commit"] == COMMIT
        for name, row in manifest["files"].items():
            assert digest(root/name) == row["sha256"], name
        return
    files = {}
    for name in FILES:
        destination = root/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/lcodeca/MoSTScenario/{COMMIT}/{name}"
        if name.endswith("/most.net.xml") and not destination.exists():
            known = Path("data/maps/most.net.xml").resolve()
            if digest(known) != NETWORK_SHA:
                raise ValueError("Existing network is not the pinned native MoST network")
            destination.symlink_to(known)
        elif not destination.exists():
            temporary = destination.with_suffix(destination.suffix+".partial")
            request = urllib.request.Request(url, headers={"User-Agent": "CNR-thesis-source-audit"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
                while chunk := response.read(1024*1024):
                    target.write(chunk)
            temporary.replace(destination)
        files[name] = {"sha256": digest(destination), "bytes": destination.stat().st_size, "url": url}
        print("source", name, files[name]["bytes"], flush=True)
    config = ET.parse(root/"scenario/most.sumocfg").getroot()
    for field in ("net-file", "route-files", "additional-files"):
        for name in config.find(f"input/{field}").get("value").split(","):
            if not (root/"scenario"/name).exists():
                raise FileNotFoundError(name)
    save(root/"complete.json", {"complete": True, "repository": "https://github.com/lcodeca/MoSTScenario",
        "commit": COMMIT, "files": files, "license": "Upstream GPL-3.0; cached only, not republished in this repository",
        "scope": "Same public map, native demand; not a claim of reproducing the CNR historical run"})


def run(task):
    root, source, seed, split, end = task
    output = root/split/str(seed)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"simulation-complete.json").exists():
        return
    sumo = Path("g10-env/sumo/bin/sumo").resolve()
    cmd = [str(sumo), "-c", str((source/"scenario/most.sumocfg").resolve()),
           "--seed", str(seed), "--begin", "14400", "--end", str(end), "--step-length", ".25",
           "--output-prefix", "", "--fcd-output", str((output/"fcd.xml.gz").resolve()),
           "--fcd-output.attributes", "x,y,type,speed,lane,edge,vehicle",
           "--device.fcd.period", ".25", "--person-device.fcd.period", ".25",
           "--person-device.fcd.probability", "1", "--device.fcd.probability", "1",
           "--tripinfo-output", str((output/"tripinfo.xml").resolve()),
           "--tripinfo-output.write-unfinished", "true",
           "--vehroute-output", str((output/"vehroute.xml.gz").resolve()),
           "--vehroute-output.write-unfinished", "true",
           "--stop-output", str((output/"stops.xml").resolve()),
           "--summary-output", str((output/"summary.xml").resolve()),
           "--log", str((output/"sumo.log").resolve()), "--no-step-log", "true"]
    config = {"seed": seed, "split": split, "source_commit": COMMIT, "command": cmd,
              "begin": 14400, "end": end, "native_step_s": .25,
              "observation": "Fixed observation horizon, arrival status recorded separately",
              "randomness": "Different SUMO random realizations of identical public demand, not new OD draws",
              "source_manifest_sha256": digest(source/"complete.json"),
              "version": subprocess.check_output([str(sumo), "--version"], text=True).splitlines()[0]}
    save(output/"config.json", config)
    begin = time.monotonic()
    try:
        with (output/"console.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        names = ["fcd.xml.gz", "tripinfo.xml", "vehroute.xml.gz", "stops.xml", "summary.xml"]
        receipts = {name: {"sha256": digest(output/name), "bytes": (output/name).stat().st_size}
                    for name in names}
        summary = list(ET.parse(output/"summary.xml").getroot())
        logs = (output/"console.log").read_text(errors="replace")
        warnings = {key: sum(key in line.lower() for line in logs.splitlines())
                    for key in ("teleport", "collision", "warning:", "error:", "discard")}
        save(output/"simulation-complete.json", {"complete": True, "seconds": time.monotonic()-begin,
             "seed": seed, "files": receipts, "last_summary": summary[-1].attrib if summary else {},
             "log_keyword_line_counts": warnings, "config_sha256": digest(output/"config.json")})
        print("simulation complete", split, seed, flush=True)
    except Exception as error:
        save(output/"simulation-failed.json", {"complete": False, "error": repr(error), "seconds": time.monotonic()-begin})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fetch", "pilot", "cohort"))
    parser.add_argument("--root", type=Path, default=Path("g30-data/native"))
    parser.add_argument("--source", type=Path, default=Path("g30-data/source"))
    parser.add_argument("--end", type=float, default=19516)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    fetch(args.source)
    if args.action == "pilot":
        run((args.root, args.source, 300001, "pilot", args.end))
    elif args.action == "cohort":
        tasks = [(args.root, args.source, seed, split, args.end)
                 for split, seeds in (("train", range(300101, 300107)), ("val", range(300201, 300204))) for seed in seeds]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(run, tasks))
