"""Bounded, immutable-attempt G35 runs of the pinned full native MoST demand.

All traffic and person dynamics remain in the source configuration. Person FCD
logging alone is disabled. This is a new realization, not a G31 replay-equality
claim. Run sequentially from the caller; never resume or overwrite an attempt.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

from g30_native import COMMIT, EXPECTED_BYTES, FILES, NETWORK_SHA, save
from g31_replay import ENHANCEMENTS, NEW_ATTRIBUTES, OUTPUT_FILES, digest, options


SCHEMA = "g35-native-full-day-v1"
SOURCE_MANIFEST_SHA = "04223d586569aa4368e6a7e3a0108b28395c951c05da036d73e9ea843bd78d07"
EPISODES = {350101: "train", 350102: "train", 350201: "val"}
BEGIN, END, DT = 14400, 50400, .25
MIN_FREE_BYTES, MAX_NEW_BYTES = 8 * 1024**3, 12 * 1024**3
POLL_SECONDS = 10
RECEIPT_FILES = tuple(OUTPUT_FILES.values()) + ("console.log",)
WARNING_KEYS = ("teleport", "collision", "warning:", "error:", "discard")


def verify_source(source):
    """Verify every cached source against the already archived G30 manifest."""
    source = Path(source)
    manifest_path = source / "complete.json"
    if digest(manifest_path) != SOURCE_MANIFEST_SHA:
        raise ValueError("Source manifest is not the frozen G30 native-demand manifest")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("complete") is not True or manifest.get("commit") != COMMIT
            or set(manifest.get("files", {})) != set(FILES)):
        raise ValueError("Unexpected source status, commit or file inventory")
    for name in FILES:
        path, item = source / name, manifest["files"][name]
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
        if path.name in EXPECTED_BYTES and path.stat().st_size != EXPECTED_BYTES[path.name]:
            raise ValueError(f"Pinned source size mismatch: {name}")
    if digest(source / "scenario/in/most.net.xml") != NETWORK_SHA:
        raise ValueError("Pinned network checksum mismatch")
    return manifest


def build_command(source, binary, output, seed):
    source, binary, output = map(lambda p: Path(p).resolve(), (source, binary, output))
    command = [str(binary), "-c", str(source / "scenario/most.sumocfg"),
        "--seed", str(seed), "--begin", str(BEGIN), "--end", str(END),
        "--step-length", str(DT), "--output-prefix", "",
        "--fcd-output.attributes", NEW_ATTRIBUTES,
        "--device.fcd.period", str(DT), "--person-device.fcd.period", str(DT),
        "--person-device.fcd.probability", "0", "--device.fcd.probability", "1",
        "--tripinfo-output.write-unfinished", "true",
        "--vehroute-output.write-unfinished", "true", "--no-step-log", "true"]
    for flag, name in OUTPUT_FILES.items():
        command.extend((flag, str(output / name)))
    for flag, value in ENHANCEMENTS.items():
        command.extend((flag, value))
    return command


def validate_config(config):
    if (config.get("schema") != SCHEMA or EPISODES.get(config.get("seed")) != config.get("split")
            or config.get("seed") not in EPISODES or config.get("begin") != BEGIN
            or config.get("end") != END or config.get("native_step_s") != DT
            or config.get("source_commit") != COMMIT
            or config.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA):
        raise ValueError("Config does not match the frozen G35 episode contract")
    if not re.search(r"\b1\.28\.0\b", config.get("version", "")):
        raise ValueError("G35 requires SUMO 1.28.0")
    for key in ("binary_sha256", "script_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", config.get(key, "")):
            raise ValueError(f"Missing provenance hash: {key}")
    expected = build_command(config["source"], config["binary"], config["output"], config["seed"])
    if config.get("command") != expected:
        raise ValueError("Saved SUMO command differs from the frozen output-only option contract")
    if config.get("resource_limits") != resource_limits():
        raise ValueError("G35 resource limits changed")


def resource_limits():
    return {"minimum_free_bytes": MIN_FREE_BYTES, "maximum_budget_root_bytes": MAX_NEW_BYTES,
            "check_interval_seconds": POLL_SECONDS,
            "scope": "All regular-file apparent bytes under budget_root, including failed attempts"}


def storage_state(root):
    """Count the entire new G35 tree, refusing symlinks to old/source trees."""
    root = Path(root)
    total = 0
    for directory, folders, names in os.walk(root, followlinks=False):
        for name in folders + names:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"Budget root must not contain symlinks: {path}")
        for name in names:
            path = Path(directory) / name
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                # Workers publish JSON by atomically renaming temporary files.
                continue
    return {"budget_root_bytes": total, "free_bytes": shutil.disk_usage(root).free}


def enforce_storage(state):
    if state["free_bytes"] < MIN_FREE_BYTES:
        raise RuntimeError(f"G35 disk guard: free bytes {state['free_bytes']} < {MIN_FREE_BYTES}")
    if state["budget_root_bytes"] > MAX_NEW_BYTES:
        raise RuntimeError(f"G35 disk guard: new bytes {state['budget_root_bytes']} > {MAX_NEW_BYTES}")


def summary_tail(path):
    """Best-effort progress only; final acceptance uses a full XML parse."""
    if not Path(path).exists():
        return None
    with Path(path).open("rb") as stream:
        stream.seek(max(0, Path(path).stat().st_size - 65536))
        tail = stream.read().decode("utf-8", errors="replace")
    rows = re.findall(r"<step\b[^>]*?/>", tail)
    return dict(ET.fromstring(rows[-1]).attrib) if rows else None


def final_summary(path):
    last = None
    for _, node in ET.iterparse(path, events=("end",)):
        if node.tag == "step":
            last = dict(node.attrib)
        node.clear()
    timestamp = float(last.get("time", "nan")) if last else math.nan
    if not math.isfinite(timestamp) or abs(timestamp - (END - DT)) > 1e-7:
        raise ValueError("SUMO did not reach the expected final step 50399.75")
    return last


def warning_counts(path):
    counts = dict.fromkeys(WARNING_KEYS, 0)
    if Path(path).exists():
        with Path(path).open(errors="replace") as stream:
            for line in stream:
                for key in counts:
                    counts[key] += key in line.lower()
    return counts


def verify_receipt(folder):
    """Return (config, receipt) only for a complete untampered G35 attempt."""
    folder = Path(folder)
    receipt = json.loads((folder / "simulation-complete.json").read_text())
    if (receipt.get("complete") is not True or receipt.get("returncode") != 0
            or (folder / "simulation-failed.json").exists()):
        raise ValueError("Simulation is not a clean completed run")
    if receipt.get("config_sha256") != digest(folder / "config.json"):
        raise ValueError("Simulation configuration hash mismatch")
    config = json.loads((folder / "config.json").read_text())
    validate_config(config)
    if receipt.get("seed") != config["seed"] or receipt.get("split") != config["split"]:
        raise ValueError("Simulation receipt episode mismatch")
    files = receipt.get("files", {})
    if set(files) != set(RECEIPT_FILES):
        raise ValueError("Simulation receipt must cover five XML outputs and both logs")
    for name, item in files.items():
        path = folder / name
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Simulation output hash mismatch: {name}")
    if final_summary(folder / "summary.xml") != receipt.get("last_summary"):
        raise ValueError("Final summary differs from completion receipt")
    if warning_counts(folder / "console.log") != receipt.get("log_keyword_line_counts"):
        raise ValueError("Console warning counts differ from completion receipt")
    return config, receipt


def stop_owned(process):
    """Stop only this runner's direct SUMO child, never other jobs or old trees."""
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run(args):
    source, binary, output, budget = [Path(p).resolve() for p in (
        args.source, args.sumo, args.output, args.budget_root)]
    if EPISODES.get(args.seed) != args.split or args.seed not in EPISODES:
        raise ValueError("Only train 350101/350102 and val 350201 are permitted")
    if output == budget or not output.is_relative_to(budget):
        raise ValueError("Attempt output must be strictly inside the dedicated new budget root")
    if (budget.is_relative_to(source) or source.is_relative_to(budget)
            or binary.is_relative_to(budget)):
        raise ValueError("Dedicated budget root must not contain or overlap source/binary trees")
    # An existing directory is evidence, even if empty or failed. No resume.
    output.mkdir(parents=True, exist_ok=False)
    start, process, returncode, config = time.monotonic(), None, None, None
    previous_handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f"Runner received signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        enforce_storage(storage_state(budget))
        verify_source(source)
        version = subprocess.check_output([str(binary), "--version"], text=True).splitlines()[0]
        if not re.search(r"\b1\.28\.0\b", version):
            raise ValueError("Native G35 runs require SUMO 1.28.0")
        help_text = subprocess.check_output([str(binary), "--help"], text=True)
        if any(flag not in help_text for flag in ENHANCEMENTS):
            raise ValueError("SUMO binary lacks a required G31 output enhancement")
        config = {"schema": SCHEMA, "seed": args.seed, "split": args.split,
            "source": str(source), "binary": str(binary), "output": str(output),
            "budget_root": str(budget), "begin": BEGIN, "end": END, "native_step_s": DT,
            "source_commit": COMMIT, "source_manifest_sha256": digest(source / "complete.json"),
            "binary_sha256": digest(binary), "script_sha256": digest(Path(__file__)),
            "version": version, "command": build_command(source, binary, output, args.seed),
            "resource_limits": resource_limits(),
            "scope": "New full native-demand realization; no G31 FCD-equality claim",
            "randomness": "Fixed public demand, new SUMO random realizations; not new OD templates",
            "person_output": "Person traffic remains simulated; only person FCD probability is zero"}
        validate_config(config)
        save(output / "config.json", config)
        with (output / "console.log").open("x") as stream:
            process = subprocess.Popen(config["command"], cwd=output, stdout=stream, stderr=subprocess.STDOUT)
            while True:
                state = storage_state(budget)
                save(output / "simulation-progress.json", {"status": "running", "complete": False,
                    "pid": process.pid, "runner_pid": os.getpid(), "seed": args.seed, "split": args.split,
                    "seconds": time.monotonic() - start, "last_summary": summary_tail(output / "summary.xml"),
                    "storage": state, "config_sha256": digest(output / "config.json")})
                enforce_storage(state)
                try:
                    returncode = process.wait(timeout=POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    pass
        if returncode != 0:
            raise RuntimeError(f"SUMO exited with code {returncode}; see preserved console.log")
        enforce_storage(storage_state(budget))
        final = final_summary(output / "summary.xml")
        # Detect concurrent source/binary changes, rather than certifying their old hashes.
        verify_source(source)
        if digest(binary) != config["binary_sha256"]:
            raise ValueError("SUMO binary changed during the run")
        files = {name: {"bytes": (output / name).stat().st_size, "sha256": digest(output / name)}
                 for name in RECEIPT_FILES}
        save(output / "simulation-complete.json", {"complete": True, "returncode": returncode,
            "seed": args.seed, "split": args.split, "seconds": time.monotonic() - start,
            "files": files, "last_summary": final, "config_sha256": digest(output / "config.json"),
            "log_keyword_line_counts": warning_counts(output / "console.log"),
            "storage": storage_state(budget), "scope": config["scope"]})
        save(output / "simulation-progress.json", {"status": "completed", "complete": True,
            "seed": args.seed, "split": args.split, "last_summary": final,
            "seconds": time.monotonic() - start})
        print(f"G35 native simulation complete: {args.split} {args.seed}", flush=True)
    except BaseException as error:
        # Restore handlers first so repeated signals cannot interrupt graceful cleanup.
        for signum, handler in previous_handlers.items():
            signal.signal(signum, signal.SIG_IGN)
        stop_owned(process)
        returncode = process.returncode if process is not None else returncode
        failed = {"status": "failed", "complete": False, "returncode": returncode,
            "seed": args.seed, "split": args.split, "seconds": time.monotonic() - start,
            "error": repr(error), "last_summary": summary_tail(output / "summary.xml"),
            "log_keyword_line_counts": warning_counts(output / "console.log")}
        save(output / "simulation-failed.json", failed)
        save(output / "simulation-progress.json", failed)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "sumo", "output", "budget-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=tuple(EPISODES), required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    run(parser.parse_args())
