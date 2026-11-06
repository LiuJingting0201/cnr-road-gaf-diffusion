"""Bounded, immutable-attempt G35b output-filtered runs of the pinned full native MoST demand.

All traffic and person settings remain in the source configuration. FCD records
only a source-defined, preassigned identity cohort. A caller may run the three
seeds concurrently with a shared budget root. Recording equivalence must be
checked separately; never resume or overwrite an attempt.
"""
import argparse
from collections import Counter
import copy
from decimal import Decimal, ROUND_CEILING
import hashlib
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
from g31_replay import ENHANCEMENTS, OUTPUT_FILES, digest, options
from g30_demand_audit import children


SCHEMA = "g35-native-full-day-v2"
NEW_ATTRIBUTES = "x,y,type,lane"
ID_FILE = "fcd-ids.txt"
OVERRIDE_FILE = "native-recording.sumocfg"
INPUT_ARTIFACTS = (ID_FILE, OVERRIDE_FILE, "sumo-parsed.sumocfg", "config-parse.log")
FAMILIES = {"ptw", "passenger", "bicycle", "on-demand", "bus", "highway"}
SOURCE_DOCS = {
    "explicit_fcd": "https://sumo.dlr.de/docs/Simulation/Output/FCDOutput.html",
    "configuration": "https://sumo.dlr.de/docs/Basics/Using_the_Command_Line_Applications.html",
    "flow_ids": "https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html",
    "device_assignment_1_28_0": "https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_28_0/src/microsim/devices/MSDevice.h",
    "flow_index_1_28_0": "https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_28_0/src/microsim/MSInsertionControl.cpp"}
SOURCE_MANIFEST_SHA = "04223d586569aa4368e6a7e3a0108b28395c951c05da036d73e9ea843bd78d07"
EPISODES = {350101: "train", 350102: "train", 350201: "val"}
BEGIN, END, DT = 14400, 50400, .25
MIN_FREE_BYTES, MAX_NEW_BYTES = 8 * 1024**3, 15 * 1024**3
POLL_SECONDS = 10
RECEIPT_FILES = tuple(OUTPUT_FILES.values()) + ("console.log",) + INPUT_ARTIFACTS
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


def entity_partition(ident, original):
    """Byte-identical stdlib form of G30 identity partition; no label input."""
    if ident in original:
        return original[ident]
    raw = json.dumps([30042, "entity-partition", ident], ensure_ascii=True, separators=(",", ":")).encode()
    return "val" if int.from_bytes(hashlib.sha256(raw).digest(), "big") < 2**256 // 5 else "train"


def assigned_train_seed(ident):
    value = hashlib.sha256(("g35-realization-v1:" + ident).encode()).digest()
    return (350101, 350102)[int.from_bytes(value, "big") % 2]


def flow_ids(attributes):
    """Only pinned deterministic-period native flows; IDs are zero-based."""
    if (any(name in attributes for name in ("probability", "number", "vehsPerHour"))
            or not all(name in attributes for name in ("id", "begin", "end", "period"))):
        raise ValueError("Recording enumeration requires a deterministic explicit-period flow")
    begin, end, period = [Decimal(attributes[key]) for key in ("begin", "end", "period")]
    if (not all(x.is_finite() for x in (begin, end, period)) or begin < BEGIN or end <= begin or period <= 0
            or period % Decimal(str(DT)) != 0):
        raise ValueError("Unsupported native flow timing; never guess realized flow IDs")
    horizon = min(end, Decimal(END))
    count = max(0, int(((horizon - begin) / period).to_integral_value(rounding=ROUND_CEILING)))
    return [f"{attributes['id']}.{i}" for i in range(count)]


def config_options(root):
    values = {}
    for node in root.iter():
        if "value" in node.attrib:
            if node.tag in values:
                raise ValueError("Duplicate SUMO option in XML configuration: " + node.tag)
            values[node.tag] = node.get("value")
    return values


def source_input_paths(source):
    source = Path(source).resolve()
    root = ET.parse(source / "scenario/most.sumocfg").getroot()
    result = {}
    for key in ("net-file", "route-files", "additional-files"):
        item = root.find("input/" + key)
        if item is None:
            raise ValueError("Pinned source input missing: " + key)
        names = item.get("value").split(",")
        for name in names:
            if "scenario/" + name not in FILES:
                raise ValueError("Source references an unmanifested input: " + name)
        result[key] = [(source / "scenario" / name).resolve() for name in names]
    return root, result


def plan_recording(source, split_path, seed, split):
    if EPISODES.get(seed) != split:
        raise ValueError("Unregistered G35 episode")
    original = json.loads(Path(split_path).read_text())["split_by_id"]
    if (not isinstance(original, dict) or not original or any(not isinstance(k, str) or not k for k in original)
            or not set(original.values()).issubset({"train", "val", "test"})):
        raise ValueError("Expected CNR identity/split metadata only")
    _, inputs = source_input_paths(source)
    selected, all_ids = [], set()
    family_candidates, family_selected, excluded, ignored = Counter(), Counter(), Counter(), Counter()
    for path in inputs["additional-files"] + inputs["route-files"]:
        for node in children(path):
            if any("fcd" in p.get("key", "").lower() for p in node.iter("param")):
                raise ValueError("Source FCD parameter could override the explicit recording contract")
            if node.tag not in ("vehicle", "flow"):
                continue
            family = node.get("type", "")
            if family not in FAMILIES:
                if family not in {"commercial", "special", "train"}:
                    raise ValueError("Unknown source type family requires a source-only taxonomy audit: " + family)
                ignored[family] += 1
                continue
            candidates = [node.get("id", "")] if node.tag == "vehicle" else flow_ids(node.attrib)
            for ident in candidates:
                if not ident or any(x.isspace() for x in ident) or "," in ident or ident in all_ids:
                    raise ValueError("Invalid, ambiguous or duplicate public vehicle identity")
                all_ids.add(ident)
                family_candidates[family] += 1
                partition = entity_partition(ident, original)
                if partition == "test":
                    excluded["original_test_parent"] += 1
                elif partition != split:
                    excluded["other_partition"] += 1
                elif split == "train" and assigned_train_seed(ident) != seed:
                    excluded["other_preassigned_train_realization"] += 1
                else:
                    selected.append(ident)
                    family_selected[family] += 1
    if not selected:
        raise ValueError("No predeclared public identities available for recording")
    return sorted(selected), {
        "selection": "Public input vehicle IDs plus fixed-period flowId.zeroBasedIndex; inherited CNR split or G30 identity hash, then fixed G35 train-seed hash, before any realized type/completion",
        "type_scope": "All potential five-class families including whole ptw and highway distributions; realized unsupported types remain for the unchanged extraction taxonomy to reject",
        "candidate_family_counts": dict(family_candidates), "selected_family_counts": dict(family_selected),
        "excluded_identity_counts": dict(excluded), "unsupported_source_definitions_not_recorded": dict(ignored),
        "source_only_identity_count": len(all_ids), "no_realized_type_duration_or_score_input": True}


def build_override(source, output, ids):
    root, paths = source_input_paths(source)
    before = config_options(root)
    changed = copy.deepcopy(root)
    for key, values in paths.items():
        changed.find("input/" + key).set("value", ",".join(map(str, values)))
    if any(name.startswith("device.fcd.") or name.startswith("person-device.fcd.") for name in before):
        raise ValueError("Unexpected FCD setting in the pinned source configuration")
    device = ET.SubElement(changed, "fcd_device")
    ET.SubElement(device, "device.fcd.explicit", value=",".join(ids))
    after = config_options(changed)
    allowed = {"net-file", "route-files", "additional-files", "device.fcd.explicit"}
    if {k: v for k, v in before.items() if k not in allowed} != {k: v for k, v in after.items() if k not in allowed}:
        raise ValueError("Native dynamics changed while constructing the recording override")
    ET.ElementTree(changed).write(Path(output) / OVERRIDE_FILE, encoding="utf-8", xml_declaration=True)


def read_recorded_ids(folder, config):
    path = Path(folder) / ID_FILE
    ids = path.read_text(encoding="utf-8").splitlines()
    if (not ids or ids != sorted(set(ids)) or any(not x or "," in x or any(c.isspace() for c in x) for x in ids)
            or len(ids) != config["recording"]["vehicle_ids_count"]
            or digest(path) != config["recording"]["vehicle_ids_sha256"]):
        raise ValueError("Private recording identity list changed")
    return ids


def verify_recording(folder, config):
    folder = Path(folder)
    ids = read_recorded_ids(folder, config)
    for name in INPUT_ARTIFACTS:
        path, spec = folder / name, config["input_artifacts"][name]
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError("Recording input artifact changed: " + name)
    source = Path(config["source"])
    if digest(source / "scenario/most.sumocfg") != config["baseline_source_config_sha256"]:
        raise ValueError("Baseline public configuration changed")
    before_root, input_paths = source_input_paths(source)
    before, actual = config_options(before_root), config_options(ET.parse(folder / OVERRIDE_FILE).getroot())
    expected = dict(before)
    expected.update({key: ",".join(map(str, values)) for key, values in input_paths.items()})
    expected["device.fcd.explicit"] = ",".join(ids)
    if actual != expected:
        raise ValueError("Recording override changes an undeclared source option")


def validate_binary_config(binary, command, output, ids):
    """SUMO --save-configuration parses then exits; it does not simulate."""
    output = Path(output)
    command = list(command)
    command[command.index("--log") + 1] = str(output / "config-parser-sumo.log")
    command += ["--save-configuration", str(output / "sumo-parsed.sumocfg"),
                "--save-configuration.relative", "false"]
    with (output / "config-parse.log").open("x") as stream:
        subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60)
    parsed = config_options(ET.parse(output / "sumo-parsed.sumocfg").getroot())
    if parsed.get("device.fcd.explicit", "").split(",") != ids:
        raise ValueError("SUMO did not parse the exact explicit vehicle identity list")
    for key, value in {"device.fcd.probability": 0., "person-device.fcd.probability": 0.,
                       "device.fcd.period": DT, "step-length": DT, "begin": BEGIN, "end": END}.items():
        if float(parsed.get(key, "nan")) != value:
            raise ValueError("SUMO parsed a different recording/time option: " + key)
    if parsed.get("fcd-output.attributes") != NEW_ATTRIBUTES:
        raise ValueError("SUMO parsed different reduced FCD attributes")


def build_command(source, binary, output, seed):
    source, binary, output = map(lambda p: Path(p).resolve(), (source, binary, output))
    command = [str(binary), "-c", str(output / OVERRIDE_FILE),
        "--seed", str(seed), "--begin", str(BEGIN), "--end", str(END),
        "--step-length", str(DT), "--output-prefix", "",
        "--fcd-output.attributes", NEW_ATTRIBUTES,
        "--device.fcd.period", str(DT), "--person-device.fcd.period", str(DT),
        "--person-device.fcd.probability", "0", "--device.fcd.probability", "0",
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
    rec = config.get("recording", {})
    if (rec.get("vehicle_ids_file") != ID_FILE or rec.get("attributes") != NEW_ATTRIBUTES
            or not isinstance(rec.get("vehicle_ids_count"), int) or rec["vehicle_ids_count"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", rec.get("vehicle_ids_sha256", ""))
            or set(config.get("input_artifacts", {})) != set(INPUT_ARTIFACTS)):
        raise ValueError("Missing preselected recording contract")


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
                # Parallel peers atomically rename their temporary JSON files.
                # A file that disappeared contributes no currently allocated bytes.
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
        raise ValueError("Simulation receipt must cover outputs, logs and recording-input artifacts")
    for name, item in files.items():
        path = folder / name
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Simulation output hash mismatch: {name}")
    if final_summary(folder / "summary.xml") != receipt.get("last_summary"):
        raise ValueError("Final summary differs from completion receipt")
    if warning_counts(folder / "console.log") != receipt.get("log_keyword_line_counts"):
        raise ValueError("Console warning counts differ from completion receipt")
    verify_recording(folder, config)
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
        ids, selection = plan_recording(source, args.cnr_splits, args.seed, args.split)
        (output / ID_FILE).write_text("\n".join(ids) + "\n", encoding="utf-8")
        build_override(source, output, ids)
        help_text = subprocess.check_output([str(binary), "--help"], text=True)
        if any(flag not in help_text for flag in (*ENHANCEMENTS, "--device.fcd.explicit", "--save-configuration")):
            raise ValueError("SUMO binary lacks a required G31 output enhancement")
        command = build_command(source, binary, output, args.seed)
        validate_binary_config(binary, command, output, ids)
        input_artifacts = {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                           for name in INPUT_ARTIFACTS}
        config = {"schema": SCHEMA, "seed": args.seed, "split": args.split,
            "source": str(source), "binary": str(binary), "output": str(output),
            "budget_root": str(budget), "begin": BEGIN, "end": END, "native_step_s": DT,
            "source_commit": COMMIT, "source_manifest_sha256": digest(source / "complete.json"),
            "binary_sha256": digest(binary), "script_sha256": digest(Path(__file__)),
            "version": version, "command": build_command(source, binary, output, args.seed),
            "resource_limits": resource_limits(),
            "scope": "New full native-demand realization; no G31 FCD-equality claim",
            "randomness": "Fixed public demand, new SUMO random realizations; not new OD templates",
            "person_output": "All native traffic remains simulated; person FCD is zero, vehicle FCD is a predeclared explicit identity set",
            "recording": {"vehicle_ids_file": ID_FILE, "vehicle_ids_sha256": digest(output / ID_FILE),
                "vehicle_ids_count": len(ids), "attributes": NEW_ATTRIBUTES, **selection},
            "input_artifacts": input_artifacts,
            "baseline_source_config_sha256": digest(source / "scenario/most.sumocfg"),
            "original_split_manifest_sha256": digest(args.cnr_splits),
            "source_docs": SOURCE_DOCS, "recording_equivalence": "Unverified; separate fixed-prefix comparison required"}
        validate_config(config)
        save(output / "config.json", config)
        verify_recording(output, config)
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
        verify_recording(output, config)
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
    for name in ("source", "sumo", "output", "budget-root", "cnr-splits"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=tuple(EPISODES), required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    run(parser.parse_args())
