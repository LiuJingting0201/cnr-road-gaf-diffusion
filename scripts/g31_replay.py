"""One output-only replay of the completed G30 pilot; never mutate its files.

Run from the new g31-job directory. Only the declared output options may differ
from the pilot's saved command. This script does not fetch data or fit models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

from g30_native import COMMIT, FILES, NETWORK_SHA, save


OUTPUT_FILES = {
    "--fcd-output": "fcd.xml.gz",
    "--tripinfo-output": "tripinfo.xml",
    "--vehroute-output": "vehroute.xml.gz",
    "--stop-output": "stops.xml",
    "--summary-output": "summary.xml",
    "--log": "sumo.log",
}
ENHANCEMENTS = {
    "--vehroute-output.exit-times": "true",
    "--stop-output.write-unfinished": "true",
    "--vehroute-output.stop-edges": "true",
    "--vehroute-output.route-length": "true",
    "--vehroute-output.last-route": "false",
}
OLD_ATTRIBUTES = "x,y,type,speed,lane,edge,vehicle"
NEW_ATTRIBUTES = OLD_ATTRIBUTES + ",pos"
RECEIPT_FILES = tuple(name for name in OUTPUT_FILES.values() if name != "sumo.log")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def options(command):
    if not isinstance(command, list) or not command or len(command) % 2 != 1:
        raise ValueError("Expected the saved executable followed by option/value pairs")
    result = {}
    for flag, value in zip(command[1::2], command[2::2]):
        if not isinstance(flag, str) or not flag.startswith("-") or flag in result:
            raise ValueError("Invalid or duplicate saved SUMO option")
        if not isinstance(value, str):
            raise ValueError("SUMO option values must be strings")
        result[flag] = value
    return result


def validate_pilot(config):
    if (config.get("seed") != 300001 or config.get("split") != "pilot"
            or config.get("begin") != 14400 or config.get("end") != 19516
            or config.get("native_step_s") != .25 or config.get("source_commit") != COMMIT):
        raise ValueError("Only the frozen pilot 300001 / 14400..19516 / 0.25 s is permitted")
    old = options(config["command"])
    expected = {"--seed": 300001., "--begin": 14400., "--end": 19516.,
                "--step-length": .25, "--device.fcd.period": .25,
                "--person-device.fcd.period": .25, "--device.fcd.probability": 1.,
                "--person-device.fcd.probability": 1.}
    if any(float(old.get(flag, "nan")) != value for flag, value in expected.items()):
        raise ValueError("Saved pilot command and its frozen numeric settings disagree")
    if old.get("--fcd-output.attributes") != OLD_ATTRIBUTES:
        raise ValueError("Unexpected original FCD attribute contract")
    for flag in ("--tripinfo-output.write-unfinished", "--vehroute-output.write-unfinished"):
        if old.get(flag) != "true":
            raise ValueError("Original unfinished-trip output must be enabled")
    if not all(flag in old for flag in OUTPUT_FILES) or "-c" not in old:
        raise ValueError("Saved pilot command is missing an expected input/output option")
    if any(flag in old for flag in ENHANCEMENTS):
        raise ValueError("The baseline already contains a G31 enhancement; use the original pilot")
    return old


def verify_receipt(directory):
    directory = Path(directory)
    receipt_path = directory / "simulation-complete.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("complete") is not True or (directory / "simulation-failed.json").exists():
        raise ValueError("Simulation is not a clean completed run")
    if receipt.get("config_sha256") != digest(directory / "config.json"):
        raise ValueError("Simulation config hash does not match its completion receipt")
    files = receipt.get("files", {})
    if set(files) != set(RECEIPT_FILES):
        raise ValueError("Simulation receipt does not cover the expected five output files")
    for name in RECEIPT_FILES:
        path = directory / name
        if path.stat().st_size != files[name]["bytes"] or digest(path) != files[name]["sha256"]:
            raise ValueError(f"Simulation receipt mismatch: {name}")
    return json.loads((directory / "config.json").read_text()), receipt


def build_command(config, output, source, binary):
    old = validate_pilot(config)
    command = [str(Path(binary).resolve())]
    output = Path(output).resolve()
    for flag, value in old.items():
        if flag in OUTPUT_FILES:
            value = str(output / OUTPUT_FILES[flag])
        elif flag == "-c":
            value = str((Path(source) / "scenario/most.sumocfg").resolve())
        elif flag == "--fcd-output.attributes":
            value = NEW_ATTRIBUTES
        command.extend((flag, value))
    for flag, value in ENHANCEMENTS.items():
        command.extend((flag, value))
    allowed = set(OUTPUT_FILES) | set(ENHANCEMENTS) | {"-c", "--fcd-output.attributes"}
    changed = options(command)
    if {k: v for k, v in old.items() if k not in allowed} != {
            k: v for k, v in changed.items() if k not in allowed}:
        raise ValueError("Output-only replay unexpectedly changes a dynamics option")
    return command


def verify_source(source, config):
    source = Path(source)
    manifest = json.loads((source / "complete.json").read_text())
    if (manifest.get("complete") is not True or manifest.get("commit") != COMMIT
            or digest(source / "complete.json") != config["source_manifest_sha256"]):
        raise ValueError("Source manifest differs from the completed pilot")
    if set(manifest.get("files", {})) != set(FILES):
        raise ValueError("Source manifest has an unexpected file inventory")
    for name, item in manifest["files"].items():
        path = source / name
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
    if digest(source / "scenario/in/most.net.xml") != NETWORK_SHA:
        raise ValueError("Native network checksum mismatch")
    original_config = Path(options(config["command"])["-c"])
    if digest(original_config) != digest(source / "scenario/most.sumocfg"):
        raise ValueError("Input configuration differs from the saved pilot command")


def run(args):
    baseline, source, output, binary = [p.resolve() for p in (
        args.baseline, args.source, args.output, args.sumo)]
    for protected in (baseline, source):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Replay output must be separate from the baseline and source trees")
    # No resume or overwrite: an existing run, including a failed one, is evidence.
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    returncode = None
    try:
        config, receipt = verify_receipt(baseline)
        validate_pilot(config)
        verify_source(source, config)
        version = subprocess.check_output([str(binary), "--version"], text=True).splitlines()[0]
        if version != config["version"] or " 1.28.0" not in version:
            raise ValueError("Replay must use the pilot's SUMO 1.28.0 version")
        original_binary = Path(config["command"][0])
        if digest(original_binary) != digest(binary):
            raise ValueError("Replay binary differs from the currently available original binary")
        help_text = subprocess.check_output([str(binary), "--help"], text=True)
        if any(flag not in help_text for flag in ENHANCEMENTS):
            raise ValueError("Installed SUMO does not support all declared output enhancements")
        command = build_command(config, output, source, binary)
        replay_config = {
            "seed": 300001, "split": "pilot", "begin": 14400, "end": 19516,
            "native_step_s": .25, "source_commit": COMMIT, "version": version,
            "command": command, "baseline": str(baseline), "source": str(source),
            "source_manifest_sha256": config["source_manifest_sha256"],
            "baseline_config_sha256": digest(baseline / "config.json"),
            "baseline_receipt_sha256": digest(baseline / "simulation-complete.json"),
            "baseline_command_sha256": json_hash(config["command"]),
            "baseline_files": receipt["files"], "binary_sha256": digest(binary),
            "script_sha256": digest(Path(__file__)),
            "enhancements": ENHANCEMENTS, "fcd_attributes": NEW_ATTRIBUTES,
            "dynamics_command_unchanged": True,
            "scope": "One output-only replay; requires subsequent streaming FCD comparison",
            "binary_history_limit": "Original G30 receipt recorded version, not historical binary hash",
        }
        save(output / "config.json", replay_config)
        with (output / "console.log").open("w") as stream:
            result = subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT)
        returncode = result.returncode
        if returncode:
            raise RuntimeError(f"SUMO exited with code {returncode}; see preserved console.log")
        files = {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                 for name in RECEIPT_FILES}
        final_step = None
        for _, node in ET.iterparse(output / "summary.xml", events=("end",)):
            if node.tag == "step":
                final_step = dict(node.attrib)
            node.clear()
        if final_step is None or abs(float(final_step["time"]) - 19515.75) > 1e-7:
            raise ValueError("Replay did not reach the expected final simulation step")
        warning_counts = dict.fromkeys(("teleport", "collision", "warning:", "error:", "discard"), 0)
        with (output / "console.log").open(errors="replace") as stream:
            for line in stream:
                for key in warning_counts:
                    warning_counts[key] += key in line.lower()
        save(output / "simulation-complete.json", {
            "complete": True, "returncode": returncode, "seed": 300001,
            "seconds": time.monotonic() - started, "files": files,
            "last_summary": final_step, "log_keyword_line_counts": warning_counts,
            "config_sha256": digest(output / "config.json"),
            "comparison_status": "not_run; simulation completion does not imply FCD equality",
        })
        print("G31 output-enhanced pilot completed; FCD comparison still required", flush=True)
    except BaseException as error:
        save(output / "simulation-failed.json", {
            "complete": False, "returncode": returncode,
            "seconds": time.monotonic() - started, "error": repr(error),
        })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("../g30-job/g30-data/native/pilot/300001"))
    parser.add_argument("--source", type=Path, default=Path("../g30-job/g30-data/source"))
    parser.add_argument("--sumo", type=Path, default=Path("../g30-job/g10-env/sumo/bin/sumo"))
    parser.add_argument("--output", type=Path, default=Path("g31-data/replay/pilot/300001"))
    run(parser.parse_args())
