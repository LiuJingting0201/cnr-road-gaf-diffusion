"""Adopt already running SUMO without restarting it or changing frozen code.

Old Python parents are SIGSTOP'ed, so their SUMO children retain a readable
zombie wait status on natural exit. No SUMO signal is sent in normal operation.
On a handoff failure only still-identified running SUMO children are SIGSTOP'ed
for recovery, never terminated. Linux /proc field 52 supplies the real wait
status; no synthetic exit status is allowed.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import xml.parsers.expat

SCHEMA = "g35c-running-sumo-handoff-v1"
RECEIPT_SCHEMA = "g35c-adopted-sumo-complete-v1"
EPISODES = {350101: "train", 350102: "train", 350201: "val"}
PROC_DOC = "https://www.man7.org/linux/man-pages/man5/proc_pid_stat.5.html"


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def digest(path, pulse=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024**2), b""):
            h.update(chunk)
            if pulse:
                pulse()
    return h.hexdigest()


def parse_proc_stat(raw):
    # comm may itself contain spaces or parentheses; remaining tokens start at field 3.
    tail = raw.rsplit(") ", 1)[1].split()
    if len(tail) < 50:
        raise ValueError("Kernel /proc stat lacks the required exit status field")
    return {"pid": int(raw.split(" (", 1)[0]), "state": tail[0], "ppid": int(tail[1]),
            "pgrp": int(tail[2]), "session": int(tail[3]), "start_ticks": int(tail[19]),
            "wait_status": int(tail[49]), "raw_stat": raw}


def proc_stat(pid):
    return parse_proc_stat(Path("/proc", str(pid), "stat").read_text())


def process_identity(pid):
    """Capture before handoff; callable by the operator's manifest builder."""
    p = Path("/proc", str(pid)); row = proc_stat(pid)
    if row["state"] == "Z":
        raise ValueError("Cannot initially adopt a process whose parent may already reap it")
    cmd = (p/"cmdline").read_bytes()
    return {"pid": int(pid), "ppid": row["ppid"], "start_ticks": row["start_ticks"],
            "exe": os.readlink(p/"exe"), "cwd": os.readlink(p/"cwd"),
            "cmdline_sha256": hashlib.sha256(cmd).hexdigest()}


def check_identity(expected, states=None, zombie_ok=False):
    row = proc_stat(expected["pid"])
    if any(row[k] != expected[k] for k in ("pid", "ppid", "start_ticks")):
        raise ValueError("PID identity/parent/start ticks changed")
    if states is not None and row["state"] not in states:
        raise ValueError("Unexpected process state: "+row["state"])
    if row["state"] == "Z":
        if not zombie_ok:
            raise ValueError("Unexpected zombie")
    elif process_identity(expected["pid"]) != expected:
        raise ValueError("Process executable/cwd/command changed")
    return row


def stop_python(expected):
    check_identity(expected, states={"R", "S", "D", "T"})
    os.kill(expected["pid"], signal.SIGSTOP)
    until = time.monotonic()+10
    while True:
        row = check_identity(expected)
        if row["state"] == "T":
            return row
        if time.monotonic() >= until:
            raise TimeoutError("Python supervisor did not stop")
        time.sleep(.02)


def kill_stopped_python(expected):
    check_identity(expected, states={"T"})
    os.kill(expected["pid"], signal.SIGKILL)


def storage_state(roots):
    total = 0
    for root in roots:
        for directory, folders, names in os.walk(root, followlinks=False):
            for name in folders+names:
                p = Path(directory)/name
                if p.is_symlink():
                    raise ValueError("A budget root contains a symlink")
            for name in names:
                try:
                    s = (Path(directory)/name).stat()
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(s.st_mode):
                    total += s.st_size
    return {"total_budget_bytes": total, "free_bytes": min(shutil.disk_usage(p).free for p in roots)}


def xml_integrity(path, pulse=None):
    """Expat checks full XML; gzip reading to EOF verifies its checksum/footer."""
    parser = xml.parsers.expat.ParserCreate()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b""):
            parser.Parse(chunk, False)
            if pulse:
                pulse()
    parser.Parse(b"", True)


def validate_manifest(manifest, output, roots):
    if manifest.get("schema") != SCHEMA or manifest.get("poll_seconds") != 10:
        raise ValueError("Unexpected frozen adoption contract")
    old, new = Path(manifest["old_job"]), Path(manifest["new_job"])
    if old != old.resolve() or new != new.resolve() or old == new or Path.cwd().resolve() != new:
        raise ValueError("Expected distinct canonical old/new jobs and new-job cwd")
    expected_roots = [old/"g35-data", new/"g35-data"]
    if [Path(x) for x in manifest["budget_roots"]] != expected_roots or roots != expected_roots:
        raise ValueError("Budget must count both old raw outputs and new derived outputs")
    if any(p.is_symlink() for p in roots) or not output.is_relative_to(roots[1]):
        raise ValueError("Adoption output must reside under the new budget root")
    if (manifest.get("minimum_free_bytes", 0) < 8*1024**3
            or not 15*1024**3 < manifest.get("maximum_total_bytes", 0) <= 40*1024**3):
        raise ValueError("Unsafe or missing explicit capacity amendment")
    if {x.get("role") for x in manifest["supervisors"]} != {"data", "training"} or len(manifest["supervisors"]) != 2:
        raise ValueError("Exactly the original two Python supervisors are required")
    if {(x["seed"], x["split"]) for x in manifest["episodes"]} != set(EPISODES.items()) or len(manifest["episodes"]) != 3:
        raise ValueError("Exactly the three frozen SUMO episodes are required")
    pids = []
    for entry in manifest["supervisors"]:
        ident = entry["identity"]; check_identity(ident, states={"R", "S", "D", "T"})
        cmd = Path("/proc", str(ident["pid"]), "cmdline").read_bytes().split(b"\0")
        script = b"scripts/g35b_data_run.py" if entry["role"] == "data" else b"scripts/g35_train_run.py"
        if ident["cwd"] != str(old) or script not in cmd:
            raise ValueError("Wrong old Python supervisor")
        pids.append(ident["pid"])
    data_pid = next(x["identity"]["pid"] for x in manifest["supervisors"] if x["role"] == "data")
    for entry in manifest.get("old_workers", []):
        ident = entry["identity"]; check_identity(ident, states={"R", "S", "D", "T"})
        cmd = Path("/proc", str(ident["pid"]), "cmdline").read_bytes().split(b"\0")
        if (entry.get("role") != "extraction" or ident["cwd"] != str(old)
                or ident["ppid"] != data_pid or b"scripts/g35b_extract.py" not in cmd):
            raise ValueError("Wrong optional old extraction worker")
        pids.append(ident["pid"])
    import g35b_simulate as original
    verified_completed = {}
    for episode in manifest["episodes"]:
        folder = Path(episode["simulation_path"])
        if folder != old/"g35-data/native"/episode["split"]/str(episode["seed"]):
            raise ValueError("Simulation folder differs from declared episode")
        if "original_receipt_sha256" in episode:
            if "runner" in episode or "sumo" in episode:
                raise ValueError("Completed episode cannot also declare live process identities")
            if digest(folder/"simulation-complete.json") != episode["original_receipt_sha256"]:
                raise ValueError("Completed original receipt changed")
            config, original_receipt = original.verify_receipt(folder)
            verified_completed[episode["seed"]] = (config, original_receipt)
            continue
        if (folder/"simulation-failed.json").exists() or (folder/"simulation-complete.json").exists():
            raise ValueError("Only currently incomplete unfailed episodes can be handed off")
        runner, sumo = episode["runner"], episode["sumo"]
        check_identity(runner, states={"R", "S", "D", "T"}); check_identity(sumo, states={"R", "S", "D", "Z"}, zombie_ok=True)
        rcmd = Path("/proc", str(runner["pid"]), "cmdline").read_bytes().split(b"\0")
        config = json.loads((folder/"config.json").read_text())
        original.validate_config(config); original.verify_recording(folder, config)
        if (runner["ppid"] != data_pid or runner["cwd"] != str(old)
                or b"scripts/g35b_simulate.py" not in rcmd or sumo["ppid"] != runner["pid"]
                or sumo["cwd"] != str(folder) or sumo["exe"] != str(Path(config["binary"]).resolve())):
            raise ValueError("SUMO/runner ancestry or executable does not match")
        expected_cmd = b"\0".join(x.encode() for x in config["command"])+b"\0"
        if hashlib.sha256(expected_cmd).hexdigest() != sumo["cmdline_sha256"]:
            raise ValueError("SUMO is not executing the frozen scientific command")
        if digest(config["binary"]) != config["binary_sha256"]:
            raise ValueError("SUMO binary changed before handoff")
        pids.extend((runner["pid"], sumo["pid"]))
    if len(pids) != len(set(pids)) or os.getpid() in pids:
        raise ValueError("Duplicate or self process in handoff manifest")
    # All scenes share one source: full source verification once before signals.
    original.verify_source(Path(config["source"]))
    return verified_completed


def verify_source(source):
    from g35b_simulate import verify_source as original_verify_source
    return original_verify_source(source)


def verify_adopted_receipt(folder):
    """Return original config and audited adoption receipt; never invent exit=0."""
    import g35b_simulate as original
    folder = Path(folder)
    receipt = json.loads((folder/"complete.json").read_text())
    if (receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("complete") is not True
            or receipt.get("returncode") != 0 or (folder/"failed.json").exists()):
        raise ValueError("Unclean adoption completion")
    if digest(folder.parent/"manifest.json") != receipt["manifest_sha256"]:
        raise ValueError("Adoption manifest snapshot changed")
    kind = receipt.get("completion_kind", "kernel_wait_status")
    if kind == "original_runner":
        source = Path(receipt["simulation_path"])
        if digest(source/"simulation-complete.json") != receipt["original_receipt_sha256"]:
            raise ValueError("Original completed receipt changed")
        config, original_receipt = original.verify_receipt(source)
        if (config != receipt["config"] or original_receipt["files"] != receipt["files"]
                or original_receipt["config_sha256"] != receipt["config_sha256"]
                or original_receipt["last_summary"] != receipt["last_summary"]):
            raise ValueError("Imported original receipt fields differ")
        return config, receipt
    if kind != "kernel_wait_status":
        raise ValueError("Unknown completion evidence kind")
    proof = receipt["kernel_exit_proof"]
    if parse_proc_stat(proof["raw_stat"]) != proof:
        raise ValueError("Structured kernel proof differs from the preserved raw /proc stat")
    if (proof["state"] != "Z" or os.waitstatus_to_exitcode(proof["wait_status"]) != receipt["returncode"]
            or proof["pid"] != receipt["sumo_identity"]["pid"]
            or proof["start_ticks"] != receipt["sumo_identity"]["start_ticks"]
            or proof["ppid"] != receipt["sumo_identity"]["ppid"]):
        raise ValueError("Kernel exit status proof does not match adopted process")
    source = Path(receipt["simulation_path"])
    if digest(source/"config.json") != receipt["config_sha256"]:
        raise ValueError("Original config changed")
    config = json.loads((source/"config.json").read_text())
    if config != receipt["config"]:
        raise ValueError("Original config snapshot changed")
    original.validate_config(config); original.verify_recording(source, config)
    if (source/"simulation-failed.json").exists():
        raise ValueError("Original runner recorded a failure")
    if receipt["seed"] != config["seed"] or receipt["split"] != config["split"]:
        raise ValueError("Episode identity mismatch")
    if set(receipt["files"]) != set(original.RECEIPT_FILES):
        raise ValueError("Unexpected output artifact inventory")
    for name, item in receipt["files"].items():
        p = source/name
        if p.is_symlink() or p.stat().st_size != item["bytes"] or digest(p) != item["sha256"]:
            raise ValueError("Adopted output changed: "+name)
    if original.final_summary(source/"summary.xml") != receipt["last_summary"]:
        raise ValueError("Final time/summary mismatch")
    if original.warning_counts(source/"console.log") != receipt["log_keyword_line_counts"]:
        raise ValueError("Console log changed")
    if digest(config["binary"]) != config["binary_sha256"]:
        raise ValueError("SUMO binary changed")
    return config, receipt


def run(args):
    import g35b_simulate as original
    manifest_path = args.manifest.resolve(); manifest = json.loads(manifest_path.read_text())
    output = args.output.resolve(); roots = [x.resolve() for x in args.budget_root]
    if os.getsid(0) != os.getpid():
        raise ValueError("Launch adoption monitor with start_new_session=True")
    # Creating the new budget parent is allowed before validation; no old file is changed.
    roots[1].mkdir(parents=True, exist_ok=True)
    verified_completed = validate_manifest(manifest, output, roots)
    output.mkdir(parents=True, exist_ok=False)
    manifest_hash = digest(manifest_path)
    (output/"manifest.json").write_bytes(manifest_path.read_bytes())
    if digest(output/"manifest.json") != manifest_hash:
        raise ValueError("Manifest changed while saving its snapshot")
    script_hash = digest(Path(__file__))
    stopped, killed, completed, configs = [], [], {}, {}
    start = time.monotonic(); handed_off = False; last_guard = -float("inf")
    limits = {k: manifest[k] for k in ("budget_roots", "minimum_free_bytes", "maximum_total_bytes", "poll_seconds")}
    with (output/"adoption.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def guard(force=False):
            nonlocal last_guard
            if not force and time.monotonic()-last_guard < 10:
                return None
            if digest(manifest_path) != manifest_hash or digest(Path(__file__)) != script_hash:
                raise ValueError("Frozen adoption manifest or source changed")
            state = storage_state(roots)
            if state["free_bytes"] < manifest["minimum_free_bytes"] or state["total_budget_bytes"] > manifest["maximum_total_bytes"]:
                raise RuntimeError("Amended capacity guard: "+json.dumps(state))
            last_guard = time.monotonic()
            return state
        def interruption(signum, frame):
            raise InterruptedError("Adoption monitor signal "+str(signum))
        previous = {s: signal.signal(s, interruption) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            guard(True)
            save(output/"started.json", {"pid": os.getpid(), "started_unix": time.time(), "manifest_sha256": manifest_hash,
                 "script_sha256": script_hash, "resource_amendment": limits,
                 "failure_policy": "SIGSTOP only identity-verified remaining SUMO on fault; leave old Python parents stopped for explicit recovery"})
            # Stop controllers first: they cannot interpret a stopped/replaced runner as a failure.
            handed_off = True
            for entry in manifest["supervisors"]:
                stop_python(entry["identity"]); stopped.append(entry["identity"])
            for entry in manifest.get("old_workers", []):
                stop_python(entry["identity"]); stopped.append(entry["identity"])
                kill_stopped_python(entry["identity"]); killed.append(entry["identity"]["pid"])
            for episode in manifest["episodes"]:
                seed = episode["seed"]; folder = Path(episode["simulation_path"])
                dest = output/str(seed); dest.mkdir()
                if seed in verified_completed:
                    config, old_receipt = verified_completed[seed]
                    if digest(folder/"simulation-complete.json") != episode["original_receipt_sha256"]:
                        raise ValueError("Completed original receipt changed after preflight")
                    imported = {**old_receipt, "schema": RECEIPT_SCHEMA, "completion_kind": "original_runner",
                        "config": config, "simulation_path": str(folder),
                        "original_receipt_sha256": episode["original_receipt_sha256"],
                        "manifest_sha256": manifest_hash, "resource_amendment": limits,
                        "adoption_script_sha256": script_hash,
                        "scope": "Original runner completed before handoff; its true returncode and hashed outputs are preserved"}
                    save(dest/"complete.json", imported)
                    completed[seed] = digest(dest/"complete.json")
                    continue
                stop_python(episode["runner"]); stopped.append(episode["runner"])
                check_identity(episode["sumo"], states={"R", "S", "D", "Z"}, zombie_ok=True)
                configs[seed] = (json.loads((folder/"config.json").read_text()), digest(folder/"config.json"))
            save(output/"handoff.json", {"stopped_python_processes": stopped, "sumo_signals_sent": [],
                 "manifest_sha256": manifest_hash, "unix": time.time()})
            while len(completed) != len(manifest["episodes"]):
                state = guard(True); states = []
                for episode in manifest["episodes"]:
                    seed = episode["seed"]
                    if seed in completed:
                        continue
                    check_identity(episode["runner"], states={"T"})
                    row = check_identity(episode["sumo"], states={"R", "S", "D", "Z"}, zombie_ok=True)
                    source = Path(episode["simulation_path"]); dest = output/str(seed)
                    states.append({"seed": seed, "state": row["state"], "last_summary": original.summary_tail(source/"summary.xml")})
                    if row["state"] != "Z":
                        continue
                    rc = os.waitstatus_to_exitcode(row["wait_status"])
                    save(dest/"kernel-exit.json", {**row, "decoded_returncode": rc, "observed_unix": time.time(), "documentation": PROC_DOC})
                    # The real wait status is now durable. Killing its stopped parent cannot affect an exited SUMO.
                    kill_stopped_python(episode["runner"]); killed.append(episode["runner"]["pid"])
                    if rc != 0:
                        raise RuntimeError("Adopted SUMO returned "+str(rc)+" for "+str(seed))
                    config, config_hash = configs[seed]
                    if digest(source/"config.json") != config_hash:
                        raise ValueError("Original simulation config changed during adoption")
                    original.verify_source(Path(config["source"])); guard(True)
                    if digest(config["binary"], guard) != config["binary_sha256"]:
                        raise ValueError("SUMO binary changed")
                    original.validate_config(config); original.verify_recording(source, config)
                    last = original.final_summary(source/"summary.xml"); guard(True)
                    artifacts = {}
                    for name in original.RECEIPT_FILES:
                        p = source/name
                        if p.is_symlink():
                            raise ValueError("Output artifact is a symlink")
                        if name.endswith((".xml", ".xml.gz", ".sumocfg")):
                            xml_integrity(p, guard)
                        artifacts[name] = {"bytes": p.stat().st_size, "sha256": digest(p, guard)}
                    receipt = {"schema": RECEIPT_SCHEMA, "complete": True, "returncode": rc,
                        "seed": seed, "split": episode["split"], "simulation_path": str(source),
                        "config": config, "config_sha256": config_hash, "files": artifacts,
                        "last_summary": last, "log_keyword_line_counts": original.warning_counts(source/"console.log"),
                        "kernel_exit_proof": row, "sumo_identity": episode["sumo"],
                        "old_runner_identity": episode["runner"], "resource_amendment": limits,
                        "manifest_sha256": manifest_hash, "adoption_script_sha256": script_hash,
                        "seconds_since_adoption": time.monotonic()-start,
                        "scope": "Original running SUMO continued unchanged; actual kernel wait status and complete outputs independently verified"}
                    save(dest/"complete.json", receipt)
                    completed[seed] = digest(dest/"complete.json")
                    print(json.dumps({"episode_complete": seed, "returncode": rc}), flush=True)
                save(output/"progress.json", {"complete": False, "seconds": time.monotonic()-start,
                     "completed_seeds": sorted(completed), "active": states, "storage": state,
                     "killed_stopped_python_pids": killed})
                if len(completed) != len(manifest["episodes"]):
                    time.sleep(manifest["poll_seconds"])
            for entry in manifest["supervisors"]:
                kill_stopped_python(entry["identity"]); killed.append(entry["identity"]["pid"])
            save(output/"complete.json", {"complete": True, "schema": SCHEMA, "episode_receipt_sha256": completed,
                 "manifest_sha256": manifest_hash, "script_sha256": script_hash, "resource_amendment": limits,
                 "killed_stopped_python_pids": killed, "seconds": time.monotonic()-start, "storage": guard(True)})
        except BaseException as error:
            for s in previous:
                signal.signal(s, signal.SIG_IGN)
            frozen, freeze_errors = [], []
            if handed_off:
                for episode in manifest["episodes"]:
                    if episode["seed"] in completed or "sumo" not in episode:
                        continue
                    try:
                        row = check_identity(episode["sumo"], zombie_ok=True)
                        if row["state"] != "Z":
                            os.kill(row["pid"], signal.SIGSTOP)
                            frozen.append(row["pid"])
                    except (OSError, ValueError) as e:
                        freeze_errors.append({"seed": episode["seed"], "error": repr(e)})
            save(output/"failed.json", {"complete": False, "error": repr(error), "seconds": time.monotonic()-start,
                 "manifest_sha256": manifest_hash, "completed_seeds": sorted(completed),
                 "stopped_python_identities": stopped, "killed_stopped_python_pids": killed,
                 "sumo_SIGSTOP_sent_on_failure": frozen, "freeze_errors": freeze_errors,
                 "recovery": "Do not resume old 15 GiB guards blindly; inspect frozen identity-checked processes and preserved output before explicit recovery"})
            raise
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)


def self_test():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Run the process handoff self-test on Linux")
    checks = []
    with tempfile.TemporaryDirectory(prefix="g35c-adoption-check-") as temporary:
        root = Path(temporary)
        for code in (0, 7):
            folder = root/str(code); folder.mkdir()
            child_code = "import os,time;from pathlib import Path;p=Path("+repr(str(folder/"release"))+");\nwhile not p.exists():time.sleep(.01)\nos._exit("+str(code)+")"
            parent_code = "import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c',"+repr(child_code)+"]);Path("+repr(str(folder/"child"))+").write_text(str(p.pid));p.wait()"
            parent = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
            ident = child = None
            try:
                until = time.monotonic()+10
                while not (folder/"child").exists():
                    if time.monotonic()>until:raise TimeoutError("Mock parent did not create child")
                    time.sleep(.01)
                ident = process_identity(parent.pid); child = process_identity(int((folder/"child").read_text()))
                if child["ppid"] != parent.pid:raise AssertionError("Wrong mock ancestry")
                stop_python(ident)
                stop_python(ident)
                checks.append("pre_frozen_python_accepted_"+str(code))
                (folder/"release").touch()
                until = time.monotonic()+10
                while True:
                    observed = check_identity(child, zombie_ok=True)
                    if observed["state"] == "Z":break
                    if time.monotonic()>until:raise TimeoutError("Mock child never became zombie")
                    time.sleep(.01)
                assert os.waitstatus_to_exitcode(observed["wait_status"]) == code
                assert check_identity(ident)["state"] == "T"
                checks.append("stopped_parent_retains_true_child_exit_"+str(code))
                altered = dict(child, start_ticks=child["start_ticks"]+1)
                try:check_identity(altered, zombie_ok=True)
                except ValueError:pass
                else:raise AssertionError("Reused PID identity accepted")
                checks.append("wrong_start_ticks_rejected_"+str(code))
                kill_stopped_python(ident); parent.wait(timeout=5)
            finally:
                if parent.poll() is None:
                    if child is not None:
                        try:
                            row=check_identity(child,zombie_ok=True)
                            if row["state"]!="Z":os.kill(child["pid"],signal.SIGKILL)
                        except (OSError,ValueError):pass
                    parent.kill();parent.wait(timeout=5)
        good=root/"good.xml.gz"
        with gzip.open(good,"wb") as f:f.write(b'<root><item/></root>')
        xml_integrity(good);checks.append("complete_gzip_xml_accepted")
        bad=root/"bad.xml.gz";bad.write_bytes(good.read_bytes()[:-4])
        try:xml_integrity(bad)
        except (EOFError,OSError):checks.append("truncated_gzip_rejected")
        else:raise AssertionError("Truncated gzip accepted")
        from unittest.mock import patch
        import g35b_simulate as original
        adopted=root/"adopted";dest=adopted/"350201";dest.mkdir(parents=True)
        source=root/"source-completed";source.mkdir()
        (adopted/"manifest.json").write_text('{"fixture":true}')
        config={"seed":350201,"split":"val"}
        old_receipt={"complete":True,"returncode":0,"files":{},"config_sha256":"fixture","last_summary":{"time":"50399.75"}}
        save(source/"simulation-complete.json",old_receipt)
        imported={**old_receipt,"schema":RECEIPT_SCHEMA,"completion_kind":"original_runner",
                  "simulation_path":str(source),"config":config,
                  "manifest_sha256":digest(adopted/"manifest.json"),
                  "original_receipt_sha256":digest(source/"simulation-complete.json")}
        save(dest/"complete.json",imported)
        with patch.object(original,"verify_receipt",return_value=(config,old_receipt)):
            assert verify_adopted_receipt(dest)==(config,imported)
            checks.append("previously_completed_original_receipt_accepted")
            (source/"simulation-complete.json").write_text('{}')
            try:verify_adopted_receipt(dest)
            except ValueError:checks.append("changed_original_completion_rejected")
            else:raise AssertionError("Changed original receipt accepted")
        badxml=root/"bad.xml";badxml.write_text('<root>')
        try:xml_integrity(badxml)
        except xml.parsers.expat.ExpatError:checks.append("partial_xml_rejected")
        else:raise AssertionError("Partial XML accepted")
    print(json.dumps({"complete":True,"checks":checks,"checks_passed":len(checks),"production_processes_touched":False}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--budget-root",type=Path,action="append")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        if not args.manifest or not args.output or not args.budget_root or len(args.budget_root)!=2:
            parser.error("--manifest, --output and exactly two --budget-root values are required")
        run(args)
