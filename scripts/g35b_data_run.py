"""Three bounded parallel G35b branches: simulation -> extraction -> one merge.

Run only in the new isolated g35b-job. The output contract remains compatible
with the unchanged G35 training queue. No runtime source or old attempt is edited.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time


ROOT = Path("g35-data")
MIN_FREE = 8 * 1024**3
MAX_NEW = 15 * 1024**3
POLL_SECONDS = 10
EPISODES = ((350101, "train"), (350102, "train"), (350201, "val"))
CODE_NAMES = ("g35b_data_run.py", "g35b_simulate.py", "g35b_extract.py", "g35_merge.py",
              "g35_simulate.py", "g35_extract.py", "g31_legs.py", "g31_replay.py",
              "g30_native.py", "g30_extract.py", "g30_prepare.py", "g30_demand_audit.py",
              "g29_prepare.py", "g33_prepare.py")


def digest(path, cancel=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Receipt validation cancelled after queue failure")
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def storage_state(root):
    total = 0
    for directory, folders, names in os.walk(root, followlinks=False):
        for name in folders+names:
            path = Path(directory)/name
            if path.is_symlink():
                raise ValueError("Budget root contains a symlink")
        for name in names:
            path = Path(directory)/name
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                # Child atomic JSON publication may rename a temporary file.
                continue
    return {"budget_root_bytes": total, "free_bytes": shutil.disk_usage(root).free}


def guard(root):
    state = storage_state(root)
    if state["free_bytes"] < MIN_FREE or state["budget_root_bytes"] > MAX_NEW:
        raise RuntimeError("G35b whole-queue storage guard: "+json.dumps(state))
    return state


def environment():
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               PYTHONPATH="src:g10-env:scripts:../scripts:../vendor/original_cnr/src")
    return env


def interrupted(signum, _frame):
    raise InterruptedError(f"G35b queue received signal {signum}")


@dataclass(frozen=True)
class Stage:
    name: str
    command: list
    folder: Path
    kind: str
    seed: int = 0
    split: str = ""

    @property
    def marker(self):
        return self.folder/("simulation-complete.json" if self.kind == "simulate" else "complete.json")


def verify_marker(stage, cancel=None):
    """Hash large artifacts in a worker, leaving the storage/cancellation loop live."""
    before = digest(stage.marker, cancel)
    receipt = json.loads(stage.marker.read_text())
    if receipt.get("complete") is not True or any((stage.folder/name).exists() for name in
                                                 ("simulation-failed.json", "failed.json")):
        raise ValueError("Missing or unclean completion: "+stage.name)
    if stage.kind == "simulate":
        if receipt.get("seed") != stage.seed or receipt.get("split") != stage.split or receipt.get("returncode") != 0:
            raise ValueError("Simulation completion identity mismatch: "+stage.name)
        config_path = stage.folder/"config.json"
        if receipt.get("config_sha256") != digest(config_path, cancel):
            raise ValueError("Simulation configuration changed")
        config = json.loads(config_path.read_text())
        if (config.get("schema") != "g35-native-full-day-v2" or config.get("seed") != stage.seed
                or config.get("split") != stage.split):
            raise ValueError("Expected the isolated G35b recording contract")
        artifacts = receipt.get("files")
    else:
        if receipt.get("schema") != "g33-direct-sumo-full-legs-v1":
            raise ValueError("Unexpected prepared numeric schema")
        if stage.kind == "extract":
            if (receipt.get("episode") != stage.seed or receipt.get("split") != stage.split
                    or receipt.get("provenance_schema") != "g35-native-full-day-extraction-v1"):
                raise ValueError("Extraction completion identity mismatch")
        elif (receipt.get("provenance_schema") != "g35-full-day-merged-v1"
              or not isinstance(receipt.get("training_ready"), bool)
              or not all(k in receipt.get("groups", {}) for k in ("train", "val", "all"))):
            raise ValueError("Merged readiness contract is missing")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict) or not {"dataset.npz", "leg-metadata.json.gz"} <= set(artifacts):
            raise ValueError("Prepared data artifacts are missing")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Completion has no artifact hash inventory")
    for name, item in artifacts.items():
        path = stage.folder/name
        if not path.resolve().is_relative_to(stage.folder.resolve()) or path.is_symlink():
            raise ValueError("Artifact escapes its isolated stage directory")
        if (not isinstance(item, dict) or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))
                or path.stat().st_size != item.get("bytes") or digest(path, cancel) != item["sha256"]):
            raise ValueError("Artifact changed: "+stage.name+"/"+name)
    if before != digest(stage.marker, cancel):
        raise ValueError("Completion marker changed during validation")
    return receipt, before


def stop_groups(processes, grace=30.):
    """Only groups created with start_new_session by this queue are signalled."""
    groups = {p.pid: p for p in processes}
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic()+grace
    while time.monotonic() < deadline:
        alive = False
        for pgid, process in groups.items():
            process.poll()
            try:
                os.killpg(pgid, 0)
                alive = True
            except ProcessLookupError:
                pass
        if not alive:
            break
        time.sleep(.05)
    for pgid, process in groups.items():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def execute(root, branches, merger, code_paths, *, poll_seconds=POLL_SECONDS,
            guard_fn=guard, child_env=None, signal_handlers=True, stop_grace=30.):
    """Finite orchestration shared by the real queue and fast mock-child checks."""
    root.mkdir(parents=True, exist_ok=True)
    start, active, completed, receipts, streams = time.monotonic(), {}, [], {}, []
    previous_handlers, cancel = {}, threading.Event()
    with (root/"data.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/"data-started.json").exists() or (root/"data-complete.json").exists() or (root/"data-failed.json").exists():
            raise FileExistsError("Existing data queue preserved; choose a new isolated attempt")
        if any(stage.folder.exists() for pair in branches for stage in pair) or merger.folder.exists():
            raise FileExistsError("An output stage already exists; do not reuse an attempt")
        save(root/"data-started.json", {"pid": os.getpid(), "started_unix": time.time(),
             "status": "initializing", "seeds": [pair[0].seed for pair in branches],
             "parallel_branches": len(branches), "training_started": False})
        pool = ThreadPoolExecutor(max_workers=len(branches))
        hashes = {}
        try:
            hashes = {str(path): digest(path) for path in code_paths}
            save(root/"data-started.json", {"pid": os.getpid(), "started_unix": time.time(),
                 "status": "started", "seeds": [pair[0].seed for pair in branches],
                 "parallel_branches": len(branches), "code_sha256": hashes,
                 "resource_limits": {"minimum_free_bytes": MIN_FREE, "maximum_budget_root_bytes": MAX_NEW,
                                     "check_interval_seconds": poll_seconds}, "training_started": False})
            if signal_handlers:
                for sig in (signal.SIGTERM, signal.SIGINT):
                    previous_handlers[sig] = signal.signal(sig, interrupted)

            def check_code():
                if hashes != {str(path): digest(path) for path in code_paths}:
                    raise ValueError("Frozen G35b runtime code changed")

            def launch(stage):
                check_code()
                stream = (root/(stage.name+".log")).open("x")
                streams.append(stream)
                process = subprocess.Popen(stage.command, stdout=stream, stderr=subprocess.STDOUT,
                                           env=child_env, start_new_session=True)
                active[stage.name] = {"stage": stage, "process": process, "future": None}
                print(json.dumps({"stage_started": stage.name, "pid": process.pid,
                                  "pgid": process.pid, "seconds": time.monotonic()-start}), flush=True)

            state = guard_fn(root)
            # Dispatch all simulations before polling any branch. No simulation
            # waits for another branch's extraction or completion.
            for simulation, _ in branches:
                launch(simulation)
            extraction_for = {simulation.name: extraction for simulation, extraction in branches}
            extraction_names = {extraction.name for _, extraction in branches}
            merge_started = False
            while active:
                state = guard_fn(root)
                check_code()
                # Observe every exited failure before launching a dependent stage.
                for entry in active.values():
                    code = entry["process"].poll()
                    if code is not None and code != 0:
                        raise RuntimeError(f"{entry['stage'].name} exited {code}; preserve its log")
                ready = []
                for name, entry in active.items():
                    if entry["process"].returncode is not None and entry["future"] is None:
                        entry["future"] = pool.submit(verify_marker, entry["stage"], cancel)
                    if entry["future"] is not None and entry["future"].done():
                        ready.append((name, entry["future"].result()))
                for name, (receipt, marker_hash) in ready:
                    del active[name]
                    completed.append(name)
                    receipts[name] = {"marker_sha256": marker_hash}
                    print(json.dumps({"stage_completed": name, "seconds": time.monotonic()-start}), flush=True)
                    if name in extraction_for:
                        launch(extraction_for[name])
                    if name == merger.name:
                        merged = receipt
                if not merge_started and extraction_names <= set(completed):
                    launch(merger)
                    merge_started = True
                save(root/"data-progress.json", {"status": "running" if active else "finalizing",
                     "active": {name: {"pid": entry["process"].pid, "pgid": entry["process"].pid,
                         "phase": "validating_receipt" if entry["future"] is not None else "running"}
                         for name, entry in active.items()}, "completed_stages": completed,
                     "elapsed_s": time.monotonic()-start, "storage": state})
                if active:
                    cancel.wait(poll_seconds)
            expected = {stage.name for pair in branches for stage in pair} | {merger.name}
            if set(completed) != expected:
                raise ValueError("Finite queue did not complete all seven stages")
            check_code()
            state = guard_fn(root)
            result = {"complete": True, "completed_stages": completed, "seconds": time.monotonic()-start,
                      "training_ready": merged["training_ready"], "groups": merged["groups"],
                      "merged_receipt_sha256": digest(merger.marker), "training_completed": False,
                      "code_sha256": hashes, "stage_receipts": receipts, "storage": state,
                      "scope": "Three parallel source simulations with per-branch extraction; no model fits in this queue"}
            if result["merged_receipt_sha256"] != receipts[merger.name]["marker_sha256"]:
                raise ValueError("Merged receipt changed after validation")
            save(root/"data-complete.json", result)
            return result
        except BaseException as error:
            cancel.set()
            if signal_handlers:
                for sig in previous_handlers:
                    signal.signal(sig, signal.SIG_IGN)
            stopped = {name: entry["process"].pid for name, entry in active.items()}
            stop_groups([entry["process"] for entry in active.values()], grace=stop_grace)
            save(root/"data-failed.json", {"complete": False, "error": repr(error),
                 "completed_stages": completed, "stopped_owned_process_groups": stopped,
                 "seconds": time.monotonic()-start, "code_sha256": hashes, "training_started": False})
            raise
        finally:
            cancel.set()
            pool.shutdown(wait=True)
            for stream in streams:
                stream.close()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def build_stages(args):
    common, branches = [sys.executable, "-u"], []
    for seed, split in EPISODES:
        simulation = args.root/"native"/split/str(seed)
        extraction = args.root/"extracted"/str(seed)
        branches.append((Stage(f"simulate-{seed}", common+[str(args.scripts/"g35b_simulate.py"),
            "--source", str(args.source), "--sumo", str(args.sumo), "--output", str(simulation),
            "--seed", str(seed), "--split", split, "--budget-root", str(args.root),
            "--cnr-splits", str(args.cnr_splits)], simulation, "simulate", seed, split),
            Stage(f"extract-{seed}", common+[str(args.scripts/"g35b_extract.py"),
            "--source", str(args.source), "--simulation", str(simulation),
            "--cnr-splits", str(args.cnr_splits), "--output", str(extraction)],
                  extraction, "extract", seed, split)))
    merger = Stage("merge", common+[str(args.scripts/"g35_merge.py"), "--inputs",
                   *[str(args.root/"extracted"/str(seed)) for seed, _ in EPISODES],
                   "--output", str(args.root/"direct-legs")], args.root/"direct-legs", "merge")
    return branches, merger


MOCK_CHILD = r'''
import hashlib,json,os,signal,sys,time
from pathlib import Path
root,folder,kind,seed,split,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]),sys.argv[5],sys.argv[6]
folder.mkdir(parents=True)
def log(event):
 fd=os.open(root/'events.jsonl',os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
 os.write(fd,(json.dumps({'event':event,'kind':kind,'seed':seed,'pid':os.getpid(),'time':time.time()})+'\n').encode());os.close(fd)
def stop(signum,frame):log('terminated');raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,stop)
log('started')
if kind=='simulate':
 deadline=time.monotonic()+5
 while len([p for p in (root/'native').glob('*/*') if p.is_dir()])<3:
  if time.monotonic()>deadline:raise RuntimeError('Three branches were not dispatched')
  time.sleep(.01)
 if mode=='failure' and seed==350101:time.sleep(.12);raise SystemExit(7)
 time.sleep(5 if mode=='failure' else (.1 if seed==350101 else .35))
elif kind=='extract':
 assert (root/'native'/split/str(seed)/'simulation-complete.json').exists()
elif kind=='merge':
 assert all((root/'extracted'/str(s)/'complete.json').exists() for s in (350101,350102,350201))
def artifact(name):
 p=folder/name;p.write_bytes(b'fixture')
 return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
if kind=='simulate':
 config={'schema':'g35-native-full-day-v2','seed':seed,'split':split}
 (folder/'config.json').write_text(json.dumps(config))
 receipt={'complete':True,'seed':seed,'split':split,'returncode':0,'config_sha256':hashlib.sha256((folder/'config.json').read_bytes()).hexdigest(),'files':{'fcd.xml.gz':artifact('fcd.xml.gz')}}
 marker='simulation-complete.json'
else:
 receipt={'complete':True,'schema':'g33-direct-sumo-full-legs-v1','artifacts':{n:artifact(n) for n in ('dataset.npz','leg-metadata.json.gz')}}
 marker='complete.json'
 if kind=='extract':receipt.update(provenance_schema='g35-native-full-day-extraction-v1',episode=seed,split=split)
 else:receipt.update(provenance_schema='g35-full-day-merged-v1',training_ready=True,groups={'train':{},'val':{},'all':{}})
if mode=='bad-marker' and kind=='simulate' and seed==350101:receipt['complete']=False
(folder/marker).write_text(json.dumps(receipt));log('completed')
'''


def self_test():
    checks = []
    with tempfile.TemporaryDirectory(prefix="g35b-queue-check-") as temporary:
        base = Path(temporary)
        for mode in ("success", "failure", "bad-marker"):
            root = base/mode
            args = argparse.Namespace(root=root, scripts=Path("scripts"), source=Path("source"),
                                      sumo=Path("sumo"), cnr_splits=Path("split.json"))
            branches, merger = build_stages(args)
            def mock(stage):
                return Stage(stage.name, [sys.executable, "-u", "-c", MOCK_CHILD, str(root),
                    str(stage.folder), stage.kind, str(stage.seed), stage.split, mode],
                    stage.folder, stage.kind, stage.seed, stage.split)
            branches, merger = [(mock(a), mock(b)) for a, b in branches], mock(merger)
            kwargs = dict(poll_seconds=.02, guard_fn=lambda _: {"free_bytes": MIN_FREE+1, "budget_root_bytes": 0},
                          signal_handlers=False, stop_grace=1.)
            try:
                result = execute(root, branches, merger, [Path(__file__)], **kwargs)
            except (ValueError, RuntimeError):
                if mode == "success":
                    raise
                assert (root/"data-failed.json").exists() and not (root/"data-complete.json").exists()
            else:
                assert mode == "success" and result["training_ready"] and len(result["completed_stages"]) == 7
                assert result["merged_receipt_sha256"] == digest(merger.marker)
            events = [json.loads(line) for line in (root/"events.jsonl").read_text().splitlines()]
            starts = [e for e in events if e["kind"] == "simulate" and e["event"] == "started"]
            assert len(starts) == 3
            for e in [e for e in events if e["kind"] == "extract" and e["event"] == "started"]:
                assert any(p["kind"] == "simulate" and p["event"] == "completed" and p["seed"] == e["seed"] and p["time"] <= e["time"] for p in events)
            if mode == "success":
                early = next(e["time"] for e in events if e["kind"] == "extract" and e["seed"] == 350101 and e["event"] == "started")
                later = next(e["time"] for e in events if e["kind"] == "simulate" and e["seed"] == 350102 and e["event"] == "completed")
                assert early < later, "Fast branch extraction waited for all simulations"
                checks.extend(["three_simulations_dispatched", "per_branch_dependency", "early_extraction", "merge_after_all_extractions", "training_completion_contract"])
                try:
                    execute(root, branches, merger, [Path(__file__)], **kwargs)
                except FileExistsError:
                    checks.append("duplicate_attempt_refused")
                else:
                    raise AssertionError("Duplicate queue was accepted")
                assert not (root/"data-failed.json").exists()
                (merger.folder/"dataset.npz").write_bytes(b'tampered')
                try:
                    verify_marker(merger)
                except ValueError:
                    checks.append("artifact_tampering_rejected")
                else:
                    raise AssertionError("Changed artifact accepted")
            elif mode == "failure":
                assert not any(e["kind"] in ("extract", "merge") for e in events)
                assert len([e for e in events if e["event"] == "terminated"]) == 2
                checks.append("failed_branch_cancels_only_owned_siblings")
            else:
                assert not any(e["kind"] == "merge" for e in events)
                assert not any(e["kind"] == "extract" and e["seed"] == 350101 for e in events)
                checks.append("invalid_marker_prevents_dependent_stage")
    print(json.dumps({"complete": True, "self_test_checks": checks, "real_sumo_launched": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scripts", type=Path, default=Path("scripts"))
    parser.add_argument("--source", type=Path, default=Path("../g30-job/g30-data/source"))
    parser.add_argument("--sumo", type=Path, default=Path("g10-env/sumo/bin/sumo"))
    parser.add_argument("--cnr-splits", type=Path, default=Path("../data/processed/most64/dataset.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        stages, merger = build_stages(args)
        code_paths = [args.scripts/name for name in CODE_NAMES]+[Path("src/cnr/data.py")]
        if (args.scripts/"g35b_data_run.py").resolve() != Path(__file__).resolve():
            raise ValueError("Executing runner differs from the snapshotted scripts directory")
        execute(args.root, stages, merger, code_paths, child_env=environment())
