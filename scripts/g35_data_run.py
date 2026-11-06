"""Finite sequential G35 simulation/extraction/merge queue; new attempts only."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from g33_prepare import digest, save
from g35_simulate import storage_state, stop_owned

ROOT = Path("g35-data")
MIN_FREE = 8 * 1024**3
MAX_NEW = 15 * 1024**3


def guard():
    state = storage_state(ROOT)
    if state["free_bytes"] < MIN_FREE or state["budget_root_bytes"] > MAX_NEW:
        raise RuntimeError("G35 whole-queue storage guard: " + json.dumps(state))
    return state


def interrupted(signum, _frame):
    raise InterruptedError(f"Queue received signal {signum}")


def environment():
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               PYTHONPATH="src:g10-env:scripts:../scripts:../vendor/original_cnr/src")
    return env


def data_queue():
    ROOT.mkdir(exist_ok=True)
    start, current = time.monotonic(), None
    with (ROOT / "data.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / "data-started.json").exists():
            raise FileExistsError("Existing G35 queue preserved; do not restart or duplicate")
        names = ["g35_data_run.py", "g35_simulate.py", "g35_extract.py", "g35_merge.py",
                 "g31_legs.py", "g31_replay.py", "g30_native.py", "g30_extract.py",
                 "g30_prepare.py", "g30_demand_audit.py", "g29_prepare.py", "g33_prepare.py"]
        hashes = {name: digest(Path("scripts") / name) for name in names}
        save(ROOT / "data-started.json", {"pid": os.getpid(), "started_unix": time.time(),
             "code_sha256": hashes, "seeds": [350101, 350102, 350201], "status": "started"})
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        source = "../g30-job/g30-data/source"
        common = [sys.executable, "-u"]
        commands = []
        for seed, split in ((350101, "train"), (350102, "train"), (350201, "val")):
            simulation = f"g35-data/native/{split}/{seed}"
            commands.append((f"simulate-{seed}", common + ["scripts/g35_simulate.py",
                "--source", source, "--sumo", "g10-env/sumo/bin/sumo", "--seed", str(seed),
                "--split", split, "--output", simulation, "--budget-root", str(ROOT)]))
            commands.append((f"extract-{seed}", common + ["scripts/g35_extract.py",
                "--source", source, "--simulation", simulation,
                "--cnr-splits", "../data/processed/most64/dataset.json",
                "--output", f"g35-data/extracted/{seed}"]))
        commands.append(("merge", common + ["scripts/g35_merge.py", "--inputs",
            *[f"g35-data/extracted/{s}" for s in (350101, 350102, 350201)],
            "--output", "g35-data/direct-legs"]))
        completed = []
        try:
            for name, command in commands:
                if hashes != {n: digest(Path("scripts") / n) for n in names}:
                    raise ValueError("Frozen data queue code changed")
                guard()
                with (ROOT / f"{name}.log").open("x") as stream:
                    current = subprocess.Popen(command, env=environment(), stdout=stream,
                                               stderr=subprocess.STDOUT)
                    while current.poll() is None:
                        state = guard()
                        save(ROOT / "data-progress.json", {"stage": name, "pid": current.pid,
                             "completed_stages": completed, "elapsed_s": time.monotonic()-start,
                             "storage": state, "status": "running"})
                        try:
                            current.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                    if current.returncode:
                        raise RuntimeError(f"{name} failed with code {current.returncode}; preserve log")
                completed.append(name)
                print(json.dumps({"stage_completed": name, "elapsed_s": time.monotonic()-start}), flush=True)
            receipt_path = ROOT / "direct-legs/complete.json"
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("complete") is not True:
                raise ValueError("Merged data receipt is not complete")
            save(ROOT / "data-complete.json", {"complete": True, "completed_stages": completed,
                 "seconds": time.monotonic()-start, "training_ready": receipt.get("training_ready") is True,
                 "merged_receipt_sha256": digest(receipt_path), "groups": receipt["groups"],
                 "training_completed": False})
        except BaseException as error:
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
            stop_owned(current)
            save(ROOT / "data-failed.json", {"complete": False, "error": repr(error),
                 "completed_stages": completed, "seconds": time.monotonic()-start})
            raise


if __name__ == "__main__":
    data_queue()
