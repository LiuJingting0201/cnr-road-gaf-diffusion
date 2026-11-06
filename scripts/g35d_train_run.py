"""Wait for the bounded G35 data queue, then run four fits and two GAF readers."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from g33_prepare import digest, save
from g35c_runtime import ROOT, environment, guard, interrupted


def stop_group(process):
    # Each child is launched in its own session; only our owned group is stopped.
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def verify_completion(folder):
    receipt = json.loads((folder / "complete.json").read_text())
    if receipt.get("complete") is not True or (folder / "failed.json").exists():
        raise ValueError("Unclean component completion: " + str(folder))
    for name, item in receipt.get("artifacts", {}).items():
        if digest(folder / name) != item["sha256"]:
            raise ValueError("Component artifact changed: " + str(folder / name))
    for field, name in (("model_sha256", "model.pt"), ("config_sha256", "config.json"),
                        ("stats_sha256", "stats.json")):
        if field in receipt and digest(folder / name) != receipt[field]:
            raise ValueError("Component hash changed: " + str(folder / name))
    return receipt


def training_queue():
    ROOT.mkdir(exist_ok=True)
    start, active, finished, streams = time.monotonic(), {}, {}, []
    names = ["g35d_train_run.py", "g35d_data_run.py", "g35d_simulate.py", "g35d_extract.py", "g35c_runtime.py", "g35c_teachers.py",
             "g35_data_run.py", "g35b_data_run.py", "g35_simulate.py",
             "g33_train.py", "g33_model.py", "g33_prepare.py", "g33_teachers.py",
             "g29_encode.py", "g28_teachers.py", "g23_modes.py"]
    hashes = {name: digest(Path("scripts") / name) for name in names}
    with (ROOT / "training.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / "training-started.json").exists():
            raise FileExistsError("G35 training queue already exists; do not duplicate")
        save(ROOT / "training-started.json", {"started_unix": time.time(), "pid": os.getpid(),
             "status": "waiting_for_full_day_data", "steps": 10000, "seeds": [42, 43],
             "teacher_epochs": 50, "code_sha256": hashes,
             "scope": "Queue start is not model training completion or generation evaluation"})
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        try:
            while not (ROOT / "data-complete.json").exists():
                if (ROOT / "data-failed.json").exists():
                    raise RuntimeError("Upstream data queue failed; no models launched")
                if time.monotonic() - start > 12 * 3600:
                    raise TimeoutError("Full-day data queue did not finish within twelve hours")
                state = guard()
                save(ROOT / "training-progress.json", {"stage": "waiting_for_full_day_data",
                     "seconds": time.monotonic()-start, "storage": state, "model_fits_started": 0})
                time.sleep(10)
            completion = json.loads((ROOT / "data-complete.json").read_text())
            receipt_path = ROOT / "direct-legs/complete.json"
            dataset = json.loads(receipt_path.read_text())
            if ((ROOT / "data-failed.json").exists() or completion.get("complete") is not True
                    or completion["merged_receipt_sha256"] != digest(receipt_path)
                    or dataset.get("training_ready") is not True
                    or dataset["artifacts"]["dataset.npz"]["sha256"] != digest(ROOT / "direct-legs/dataset.npz")):
                raise ValueError("Merged full-day data did not pass the declared training readiness checks")
            if hashes != {name: digest(Path("scripts") / name) for name in names}:
                raise ValueError("Frozen training queue code changed while waiting")
            # Exclusive server was authorized; refuse accidental GPU contention.
            busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            if busy:
                raise RuntimeError("GPUs contain another compute process; preserve queue without killing it")
            env = environment()
            common = [sys.executable, "-u"]

            def launch(name, command, folder, gpu=None):
                guard()
                if hashes != {n: digest(Path("scripts") / n) for n in names}:
                    raise ValueError("Frozen training source changed")
                stream = (ROOT / f"{name}.log").open("x")
                streams.append(stream)
                child_env = dict(env)
                if gpu is not None:
                    child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(common + command, stdout=stream, stderr=subprocess.STDOUT,
                                           env=child_env, start_new_session=True)
                active[name] = (process, folder)
                print(json.dumps({"component_started": name, "pid": process.pid,
                                  "gpu": gpu, "seconds": time.monotonic()-start}), flush=True)

            for gpu, arm, seed in ((0, "native_0_25s", 42), (1, "native_0_25s", 43),
                                   (2, "relative_257", 42), (3, "relative_257", 43)):
                name = f"{arm}-{seed}"
                folder = ROOT / "models" / name
                launch(name, ["scripts/g33_train.py", "--data", str(ROOT / "direct-legs/dataset.npz"),
                       "--arm", arm, "--seed", str(seed), "--steps", "10000", "--output", str(folder)], folder, gpu)
            launch("gaf-prepare", ["scripts/g35c_teachers.py", "prepare", "--data", str(ROOT / "direct-legs"),
                   "--cache", str(ROOT / "gaf-cache"), "--budget-root", str(ROOT), "--workers", "4"], ROOT / "gaf-cache")
            teachers_started = set()
            while active:
                state = guard()
                for name, (process, folder) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f"{name} exited {code}; preserve logs and completed components")
                    finished[name] = verify_completion(folder)
                    del active[name]
                    print(json.dumps({"component_completed": name, "seconds": time.monotonic()-start}), flush=True)
                for gpu, arm_name, view in ((2, "relative_257-42", "spatial"), (3, "relative_257-43", "speed")):
                    if view not in teachers_started and arm_name in finished and "gaf-prepare" in finished:
                        launch("teacher-" + view, ["scripts/g35c_teachers.py", "train", "--view", view,
                               "--cache", str(ROOT / "gaf-cache"), "--output", str(ROOT / "teachers"),
                               "--budget-root", str(ROOT)], ROOT / "teachers" / f"{view}-42", gpu)
                        teachers_started.add(view)
                save(ROOT / "training-progress.json", {"stage": "training",
                     "active": {n: p.pid for n, (p, _) in active.items()},
                     "completed_components": list(finished), "seconds": time.monotonic()-start,
                     "storage": state, "data_groups": dataset["groups"]})
                if active:
                    time.sleep(10)
            expected = {f"{arm}-{seed}" for arm in ("native_0_25s", "relative_257") for seed in (42, 43)}
            expected |= {"gaf-prepare", "teacher-spatial", "teacher-speed"}
            if set(finished) != expected:
                raise ValueError("Training queue did not complete all seven components")
            save(ROOT / "training-complete.json", {"complete": True, "seconds": time.monotonic()-start,
                 "components": finished, "data_groups": dataset["groups"], "generation_evaluated": False,
                 "data_receipt_sha256": digest(receipt_path), "code_sha256": hashes})
        except BaseException as error:
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
            for process, _ in active.values():
                stop_group(process)
            save(ROOT / "training-failed.json", {"complete": False, "error": repr(error),
                 "seconds": time.monotonic()-start, "completed_components": list(finished)})
            raise
        finally:
            for stream in streams:
                stream.close()


if __name__ == "__main__":
    training_queue()
