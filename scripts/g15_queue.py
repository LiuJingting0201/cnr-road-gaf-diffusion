"""Finite, resumable, split-by-episode simulation queue with per-class quotas."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from g15_episode import MODES

SPLITS = {"train": (262001, 8000), "validation": (263001, 1000), "holdout": (264001, 1000)}
WORKERS = 12
PER_CLASS = 100


def execute(task):
    folder, seed = task
    episode = folder / str(seed)
    episode.mkdir(parents=True, exist_ok=True)
    with (episode / "runner.log").open("a") as stream:
        subprocess.run([sys.executable, "scripts/g15_episode.py", str(folder), str(seed), str(PER_CLASS)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def collect(folder):
    counts = Counter()
    seeds = []
    for receipt in sorted(folder.glob("*/complete.json")):
        info = json.loads(receipt.read_text())
        counts.update(info["eligible"])
        seeds.append(info["seed"])
    return counts, seeds


def run(base):
    for split, (start, quota) in SPLITS.items():
        folder = base / split
        folder.mkdir(exist_ok=True)
        while True:
            counts, seeds = collect(folder)
            print(split, len(seeds), dict(counts), flush=True)
            # Slack leaves room for exact duplicate removal during export.
            required = int(quota * 1.025)
            remaining = required - min(counts[label] for label in MODES)
            if remaining <= 0:
                break
            batch = min(WORKERS, max(1, (remaining + PER_CLASS - 1) // PER_CLASS))
            limit = 2 * ((quota + PER_CLASS - 1) // PER_CLASS) + 10
            available = [s for s in range(start, start + limit) if s not in seeds]
            if len(available) < batch:
                raise RuntimeError(f"Episode cap reached for {split}")
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                list(pool.map(execute, [(folder, seed) for seed in available[:batch]]))
    subprocess.run([sys.executable, "scripts/g15_pack.py", str(base)], check=True)


if __name__ == "__main__":
    base = Path("g15-data")
    base.mkdir(exist_ok=True)
    with (base / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (base / "complete.json").exists():
            try:
                (base / "failed.txt").unlink(missing_ok=True)
                run(base)
            except Exception as exc:
                (base / "failed.txt").write_text(str(exc))
                raise
