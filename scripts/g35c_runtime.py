"""Locked operational capacity amendment; no scientific hyperparameters change."""
import hashlib
import json
import os
from pathlib import Path
import shutil

from g35_data_run import environment, interrupted
from g35b_data_run import storage_state as tree_storage

ROOT = Path('g35-data')
MIN_FREE_BYTES = 8 * 1024**3
MAX_NEW_BYTES = 40 * 1024**3
_manifest_sha = None


def policy():
    global _manifest_sha
    path = ROOT / 'handoff-manifest.json'
    payload = path.read_bytes()
    current = hashlib.sha256(payload).hexdigest()
    if _manifest_sha is not None and current != _manifest_sha:
        raise ValueError('Frozen G35c capacity manifest changed')
    value = json.loads(payload)
    if (value['minimum_free_bytes'] != MIN_FREE_BYTES
            or value['maximum_total_bytes'] != MAX_NEW_BYTES
            or Path(value['new_job']).resolve() != Path.cwd().resolve()):
        raise ValueError('Unexpected operational capacity policy or working directory')
    roots = [Path(p).resolve() for p in value['budget_roots']]
    if len(roots) != 2 or len(set(roots)) != 2 or ROOT.resolve() not in roots:
        raise ValueError('Both old simulation and new artifact roots must be counted')
    if any(a.is_relative_to(b) for a in roots for b in roots if a != b):
        raise ValueError('Budget roots overlap')
    _manifest_sha = current
    return value, roots


def storage_state(root):
    if Path(root).resolve() != ROOT.resolve():
        raise ValueError('Expected the dedicated new output root')
    _, roots = policy()
    counts = {str(p): tree_storage(p)['budget_root_bytes'] for p in roots}
    return {'budget_root_bytes': sum(counts.values()), 'roots': counts,
            'free_bytes': min(shutil.disk_usage(p).free for p in roots)}


def guard():
    state = storage_state(ROOT)
    if state['free_bytes'] < MIN_FREE_BYTES or state['budget_root_bytes'] > MAX_NEW_BYTES:
        raise RuntimeError('G35c combined capacity guard: ' + json.dumps(state))
    return state
