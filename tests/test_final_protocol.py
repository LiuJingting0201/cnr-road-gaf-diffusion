import json
import numpy as np
import pytest
from cnr.final_protocol import verify_lock, sha256, plan_conditions, validate_requests, frozen_inputs


def test_lock_rejects_draft_and_modified_inputs(tmp_path):
    docs = tmp_path / 'docs'; docs.mkdir()
    for name in frozen_inputs():
        f = tmp_path / name; f.parent.mkdir(parents=True, exist_ok=True); f.write_text('fixture')
    protocol = docs / 'FINAL_TEST_PROTOCOL.md'
    lock = tmp_path / 'manifest.json'
    manifest = {'status': 'preflight_only', 'checks_passed': True, 'git_commit': 'abc',
                'files': {name: sha256(tmp_path / name) for name in frozen_inputs()}}
    lock.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): verify_lock(lock, tmp_path)
    manifest['status'] = 'locked'; lock.write_text(json.dumps(manifest))
    assert verify_lock(lock, tmp_path)['checks_passed']
    missing = manifest | {'files': {'docs/FINAL_TEST_PROTOCOL.md': sha256(protocol)}}
    lock.write_text(json.dumps(missing))
    with pytest.raises(ValueError): verify_lock(lock, tmp_path)
    lock.write_text(json.dumps(manifest))
    protocol.write_text('changed after locking')
    with pytest.raises(ValueError): verify_lock(lock, tmp_path)


def test_planner_schema_forbids_future_and_unpaired_ids():
    request = {'starts': np.zeros((1, 2)), 'labels': np.array([3]), 'source_index': np.array([5])}
    validate_requests(request)
    with pytest.raises(ValueError): validate_requests(request | {'xy': np.zeros((1, 512, 2))})
    with pytest.raises(ValueError):
        validate_requests({'starts': np.zeros((2, 2)), 'labels': np.array([3, 3]), 'source_index': np.array([5, 5])})


def test_fixed_budget_planning_has_no_future_and_records_partial_failure():
    requests = {'starts': np.array([[1., 2.], [3., 4.]]), 'labels': np.array([1, 3]), 'source_index': np.array([2, 6])}
    def planner(start, label, rng):
        if label == 1: return None
        route = np.tile(start, (1025, 1)); route[:, 0] += np.linspace(0, rng.uniform(10, 20), 1025)
        return route, float(route[-1, 0] - start[0])
    first, failures = plan_conditions(requests, planner)
    second, _ = plan_conditions(requests, planner)
    assert len(first['labels']) == 8
    assert failures[0]['failed_candidate_slots'] == list(range(8))
    assert set(first['source_index']) == {6}
    for key in first: np.testing.assert_array_equal(first[key], second[key])
    with pytest.raises(ValueError): plan_conditions(requests | {'real_future': np.ones((2, 512, 2))}, planner)
