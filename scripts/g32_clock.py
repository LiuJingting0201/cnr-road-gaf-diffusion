"""Three fixed clocks on the identical G31 oracle path and accepted observations.

This is a finite representation check, not model fitting or GAF efficacy.  The
new clocks store control times in seconds, including a final fractional bin;
using G22's uniform-relative-time decode for those knots would be incorrect.
"""
import argparse
import gzip
import inspect
import json
from pathlib import Path

import numpy as np

from g22_diffusion import positions, decode
from g31_roundtrip import (CLASSES, CONSTANT_ARC_M, DT, DURATION_BINS, EPS,
                           PATH_POINTS, PROGRESS_POINTS, UNITS, digest,
                           error_summary, oracle, signals, stats, summarize)


ARMS = ("relative_257", "physical_1s", "native_0_25s")
STEPS = {"physical_1s": 1., "native_0_25s": .25}
NATIVE_FIELDS = tuple(f"native_quarter_second_{signal}_{stat}_{unit}"
                      for signal, unit in (("speed", "mps"), ("acceleration", "mps2"))
                      for stat in ("mae", "rmse", "maximum_absolute"))
ERROR_FIELDS = ("position_mean_error_m", "position_p95_error_m", "position_max_error_m",
                "one_second_speed_mae_mps", "one_second_speed_rmse_mps",
                "one_second_speed_maximum_absolute_mps",
                "one_second_acceleration_mae_mps2", "one_second_acceleration_rmse_mps2",
                "one_second_acceleration_maximum_absolute_mps2",
                "stopped_state_disagreement_fraction", "steady_state_disagreement_fraction")


def save(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def control_clock(duration, arm):
    if not np.isfinite(duration) or duration <= 0 or arm not in ARMS:
        raise ValueError("A supported clock and a positive finite observation span are required")
    if arm == "relative_257":
        return np.linspace(0., duration, PROGRESS_POINTS)
    # arange excludes the end; append it even for a non-integral last interval.
    return np.r_[np.arange(0., duration, STEPS[arm]), duration]


def shared_path(t, xy):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    length = float(arc[-1])
    if length <= CONSTANT_ARC_M:
        path = np.repeat(xy[:1], PATH_POINTS, axis=0)
    else:
        keep = np.r_[True, np.diff(arc) > 0.]
        path = np.stack([np.interp(np.linspace(0., length, PATH_POINTS), arc[keep], v)
                         for v in xy[keep].T], axis=1)
    return path, arc, length


def decode_seconds(path, output_time, knot_time, progress):
    """Continuous interpolation with explicit physical knot times; no chunks."""
    output_time, knot_time, progress = [np.asarray(x) for x in (output_time, knot_time, progress)]
    if (output_time.ndim != 1 or knot_time.ndim != 1 or progress.shape != knot_time.shape
            or len(output_time) < 2 or len(knot_time) < 2
            or not all(np.isfinite(x).all() for x in (path, output_time, knot_time, progress))
            or np.any(np.diff(output_time) <= 0) or np.any(np.diff(knot_time) <= 0)
            or np.any(np.diff(progress) < 0) or progress[0] != 0 or progress[-1] != 1
            or output_time[0] != knot_time[0] or output_time[-1] != knot_time[-1]):
        raise ValueError("Invalid continuous physical-time progress contract")
    return positions(path, np.interp(output_time, knot_time, progress))


def native_series(t, xy):
    """Coordinate-derived interval speed and midpoint-time acceleration, not SUMO speed."""
    interval = np.diff(t)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)/interval
    acceleration = np.diff(speed)/((interval[:-1]+interval[1:])/2.)
    return speed, acceleration


def native_summary(rows):
    return {"n_legs": len(rows),
            "errors": {key: stats(row["native_errors"][key] for row in rows) for key in NATIVE_FIELDS}}


def native_population(rows):
    return {"all": native_summary(rows),
            "by_class": {name: native_summary([r for r in rows if r["label"] == c])
                         for c, name in enumerate(CLASSES)},
            "by_observed_duration_bin": {name: native_summary([r for r in rows if low <= r["geometry"]["observation_duration_s"] < high])
                                         for low, high, name in DURATION_BINS}}


def numeric_equal(left, right, path="root"):
    """Strict shape/contract comparison, with tolerance only for finite scalars."""
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError(f"Legacy aggregate keys differ at {path}")
        for key in left:
            numeric_equal(left[key], right[key], path+"/"+str(key))
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise ValueError(f"Legacy sequence differs at {path}")
        for i, (a, b) in enumerate(zip(left, right)):
            numeric_equal(a, b, path+f"/{i}")
    elif isinstance(left, (int, float, np.number)) and not isinstance(left, (bool, np.bool_)):
        if (not isinstance(right, (int, float, np.number))
                or not np.isfinite([left, right]).all()
                or not np.isclose(left, right, atol=1e-12, rtol=1e-12)):
            raise ValueError(f"Legacy scalar differs at {path}")
    elif left != right:
        raise ValueError(f"Legacy field differs at {path}")


def compare_leg(txy, event_start, event_end):
    # Reuse the frozen baseline's validation and actual row, not a rewritten arm.
    baseline, original_example = oracle(txy, event_start, event_end)
    txy = np.asarray(txy, dtype=float)
    t, xy = txy[:, 0]-txy[0, 0], txy[:, 1:]
    path, arc, length = shared_path(t, xy)
    actual, speed, acceleration, stop, steady, _ = signals(t, xy)
    duration, constant = float(t[-1]), length <= CONSTANT_ARC_M
    native_speed, native_acceleration = native_series(t, xy)
    rows, controls = {"relative_257": baseline}, {}
    for arm in ARMS:
        knot_time = control_clock(duration, arm)
        progress = (knot_time/duration if constant else np.interp(knot_time, t, arc)/length).astype(np.float32)
        if arm == "relative_257":
            # Reproduce the actual G22 expression, including floating evaluation.
            if constant:
                progress = np.linspace(0., 1., PROGRESS_POINTS).astype(np.float32)
            recovered_t, recovered_xy = decode(path, duration, progress, dt=DT)
            if not np.array_equal(recovered_t, original_example["time"]):
                raise ValueError("Old clock no longer reproduces G31 time support")
            if not np.array_equal(recovered_xy, original_example["reconstructed_xy"]):
                raise ValueError("Shared path/old clock does not reproduce G31 coordinates exactly")
        else:
            recovered_xy = decode_seconds(path, t, knot_time, progress)
            recovered, rv, ra, rs, rst, _ = signals(t, recovered_xy)
            distance = np.linalg.norm(recovered_xy-xy, axis=1)
            geometry = dict(baseline["geometry"])
            geometry.update(position_mean_error_m=float(distance.mean()),
                            position_p95_error_m=float(np.quantile(distance, .95)),
                            position_max_error_m=float(distance.max()),
                            start_position_error_m=float(distance[0]), end_position_error_m=float(distance[-1]),
                            arc_length_fraction_error=(recovered["arc_length_m"]-length)/length if not constant else None,
                            stopped_state_disagreement_fraction=float(np.mean(rs != stop)) if len(speed) else None,
                            steady_state_disagreement_fraction=float(np.mean(rst != steady)) if len(steady) else None)
            geometry.update({f"one_second_speed_{k}_mps": v for k, v in error_summary(rv-speed).items()})
            geometry.update({f"one_second_acceleration_{k}_mps2": v for k, v in error_summary(ra-acceleration).items()})
            physical = {}
            for key in UNITS:
                a, b = actual[key], recovered[key]
                if (a is None) != (b is None):
                    raise ValueError("Metric support differs between clocks")
                delta = b-a if a is not None else None
                physical[key] = {"observed": a, "reconstructed": b, "signed_error": delta,
                                 "absolute_error": abs(delta) if delta is not None else None}
            rows[arm] = {"constant": constant, "under_5s": duration < 5.,
                         "geometry": geometry, "physical": physical}
        recovered_native_speed, recovered_native_acceleration = native_series(t, recovered_xy)
        rows[arm]["native_errors"] = {
            **{f"native_quarter_second_speed_{k}_mps": v for k, v in error_summary(recovered_native_speed-native_speed).items()},
            **{f"native_quarter_second_acceleration_{k}_mps2": v for k, v in error_summary(recovered_native_acceleration-native_acceleration).items()}}
        controls[arm] = {"control_points": len(knot_time),
                         "control_point_ratio_to_257": len(knot_time)/PROGRESS_POINTS,
                         "maximum_control_interval_s": float(np.diff(knot_time).max()),
                         "last_control_interval_s": float(knot_time[-1]-knot_time[-2]),
                         "progress_precision": "float32"}
    return rows, controls


def population(rows):
    return {"all": summarize(rows),
            "by_class": {name: summarize([r for r in rows if r["label"] == c])
                         for c, name in enumerate(CLASSES)},
            "by_observed_duration_bin": {name: summarize([r for r in rows if low <= r["geometry"]["observation_duration_s"] < high])
                                         for low, high, name in DURATION_BINS},
            "constant_only": summarize([r for r in rows if r["constant"]]),
            "under_5s_only": summarize([r for r in rows if r["under_5s"]])}


def paired_summary(left, right):
    if len(left) != len(right):
        raise ValueError("Arms have different populations")
    def difference(a, b):
        if (a is None) != (b is None):
            raise ValueError("Paired metrics have different support")
        return a-b if a is not None else None
    return {"n_legs": len(left),
            "native_quarter_second_error_difference": {key: stats(difference(a["native_errors"][key], b["native_errors"][key])
                                                                 for a, b in zip(left, right)) for key in NATIVE_FIELDS},
            "aligned_error_difference": {key: stats(difference(a["geometry"][key], b["geometry"][key])
                                                   for a, b in zip(left, right)) for key in ERROR_FIELDS},
            "physical_absolute_error_difference": {key: stats(difference(a["physical"][key]["absolute_error"],
                                                                          b["physical"][key]["absolute_error"])
                                                               for a, b in zip(left, right)) for key in UNITS}}


def paired_population(left, right):
    return {"all": paired_summary(left, right),
            "by_class": {name: paired_summary([r for r in left if r["label"] == c], [r for r in right if r["label"] == c])
                         for c, name in enumerate(CLASSES)},
            "by_observed_duration_bin": {name: paired_summary(
                [r for r in left if low <= r["geometry"]["observation_duration_s"] < high],
                [r for r in right if low <= r["geometry"]["observation_duration_s"] < high])
                for low, high, name in DURATION_BINS}}


def load(data, legacy_path):
    receipt_path = data/"complete.json"
    receipt = json.loads(receipt_path.read_text())
    legacy = json.loads(legacy_path.read_text())
    if (not receipt.get("complete") or receipt["classes"] != CLASSES or not legacy.get("complete")
            or legacy["classes"] != CLASSES or legacy["sources"]["extraction_receipt_sha256"] != digest(receipt_path)):
        raise ValueError("G31 extraction and its completed reconstruction receipt must agree")
    for name in ("legs.npz", "leg-metadata.json.gz"):
        if digest(data/name) != receipt["artifacts"][name]["sha256"]:
            raise ValueError("Saved G31 leg data changed")
    if digest(Path(inspect.getsourcefile(oracle))) != legacy["sources"]["script_sha256"]:
        raise ValueError("G31 oracle code differs from its completed execution")
    if digest(Path(inspect.getsourcefile(decode))) != legacy["sources"]["g22_diffusion_file_sha256"]:
        raise ValueError("Frozen G22 decoder differs from its completed G31 execution")
    with np.load(data/"legs.npz", allow_pickle=False) as archive:
        keys = ("ids", "parent_base_ids", "partitions", "labels", "ordinals", "event_start", "event_end", "offset", "txy")
        values = {k: archive[k] for k in keys}
    with gzip.open(data/"leg-metadata.json.gz", "rt", encoding="utf-8") as stream:
        metadata = json.load(stream)["accepted"]
    n = len(metadata)
    if (n != receipt["accepted_legs"] or n != legacy["all"]["n_legs"]
            or len(set(values["ids"])) != n
            or any(len(values[k]) != n for k in keys if k not in ("offset", "txy"))):
        raise ValueError("Accepted arrays, metadata and receipts are not aligned")
    offset = values["offset"]
    if (offset.dtype.kind not in "iu" or offset.shape != (n+1,) or offset[0] != 0
            or offset[-1] != len(values["txy"]) or np.any(np.diff(offset) < 2)):
        raise ValueError("Invalid accepted leg offsets")
    parents = {}
    for i, meta in enumerate(metadata):
        c, parent, split = int(values["labels"][i]), str(values["parent_base_ids"][i]), str(values["partitions"][i])
        if (not 0 <= c < len(CLASSES) or split not in ("train", "val")
                or parents.get(parent, split) != split
                or meta["leg_id"] != str(values["ids"][i]) or meta["parent_base_id"] != parent
                or meta["partition"] != split or int(meta["label"]) != c
                or int(meta["ordinal"]) != int(values["ordinals"][i])
                or meta["event_start"] != float(values["event_start"][i])
                or meta["event_end"] != float(values["event_end"][i])):
            raise ValueError("Leg identity, partition or event metadata is inconsistent")
        parents[parent] = split
    return receipt, legacy, values, metadata


def run(data, legacy_path, output):
    data, legacy_path, output = data.resolve(), legacy_path.resolve(), output.resolve()
    for protected in (data, legacy_path.parent):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("New output must be separate from immutable G31 inputs")
    output.mkdir(parents=True, exist_ok=False)
    try:
        receipt, legacy, values, metadata = load(data, legacy_path)
        rows, control_rows = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}
        for i, meta in enumerate(metadata):
            a, b = values["offset"][i:i+2]
            result, controls = compare_leg(values["txy"][a:b], values["event_start"][i], values["event_end"][i])
            for arm in ARMS:
                row = result[arm]
                if not np.isclose(row["geometry"]["observation_duration_s"], meta["coverage"]["observed_span_s"], atol=EPS, rtol=0):
                    raise ValueError("Observation duration differs from accepted metadata")
                row.update(parent=str(values["parent_base_ids"][i]), partition=str(values["partitions"][i]),
                           label=int(values["labels"][i]), kind=meta["kind"])
                rows[arm].append(row)
                control_rows[arm].append(controls[arm])
        aggregates = {arm: population(rows[arm]) for arm in ARMS}
        for key, value in aggregates["relative_257"].items():
            numeric_equal(value, legacy[key], path=key)
        control_keys = ("control_points", "control_point_ratio_to_257", "maximum_control_interval_s", "last_control_interval_s")
        costs = {}
        for arm in ARMS:
            costs[arm] = {"all": {key: stats(row[key] for row in control_rows[arm]) for key in control_keys},
                          "by_class": {name: {key: stats(row[key] for row, original in zip(control_rows[arm], rows[arm]) if original["label"] == c)
                                              for key in control_keys} for c, name in enumerate(CLASSES)},
                          "by_observed_duration_bin": {name: {key: stats(row[key] for row, original in zip(control_rows[arm], rows[arm])
                                                                        if low <= original["geometry"]["observation_duration_s"] < high)
                                                              for key in control_keys} for low, high, name in DURATION_BINS}}
        report = {"complete": True, "generation_ready": False, "classes": CLASSES,
                  "scope": "Fixed three-clock oracle representation comparison on every saved G31 pilot leg; no fitting, synthesis or GAF experiment",
                  "arms": aggregates, "native_quarter_second_errors": {arm: native_population(rows[arm]) for arm in ARMS},
                  "representation_cost": costs,
                  "paired_error_differences": {f"{a}-minus-{b}": paired_population(rows[a], rows[b])
                                               for a, b in (("physical_1s", "relative_257"), ("native_0_25s", "relative_257"), ("native_0_25s", "physical_1s"))},
                  "checks": {"all_saved_legs_retained": True, "same_oracle_path_all_arms": True,
                             "old_257_coordinates_exactly_reproduced_per_leg": True,
                             "all_G31_aggregate_metrics_reproduced": True, "same_metric_support_each_arm": True},
                  "method": {"spatial_path": "Identical G31 2049-point target-derived equal-arc path, float64, shared by all arms",
                             "progress": "Normalized cumulative observed arc, float32 in all arms; constant-path progress is time/duration",
                             "relative_257": "Frozen G22 positions/decode with 257 equally spaced relative-time controls",
                             "physical_1s": "Control times [0,1,...,floor(T),T], unique terminal endpoint; interpolate with actual seconds",
                             "native_0_25s": "Control times at all native .25 s samples; target-derived resolution control, not a learned model",
                             "output_and_position_errors": "Original .25-second observation clock in all arms; no invented event-boundary coordinate",
                             "motion_metrics": "Inherited G31 one-second cumulative-XY-arc speed/acceleration; same ten endpoints and support rules",
                             "native_auxiliary_metrics": "Coordinate displacement / actual native .25s interval, then speed difference / interval-midpoint time difference; MAE/RMSE/max, separate from ten main endpoints and not simulator-reported speed",
                             "aggregation": "Descriptive per-leg summaries and paired differences, lower error is better; repeated parent legs remain dependent; no confidence intervals",
                             "capacity_cost": "Control-point counts are representation size, not a measured model-runtime or equal-compute claim"},
                  "sources": {"extraction_receipt_sha256": digest(data/"complete.json"), "G31_reconstruction_receipt_sha256": digest(legacy_path),
                              "legs_sha256": digest(data/"legs.npz"), "metadata_sha256": digest(data/"leg-metadata.json.gz"),
                              "g31_roundtrip_file_sha256": digest(Path(inspect.getsourcefile(oracle))),
                              "g22_diffusion_file_sha256": digest(Path(inspect.getsourcefile(decode))), "script_sha256": digest(Path(__file__))},
                  "limits": ["All three arms use target-derived spatial paths, durations and control values unavailable at genuine generation time.",
                             "Smaller oracle error does not establish that a model can learn the longer sequence or that GAF helps it.",
                             "Integer-second motion metrics are aligned to the physical_1s knots; native .25s position, speed and acceleration errors plus the .25s control expose remaining within-second loss. Better one-second metrics do not establish that one second is universally sufficient.",
                             "No long or short leg is removed; unsupported short-leg endpoints remain explicitly counted.",
                             "One pilot with shared demand is not independent generation evaluation or a new-line generalization test.",
                             "Native-resolution control retains spatial discretization and float32 progress error and is not assumed exact."]}
        save(output/"complete.json", report)
        print(json.dumps({"complete": True, "legs": len(metadata), "arms": list(ARMS),
                          "old_aggregate_replay": True, "generation_ready": False}), flush=True)
        return report
    except BaseException as error:
        save(output/"failed.json", {"complete": False, "error": repr(error), "script_sha256": digest(Path(__file__))})
        raise


def self_test():
    numeric_equal(control_clock(1.5, "physical_1s").tolist(), [0., 1., 1.5])
    numeric_equal(control_clock(.5, "physical_1s").tolist(), [0., .5])
    numeric_equal(control_clock(2., "physical_1s").tolist(), [0., 1., 2.])
    t = np.arange(41)*DT
    linear = np.column_stack([100.+t, 2*t, 0*t])
    rows, costs = compare_leg(linear, 99.75, 110.25)
    for arm in ARMS:
        assert rows[arm]["geometry"]["position_max_error_m"] < 1e-5
        assert rows[arm]["geometry"]["event_minus_observation_span_s"] == .5
    assert [costs[a]["control_points"] for a in ARMS] == [257, 11, 41]
    constant = linear.copy(); constant[:, 1:] = 7.
    rows, _ = compare_leg(constant, 100., 110.)
    assert all(row["constant"] and row["geometry"]["position_max_error_m"] == 0. for row in rows.values())
    rows, _ = compare_leg(linear[:3], 100., 100.5)
    assert all(row["under_5s"] and row["physical"]["mean_speed_mps"]["observed"] is None for row in rows.values())
    # A within-second stop/accelerate pattern is hidden by only two 1s controls;
    # native samples must expose that difference even if the 1s average agrees.
    pulse = np.array([[0., 0., 0.], [.25, 0., 0.], [.5, 0., 0.], [.75, .5, 0.], [1., 1., 0.]])
    rows, _ = compare_leg(pulse, 0., 1.)
    assert rows["physical_1s"]["geometry"]["position_max_error_m"] > .4
    assert rows["native_0_25s"]["geometry"]["position_max_error_m"] == 0.
    assert rows["physical_1s"]["physical"]["mean_speed_mps"]["absolute_error"] == 0.
    assert rows["physical_1s"]["native_errors"]["native_quarter_second_speed_mae_mps"] == 1.
    assert rows["native_0_25s"]["native_errors"]["native_quarter_second_speed_mae_mps"] == 0.
    assert rows["physical_1s"]["native_errors"]["native_quarter_second_acceleration_mae_mps2"] > 0.
    # A 512 s trip has two seconds per legacy control interval. A stop/go cycle
    # survives a physical clock but is averaged out by 257 relative controls.
    long_time = np.arange(2049)*DT
    long_speed = np.where((np.floor(long_time[:-1]).astype(int) % 2) == 0, 0., 2.)
    long_xy = np.c_[np.r_[0., np.cumsum(long_speed*DT)], np.zeros(len(long_time))]
    rows, _ = compare_leg(np.c_[long_time, long_xy], 0., 512.)
    assert rows["relative_257"]["geometry"]["one_second_speed_mae_mps"] == 1.
    assert rows["physical_1s"]["geometry"]["one_second_speed_mae_mps"] == 0.
    assert rows["physical_1s"]["native_errors"]["native_quarter_second_speed_mae_mps"] == 0.
    # A fractional last bin must not be stretched by an equal-relative decoder.
    knot = np.array([0., 1., 1.5]); profile = np.array([0., 0., 1.], dtype=np.float32)
    path = np.c_[np.linspace(0., 1., PATH_POINTS), np.zeros(PATH_POINTS)]
    sample_time = np.arange(7)*DT
    xy = decode_seconds(path, sample_time, knot, profile)
    assert xy[4, 0] == 0. and xy[-1, 0] == 1.
    _, wrong = decode(path, 1.5, profile)
    assert wrong[4, 0] > 0.
    for invalid in (linear[[0, 2, 3]], np.array([[100., 0., 0.], [100.25, 100., 0.]])):
        try:
            compare_leg(invalid, 100., float(invalid[-1, 0]))
        except ValueError:
            pass
        else:
            raise AssertionError("Bad native observations were accepted")
    print("g32_clock_construct_checks_passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../g31-job/g31-data/legs/pilot-300001"))
    parser.add_argument("--legacy-report", type=Path, default=Path("../g31-job/g31-data/roundtrip/pilot-300001/complete.json"))
    parser.add_argument("--output", type=Path, default=Path("g32-data/clock-pilot"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.data, args.legacy_report, args.output)
