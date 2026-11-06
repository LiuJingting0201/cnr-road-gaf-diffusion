"""Fixed G22 oracle representation audit on every accepted G31 observed leg.

The spatial path and progress both use the observed target. This is an optimistic
representation diagnostic, never evidence of planned-map decoder correctness or
generation quality. Identifiers and trajectory arrays are not exported in JSON.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from g22_diffusion import positions, decode


CLASSES = ["pedestrian", "bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
PATH_POINTS = 2049
PROGRESS_POINTS = 257
DT = .25
EPS = 1e-7
CONSTANT_ARC_M = 1e-12
DURATION_BINS = [(0., 5., "under_5s"), (5., 60., "5_to_under_60s"),
                 (60., 300., "60_to_under_300s"), (300., 900., "300_to_under_900s"),
                 (900., float("inf"), "900s_and_over")]
UNITS = {"duration_s": "s", "arc_length_m": "m", "longest_steady_s": "s",
         "mean_absolute_lag4_speed_change_mps": "m/s", "mean_speed_mps": "m/s",
         "mean_absolute_acceleration_mps2": "m/s^2", "stopped_fraction": "fraction",
         "speed_p95_mps": "m/s", "longest_stop_s": "s", "stop_episodes_per_minute": "episodes/min"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    values = list(values)
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite diagnostic values cannot be silently omitted")
    result = {"n": len(x), "unsupported_n": len(values)-len(x)}
    if len(x):
        result.update(mean=float(x.mean()), min=float(x.min()), median=float(np.median(x)),
                      p95=float(np.quantile(x, .95)), p99=float(np.quantile(x, .99)), max=float(x.max()))
    return result


def runs(mask):
    edge = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    lengths = np.flatnonzero(edge == -1)-np.flatnonzero(edge == 1)
    return int(lengths.max()) if len(lengths) else 0, int(len(lengths))


def signals(t, xy):
    """Same one-second cumulative-arc clock as G23/G28/G30; no short-leg drop."""
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    grid = np.arange(int(np.floor(t[-1]))+1)
    v = np.diff(np.interp(grid, t, arc))
    acceleration = np.diff(v)
    stop = v < .3
    steady = (v[1:] >= .3) & (np.abs(acceleration) < .5)
    values = {key: None for key in UNITS}
    values.update(duration_s=float(t[-1]), arc_length_m=float(arc[-1]))
    if len(v):
        longest, number = runs(stop)
        values.update(mean_speed_mps=float(v.mean()), stopped_fraction=float(stop.mean()),
                      speed_p95_mps=float(np.quantile(v, .95)), longest_stop_s=longest,
                      stop_episodes_per_minute=60.*number/len(v))
    if len(v) >= 2:
        values.update(longest_steady_s=runs(steady)[0],
                      mean_absolute_acceleration_mps2=float(np.abs(acceleration).mean()))
    if len(v) >= 5:
        values["mean_absolute_lag4_speed_change_mps"] = float(np.abs(v[4:]-v[:-4]).mean())
    return values, v, acceleration, stop, steady, arc


def error_summary(error):
    if not len(error):
        return {"mae": None, "rmse": None, "maximum_absolute": None}
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "maximum_absolute": float(np.abs(error).max())}


def oracle(txy, event_start, event_end):
    txy = np.asarray(txy, dtype=float)
    if txy.ndim != 2 or txy.shape[1] != 3 or len(txy) < 2 or not np.isfinite(txy).all():
        raise ValueError("Each accepted leg needs at least two finite t,x,y observations")
    t, xy = txy[:, 0]-txy[0, 0], txy[:, 1:]
    if not np.allclose(np.diff(t), DT, atol=EPS, rtol=0):
        raise ValueError("An accepted observation contains a native-clock gap")
    start_delay, end_early = float(txy[0, 0]-event_start), float(event_end-txy[-1, 0])
    if (not np.isfinite([event_start, event_end]).all() or event_end <= event_start
            or start_delay < -EPS or end_early < -EPS or max(start_delay, end_early) > DT+EPS):
        raise ValueError("Observed samples do not cover the accepted event within quarter-second tolerance")
    actual, speed, acceleration, stop, steady, arc = signals(t, xy)
    if np.any(np.linalg.norm(np.diff(xy, axis=0), axis=1)/DT > 100.):
        raise ValueError("Accepted leg violates the inherited displacement-rate bound")
    duration, length = float(t[-1]), float(arc[-1])
    constant = length <= CONSTANT_ARC_M
    if constant:
        path = np.repeat(xy[:1], PATH_POINTS, axis=0)
        # The normalized progress of a constant path is unidentifiable. This
        # convention preserves the G22 [0,1] profile without inventing motion.
        profile = np.linspace(0., 1., PROGRESS_POINTS).astype(np.float32)
        direct_progress = np.zeros(len(t))
    else:
        keep = np.r_[True, np.diff(arc) > 0.]
        support, points = arc[keep], xy[keep]
        path = np.stack([np.interp(np.linspace(0., length, PATH_POINTS), support, v) for v in points.T], axis=1)
        profile = (np.interp(np.linspace(0., duration, PROGRESS_POINTS), t, arc)/length).astype(np.float32)
        direct_progress = arc/length
    recovered_t, recovered_xy = decode(path, duration, profile, dt=DT)
    if recovered_t.shape != t.shape or not np.allclose(recovered_t, t, atol=EPS, rtol=0):
        raise ValueError("G22 decoder changed the observed native time support")
    recovered, rv, ra, rs, rst, recovered_arc = signals(recovered_t, recovered_xy)
    path_only = positions(path, direct_progress)
    distance = np.linalg.norm(recovered_xy-xy, axis=1)
    path_distance = np.linalg.norm(path_only-xy, axis=1)
    geometry = {"position_mean_error_m": float(distance.mean()),
                "position_p95_error_m": float(np.quantile(distance, .95)),
                "position_max_error_m": float(distance.max()),
                "start_position_error_m": float(distance[0]), "end_position_error_m": float(distance[-1]),
                "path_only_position_mean_error_m": float(path_distance.mean()),
                "path_only_position_max_error_m": float(path_distance.max()),
                "observation_duration_s": duration, "event_duration_s": float(event_end-event_start),
                "first_observation_delay_s": start_delay, "last_observation_early_s": end_early,
                "event_minus_observation_span_s": float(event_end-event_start-duration),
                "arc_length_fraction_error": (recovered["arc_length_m"]-length)/length if not constant else None,
                "covered_whole_seconds": len(speed), "uncovered_fractional_tail_s": duration-len(speed),
                "stopped_state_disagreement_fraction": float(np.mean(rs != stop)) if len(speed) else None,
                "steady_state_disagreement_fraction": float(np.mean(rst != steady)) if len(steady) else None}
    geometry.update({f"one_second_speed_{k}_mps": v for k, v in error_summary(rv-speed).items()})
    geometry.update({f"one_second_acceleration_{k}_mps2": v for k, v in error_summary(ra-acceleration).items()})
    errors = {}
    for key in UNITS:
        a, b = actual[key], recovered[key]
        if (a is None) != (b is None):
            raise ValueError("Observed and reconstructed metric support differs")
        delta = b-a if a is not None else None
        errors[key] = {"observed": a, "reconstructed": b, "signed_error": delta,
                       "absolute_error": abs(delta) if delta is not None else None}
    row = {"constant": constant, "under_5s": duration < 5., "geometry": geometry, "physical": errors}
    example = {"time": t, "observed_xy": xy, "reconstructed_xy": recovered_xy,
               "observed_speed": speed, "reconstructed_speed": rv,
               "observed_arc": arc, "reconstructed_arc": recovered_arc}
    return row, example


def summarize(rows):
    result = {"n_legs": len(rows), "n_parents": len({r["parent"] for r in rows}),
              "constant_legs": sum(r["constant"] for r in rows),
              "under_5s_legs": sum(r["under_5s"] for r in rows),
              "partition_counts": dict(Counter(r["partition"] for r in rows)),
              "kind_counts": dict(Counter(r["kind"] for r in rows))}
    if not rows:
        return result
    result["geometry_and_aligned_series_errors"] = {key: stats(r["geometry"][key] for r in rows)
                                                    for key in rows[0]["geometry"]}
    result["physical"] = {key: {"unit": UNITS[key], **{
        field: stats(r["physical"][key][field] for r in rows)
        for field in ("observed", "reconstructed", "signed_error", "absolute_error")}} for key in UNITS}
    return result


def figure(example, path, class_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = example["time"]
    raw, rebuilt = example["observed_xy"], example["reconstructed_xy"]
    origin = raw[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for xy, label, style in ((raw, "Observed", "-"), (rebuilt, "Oracle roundtrip", "--")):
        axes[0].plot(xy[:, 0]-origin[0], xy[:, 1]-origin[1], style, label=label, linewidth=1.3)
    axes[0].scatter([0], [0], s=20, color="black", marker="o", label="Observed start")
    axes[0].set(xlabel="x relative to observed start (m)", ylabel="y relative to observed start (m)", aspect="equal")
    for key, label, style in (("observed_speed", "Observed", "-"), ("reconstructed_speed", "Oracle roundtrip", "--")):
        v = example[key]
        axes[1].step(np.arange(len(v))+.5, v, style, where="mid", label=label, linewidth=1.)
    axes[1].set(xlabel="Observed elapsed time (s)", ylabel="One-second arc speed (m/s)")
    if not len(example["observed_speed"]):
        axes[1].text(.5, .5, "No complete 1 s bin", transform=axes[1].transAxes, ha="center")
    for key, label, style in (("observed_arc", "Observed", "-"), ("reconstructed_arc", "Oracle roundtrip", "--")):
        axes[2].plot(t, example[key], style, label=label, linewidth=1.)
    axes[2].set(xlabel="Observed elapsed time (s)", ylabel="Cumulative observed-path arc (m)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"{class_name}: first accepted leg in archive order; oracle, not a generated trip")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(data, output):
    if output.exists():
        raise FileExistsError("Preserve prior audits; choose a new output directory")
    receipt_path = data/"complete.json"
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get("complete") or receipt["classes"] != CLASSES:
        raise ValueError("Expected completed six-class G31 extraction")
    for filename in ("legs.npz", "leg-metadata.json.gz"):
        if digest(data/filename) != receipt["artifacts"][filename]["sha256"]:
            raise ValueError("Accepted leg artifact has changed since extraction")
    with np.load(data/"legs.npz", allow_pickle=False) as archive:
        keys = ("ids", "parent_base_ids", "partitions", "labels", "ordinals", "event_start", "event_end", "offset", "txy")
        values = {key: archive[key] for key in keys}
    with gzip.open(data/"leg-metadata.json.gz", "rt", encoding="utf-8") as stream:
        metadata = json.load(stream)["accepted"]
    n = len(values["ids"])
    if n != receipt["accepted_legs"] or n != len(metadata) or len(set(values["ids"])) != n:
        raise ValueError("Accepted rows, metadata or leg IDs are not aligned")
    if any(len(values[k]) != n for k in keys if k not in ("offset", "txy")):
        raise ValueError("Accepted metadata arrays are not aligned")
    offset = values["offset"]
    if (offset.dtype.kind not in "iu" or offset.shape != (n+1,) or offset[0] != 0
            or offset[-1] != len(values["txy"]) or np.any(np.diff(offset) < 2)):
        raise ValueError("Invalid accepted leg offsets")
    if np.any((values["labels"] < 0) | (values["labels"] >= len(CLASSES))):
        raise ValueError("Unsupported class; no eight-class assumption is permitted")
    rows, examples, parent_partitions = [], {}, {}
    for i, meta in enumerate(metadata):
        if (meta["leg_id"] != str(values["ids"][i]) or meta["parent_base_id"] != str(values["parent_base_ids"][i])
                or int(meta["label"]) != int(values["labels"][i]) or meta["partition"] != str(values["partitions"][i])
                or int(meta["ordinal"]) != int(values["ordinals"][i])
                or not np.isclose(meta["event_start"], values["event_start"][i], atol=EPS, rtol=0)
                or not np.isclose(meta["event_end"], values["event_end"][i], atol=EPS, rtol=0)):
            raise ValueError("Accepted NPZ and JSON row metadata disagree")
        parent, partition = str(values["parent_base_ids"][i]), str(values["partitions"][i])
        if partition not in ("train", "val") or parent_partitions.get(parent, partition) != partition:
            raise ValueError("Test data or a parent crossing partitions entered the interface audit")
        parent_partitions[parent] = partition
        row, example = oracle(values["txy"][offset[i]:offset[i+1]], values["event_start"][i], values["event_end"][i])
        if not np.isclose(row["geometry"]["observation_duration_s"], meta["coverage"]["observed_span_s"], atol=EPS, rtol=0):
            raise ValueError("Saved observed-span metadata disagrees with native timestamps")
        c = int(values["labels"][i])
        row.update(parent=parent, partition=partition, kind=meta["kind"], label=c)
        rows.append(row)
        if c not in examples:
            examples[c] = example
    output.mkdir(parents=True)
    figures = {}
    for c, example in examples.items():
        filename = f"first-{CLASSES[c]}.png"
        figure(example, output/filename, CLASSES[c])
        figures[CLASSES[c]] = {"file": filename, "sha256": digest(output/filename),
                               "selection": "First accepted leg in frozen archive order, without score or duration filtering"}
    bins = {name: summarize([r for r in rows if low <= r["geometry"]["observation_duration_s"] < high])
            for low, high, name in DURATION_BINS}
    decoder_path = Path(inspect.getsourcefile(decode))
    report = {"complete": True, "generation_ready": False,
              "scope": "Oracle reconstruction of all accepted observed G31 leg samples, not a generative result or planned-map decoder validation",
              "classes": CLASSES, "all": summarize(rows),
              "by_class": {name: summarize([r for r in rows if r["label"] == c]) for c, name in enumerate(CLASSES)},
              "by_observed_duration_bin": bins,
              "constant_only": summarize([r for r in rows if r["constant"]]),
              "under_5s_only": summarize([r for r in rows if r["under_5s"]]),
              "method": {"spatial_path": f"{PATH_POINTS} equal arc-length points from observed target XY; float64",
                         "progress": f"{PROGRESS_POINTS} samples of target cumulative arc at uniform relative observation times, normalized by observed arc; float32 as in G22",
                         "decoder": "Unchanged g22_diffusion.positions/decode at .25 s",
                         "clock": "Actual first-to-last stored FCD timestamps; no invented event-boundary coordinates",
                         "physical_clock": "Cumulative XY chord arc interpolated at integer seconds relative to first observation; last fractional second excluded from speed summaries",
                         "steady": "v[1:] >= .3 m/s and abs(diff(v)) < .5 m/s, counting qualifying one-second transitions",
                         "stop": "v < .3 m/s; stop episodes count all runs including initial stop; rate per covered whole minute",
                         "lag4": "mean(abs(v[4:] - v[:-4])); not divided by four",
                         "short_support": "All legs retained for geometry. Speed/stop need >=1 whole second; acceleration/steady >=2; lag4 >=5. Unsupported values counted, not silently dropped",
                         "constant": f"Observed XY arc <= {CONSTANT_ARC_M} m: constant path and uniform [0,1] progress; exact stationary representation diagnostic, not motion-normalization training readiness",
                         "path_only_control": "Same oracle path evaluated at original observed normalized cumulative arc before 257-point time compression",
                         "aggregation": "Per-leg distributions; repeated legs of one parent remain dependent. No confidence intervals or different-unit total scores"},
              "limits": ["The target-derived path is unavailable in genuine synthesis. Errors are optimistic interface diagnostics, not a mathematical lower bound for all other path choices.",
                         "Zero observed endpoint error is by construction and does not establish exact event arrival coordinates or planned-route validity.",
                         "Accepted event coverage does not imply topology, permissible source conditions, or generative readiness.",
                         "This one pilot reuses public demand and is not independent generative evaluation. No model is fit or selected.",
                         "No threshold for declaring representation adequate is fitted to these results; long-tail and per-class errors must be considered."],
              "sources": {"extraction_receipt_sha256": digest(receipt_path), "legs_sha256": digest(data/"legs.npz"),
                          "metadata_sha256": digest(data/"leg-metadata.json.gz"), "script_sha256": digest(Path(__file__)),
                          "g22_diffusion_file_sha256": digest(decoder_path),
                          "decoder_functions_sha256": hashlib.sha256((inspect.getsource(positions)+inspect.getsource(decode)).encode()).hexdigest()},
              "figures": figures}
    temporary = output/"complete.tmp.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(output/"complete.json")
    print(json.dumps({"complete": True, "n_legs": n, "class_counts": {name: r["n_legs"] for name, r in report["by_class"].items()},
                      "constant_legs": report["all"]["constant_legs"], "under_5s_legs": report["all"]["under_5s_legs"]}), flush=True)
    return report


def self_test():
    t = np.arange(41)*DT
    linear = np.column_stack([100.+t, 2*t, 0*t])
    row, _ = oracle(linear, 99.75, 110.25)
    assert row["geometry"]["position_max_error_m"] < 1e-5
    assert row["geometry"]["event_minus_observation_span_s"] == .5
    assert row["physical"]["mean_absolute_lag4_speed_change_mps"]["observed"] == 0.
    constant = linear.copy(); constant[:, 1:] = 7.
    row, _ = oracle(constant, 100., 110.)
    assert row["constant"] and row["geometry"]["position_max_error_m"] == 0.
    assert row["physical"]["stopped_fraction"]["observed"] == 1.
    row, _ = oracle(linear[:3], 100., 100.5)
    assert row["under_5s"] and row["physical"]["mean_speed_mps"]["observed"] is None
    assert stats([None, 1.])["unsupported_n"] == 1
    bent = linear.copy(); bent[:, 1] = np.maximum(0., t-2.)**2/10.; bent[:, 2] = np.sin(t)*.3
    row, _ = oracle(bent, 100., 110.)
    assert np.isfinite(row["geometry"]["one_second_speed_mae_mps"])
    for invalid in (linear[[0, 2, 3]], np.array([[100., 0., 0.], [100.25, 100., 0.]])):
        try:
            oracle(invalid, 100., float(invalid[-1, 0]))
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid accepted observations were not rejected")
    print("g31_roundtrip_construct_checks_passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("g31-data/legs/pilot-300001"))
    parser.add_argument("--output", type=Path, default=Path("g31-data/roundtrip/pilot-300001"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.data, args.output)
