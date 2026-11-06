"""Fixed train-only class probes on matched G28 representations.

Monitor episodes are development data. These exploratory probes do not select a
regularizer, a teacher checkpoint, or a generation policy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from g22_diffusion import load

MOTOR = np.array([1, 2, 3, 5, 6, 7])


def read(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def align(data, ids):
    if len(np.unique(data["ids"])) != len(data["ids"]):
        raise ValueError("Duplicate input IDs")
    index = {str(ident): i for i, ident in enumerate(data["ids"])}
    return np.array([index[str(ident)] for ident in ids])


def route_features(data):
    path = data["paths"].astype(float)
    segments = np.diff(path, axis=1)
    headings = np.arctan2(segments[:, :, 1], segments[:, :, 0])
    turns = np.abs(np.angle(np.exp(1j * np.diff(headings, axis=1))))
    straight = np.linalg.norm(path[:, -1] - path[:, 0], axis=1)
    return np.c_[
        np.log(np.maximum(data["length"], 1)),
        path[:, 0], path[:, -1],
        turns.sum(1), np.quantile(turns, .9, axis=1),
        straight / np.maximum(data["length"], 1),
    ]


def paired_inputs(root, split):
    spatial = read(root / "teachers" / "spatial-42" / (split + ".npz"))
    speed = read(root / "teachers" / "speed-42" / (split + ".npz"))
    ids, labels = spatial["ids"], spatial["labels"]
    si = align(speed, ids)
    if not np.array_equal(labels, speed["labels"][si]):
        raise ValueError("Teacher labels differ")
    trips = load(split)
    ti = align(trips, ids)
    if not np.array_equal(labels, trips["labels"][ti]):
        raise ValueError("Trip labels differ from teacher labels")
    physical = read(Path("../g23-job/g23-data") / (split + ".npz"))
    pi = align(physical, ids)
    if not np.array_equal(labels, physical["participant"][pi]):
        raise ValueError("Physical labels differ from teacher labels")
    arrays = dict(
        spatial=spatial["features"], speed=speed["features"][si],
        route=route_features(trips)[ti], physical=physical["stats"][pi],
    )
    return ids, labels, arrays


def project(train, monitor):
    # Separate PCA maps prevent a high-variance branch from dominating the joint input.
    count = min(64, train.shape[1], len(train) - 1)
    pca = PCA(n_components=count, svd_solver="full").fit(train)
    return pca.transform(train), pca.transform(monitor), count


def metrics(confusions, counts):
    cm = np.einsum("be,eij->bij", counts, confusions)
    diagonal = np.diagonal(cm, axis1=1, axis2=2)
    f1 = 2 * diagonal / np.maximum(cm.sum(1) + cm.sum(2), 1)
    return np.c_[diagonal.sum(1) / np.maximum(cm.sum((1, 2)), 1), f1.mean(1)]


def summary(curves):
    return {
        name: dict(estimate=float(curves[0, j]), ci=np.quantile(curves[1:, j], [.025, .975]).tolist())
        for j, name in enumerate(("accuracy", "macro_f1"))
    }


def fit_group(train_ids, labels, train, monitor_ids, truth, monitor, classes):
    use = np.isin(labels, classes)
    evaluate = np.isin(truth, classes)
    labels, truth = labels[use], truth[evaluate]
    train = {key: values[use] for key, values in train.items()}
    monitor = {key: values[evaluate] for key, values in monitor.items()}
    dimensions = {}
    for key in ("spatial", "speed"):
        train[key], monitor[key], dimensions[key] = project(train[key], monitor[key])
    train["joint"] = np.c_[train["spatial"], train["speed"]]
    monitor["joint"] = np.c_[monitor["spatial"], monitor["speed"]]
    train["route_physical"] = np.c_[train["route"], train["physical"]]
    monitor["route_physical"] = np.c_[monitor["route"], monitor["physical"]]
    episodes, episode = np.unique([str(i).split("_")[0] for i in monitor_ids[evaluate]], return_inverse=True)
    n = len(episodes)
    counts = np.vstack([np.ones(n), np.random.default_rng(28041).multinomial(n, np.ones(n) / n, 2000)])
    result = dict(train_ids=len(labels), monitor_ids=len(truth), monitor_episodes=episodes.tolist(),
                  classes=classes.tolist(), pca_dimensions=dimensions, methods={})
    curves = {}
    for name in train:
        scaler = StandardScaler().fit(train[name])
        classifier = LogisticRegression(C=1., class_weight="balanced", solver="lbfgs", max_iter=3000)
        classifier.fit(scaler.transform(train[name]), labels)
        prediction = classifier.predict(scaler.transform(monitor[name]))
        y = np.searchsorted(classes, truth)
        p = np.searchsorted(classes, prediction)
        confusion = np.bincount(
            episode * len(classes) ** 2 + y * len(classes) + p,
            minlength=n * len(classes) ** 2,
        ).reshape(n, len(classes), len(classes))
        curves[name] = metrics(confusion, counts)
        result["methods"][name] = dict(
            dimension=train[name].shape[1], scores=summary(curves[name]),
            confusion=confusion.sum(0).tolist(), iterations=classifier.n_iter_.tolist(),
        )
    result["paired_differences"] = {
        "joint-" + name: summary(curves["joint"] - curves[name]) for name in ("spatial", "speed")
    }
    return result


def main(root):
    train_ids, labels, train = paired_inputs(root, "train")
    monitor_ids, truth, monitor = paired_inputs(root, "monitor")
    if set(train_ids).intersection(monitor_ids):
        raise ValueError("Training and monitor IDs overlap")
    result = dict(
        protocol="C=1 fixed; PCA and scaling fit on training only; no monitor tuning",
        interpretation="Exploratory three-episode monitor bootstrap conditional on fitted seed-42 teachers; joint input has twice the latent dimension; classification does not establish improved generation",
        physical_features=["mean_one_second_speed_mps", "mean_absolute_four_second_speed_change_mps"],
        route_features=["log_route_length", "start_x", "start_y", "end_x", "end_y", "total_absolute_turn", "p90_absolute_turn", "endpoint_distance_over_length"],
        full8=fit_group(train_ids, labels, train, monitor_ids, truth, monitor, np.arange(8)),
        motor6=fit_group(train_ids, labels, train, monitor_ids, truth, monitor, MOTOR),
    )
    path = root / "probes.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(path)
    print(json.dumps({"output": str(path)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("g28-data"))
    main(parser.parse_args().root)
