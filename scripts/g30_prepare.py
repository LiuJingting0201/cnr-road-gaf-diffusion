"""Build the fixed three-source G30 cohort without cross-seed entity reuse.

Participant IDs and trajectories are written only to NPZ files. The JSON receipt
contains aggregate counts, source hashes and deterministic selection definitions.
No pilot, frozen test trajectories/labels, model prediction, or classifier score
is read. The original ID/split manifest is used to exclude test entity templates.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from g29_prepare import sha256, span, write_rows


CLASSES = ["pedestrian", "bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
SPLITS = ("train", "val")
DOMAINS = ("cnr", "sumo", "native")
EPISODES = {"train": tuple(range(300101, 300107)), "val": tuple(range(300201, 300204))}
CAPS = {"train": np.array([611, 111, 295, 751, 141, 74]),
        "val": np.array([70, 14, 37, 109, 18, 9])}
SEED = 30042
EXTRA_FIELDS = ("base_ids", "entity_tags", "vtypes", "route_hashes", "lines",
                "arrival_complete_flags", "observation_complete_flags", "depart", "arrival", "reroute_counts")


def save_json(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rank(*parts):
    return hashlib.sha256(json.dumps([SEED, *parts], ensure_ascii=True,
                                    separators=(",", ":")).encode()).digest()


def entity_partition(base_id, original_partitions=None):
    # Hash no episode, type or label. Identical source IDs remain in one split
    # even if their realised vType is redrawn by the simulator in another seed.
    # Known original entities inherit the CNR split; test must be discarded.
    if original_partitions is not None and base_id in original_partitions:
        return original_partitions[base_id]
    return "val" if int.from_bytes(rank("entity-partition", base_id), "big") < 2**256 // 5 else "train"


def original_id_partitions(path):
    # The manifest supplies identifiers and split membership only. No CNR test
    # coordinates, labels, feature arrays or model predictions are accessed.
    partitions = json.loads(path.read_text())["split_by_id"]
    if (not isinstance(partitions, dict) or not partitions
            or any(not isinstance(key, str) or not key for key in partitions)
            or not set(partitions.values()).issubset({"train", "val", "test"})):
        raise ValueError("Original split manifest has an unexpected ID/split contract")
    return partitions


def class_counts(rows):
    return np.bincount([row["label"] for row in rows], minlength=6).tolist()


def rows_from_archive(path, indices=None, native=False, episode=None):
    """Restore saved core/full arrays exactly; never choose a new segment."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("ids", "labels", "txy", "offset", "full_txy", "full_offset",
                "duration", "core_start", "core_end", "break_count", "eligible_512")}
        if native:
            data.update({key: archive[key] for key in EXTRA_FIELDS})
            for key in ("native_speed", "full_native_speed"):
                if key in archive:
                    data[key] = archive[key]
            if episode is None:
                data["episodes"] = archive["episodes"]
    identifiers = data["ids"]
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("Duplicate saved observation IDs")
    for key, array in data.items():
        if key not in ("txy", "full_txy", "offset", "full_offset", "native_speed", "full_native_speed") and len(array) != len(identifiers):
            raise ValueError(f"Misaligned observation metadata: {key}")
    for offset_key, value_key in (("offset", "txy"), ("full_offset", "full_txy")):
        offset = data[offset_key]
        if (len(offset) != len(identifiers)+1 or offset[0] != 0
                or offset[-1] != len(data[value_key]) or np.any(np.diff(offset) <= 0)):
            raise ValueError("Invalid observation offsets")
    selected = range(len(identifiers)) if indices is None else indices
    rows = []
    for i in selected:
        core = data["txy"][data["offset"][i]:data["offset"][i+1]].copy()
        full = data["full_txy"][data["full_offset"][i]:data["full_offset"][i+1]].copy()
        left, right = int(data["core_start"][i]), int(data["core_end"][i])
        label = int(data["labels"][i])
        if not 0 <= label < 6:
            raise ValueError("Source cohort includes an undeclared participant label")
        if (core.ndim != 2 or core.shape[1] != 3 or not np.isfinite(core).all()
                or not np.isfinite(full).all() or len(core) < 2
                or not np.allclose(np.diff(core[:, 0]), .25, atol=1e-7, rtol=0)
                or np.any(np.diff(full[:, 0]) <= 0)
                or not (0 <= left < right <= len(full))
                or not np.array_equal(core, full[left:right])):
            raise ValueError("Saved continuous core does not match its observation history")
        row = {"id": str(identifiers[i]), "label": label, "core": core, "full": full,
               "core_start": left, "core_end": right, "break_count": int(data["break_count"][i])}
        if (span(row) < 5 or not np.isclose(span(row), float(data["duration"][i]), atol=1e-7, rtol=0)
                or bool(data["eligible_512"][i]) != (len(core) >= 512)):
            raise ValueError("Saved duration/window eligibility disagrees with the fixed core")
        if native:
            row["extra"] = {key: data[key][i].item() for key in EXTRA_FIELDS}
            row["extra"]["episodes"] = int(data["episodes"][i]) if episode is None else int(episode)
            for field, offset, points in (("native_speed", "offset", "txy"),
                                          ("full_native_speed", "full_offset", "full_txy")):
                if field in data:
                    if data[field].shape != (len(data[points]),):
                        raise ValueError("Native speed is not aligned with saved coordinates")
                    a, b = data[offset][i:i+2]
                    row[field] = data[field][a:b].copy()
        rows.append(row)
    return rows


def native_candidates(root, original_partitions):
    all_candidates = defaultdict(list)
    sources, aggregates = {}, {}
    observed_rows = 0
    for split, episodes in EPISODES.items():
        for episode in episodes:
            folder = root / split / str(episode) / "extracted"
            manifest = json.loads((folder / "complete.json").read_text())
            path = folder / "raw.npz"
            if (manifest.get("complete") is not True or str(manifest["episode"]) != str(episode)
                    or manifest["classes"] != CLASSES or sha256(path) != manifest["raw_sha256"]):
                raise ValueError("Native extraction receipt or source hash mismatch")
            if manifest.get("source_type_label_overrides") != {"uber": 5}:
                raise ValueError("Native source-type mapping changed")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in ("ids", "base_ids", "entity_tags", "vtypes", "labels")}
            count = len(arrays["ids"])
            if count != manifest["rows"] or any(len(values) != count for values in arrays.values()):
                raise ValueError("Native header counts disagree with extraction receipt")
            if np.bincount(arrays["labels"], minlength=6).tolist() != manifest["class_counts"]:
                raise ValueError("Native header class counts disagree with extraction receipt")
            seen = set()
            for i in range(count):
                base_id = str(arrays["base_ids"][i])
                if not base_id:
                    raise ValueError("Native entity has empty base ID")
                # A base ID is the cross-seed demand grouping unit, even across
                # entity namespaces; selecting one observation remains conservative.
                identity = (str(arrays["entity_tags"][i]), base_id)
                if identity in seen:
                    raise ValueError("An extracted episode repeats an entity")
                seen.add(identity)
                all_candidates[base_id].append({"path": path, "index": i, "split": split,
                    "episode": episode, "base_id": base_id, "id": str(arrays["ids"][i]),
                    "tag": identity[0], "vtype": str(arrays["vtypes"][i]), "label": int(arrays["labels"][i])})
            observed_rows += count
            key = f"{split}/{episode}"
            sources[key] = {"raw_sha256": manifest["raw_sha256"],
                            "extraction_manifest_sha256": sha256(folder / "complete.json")}
            aggregates[key] = {key: manifest[key] for key in (
                "rows", "class_counts", "observed_time_range_s", "counts",
                "unknown_vtype_observations_excluded", "unsupported_vclass_observations_excluded",
                "arrival_complete_counts", "observation_complete_counts", "eligible_512_counts")}
    chosen, report = {split: [] for split in SPLITS}, Counter()
    for base_id, candidates in all_candidates.items():
        report["unique_base_entities"] += 1
        partition = entity_partition(base_id, original_partitions)
        if partition == "test":
            report["original_test_entities_excluded"] += 1
            report["original_test_eligible_observations_excluded"] += len(candidates)
            continue
        if base_id in original_partitions:
            report[f"entities_inheriting_cnr_{partition}_membership"] += 1
        else:
            report[f"unknown_entities_assigned_{partition}_by_hash"] += 1
        report["entities_observed_in_multiple_episodes"] += len({r["episode"] for r in candidates}) > 1
        report["entities_with_eligible_label_changes"] += len({r["label"] for r in candidates}) > 1
        report["entities_with_eligible_vtype_changes"] += len({r["vtype"] for r in candidates}) > 1
        report["entities_with_entity_tag_changes"] += len({r["tag"] for r in candidates}) > 1
        allowed = [row for row in candidates if row["split"] == partition]
        report["observations_removed_by_entity_partition"] += len(candidates) - len(allowed)
        if not allowed:
            report["entities_without_eligible_observation_in_assigned_episode_partition"] += 1
            continue
        # Rank without the label/type. Select once before counting per-class support.
        choice = min(allowed, key=lambda row: (rank("representative", base_id, row["episode"], row["tag"]),
                                               row["episode"], row["tag"], row["index"]))
        chosen[partition].append(choice)
        report["redundant_eligible_realisations_removed"] += len(allowed)-1
    report["eligible_observation_rows_before_grouping"] = observed_rows
    report["selected_entities_before_class_matching"] = sum(map(len, chosen.values()))
    report["selection_scope"] = "One hash-ranked eligible observation per base ID; no score, label, type or duration in the ranking."
    report["label_change_scope"] = "Changes among mapped eligible observations only; excluded vTypes remain counted in extraction receipts."
    report["source_identity_scope"] = "Exact base-ID matches only, shared across person/vehicle tags; no guessed aliases or prefix stripping."
    return chosen, sources, aggregates, dict(report)


def route_overlap(train, validation, key):
    left = [str(row["extra"][key]) for row in train]
    right = [str(row["extra"][key]) for row in validation]
    a, b = set(left)-{""}, set(right)-{""}
    common = a & b
    byclass = {}
    for label in range(6):
        values = [str(row["extra"][key]) for row in validation if row["label"] == label]
        observed = [value for value in values if value]
        byclass[str(label)] = {"rows": len(values), "metadata_present": len(observed),
                               "seen_in_training": sum(value in a for value in observed),
                               "not_seen_in_training": sum(value not in a for value in observed)}
    return {"train_unique_nonempty": len(a), "val_unique_nonempty": len(b), "shared_unique_nonempty": len(common),
            "train_rows_with_metadata": sum(bool(x) for x in left), "val_rows_with_metadata": sum(bool(x) for x in right),
            "val_rows_seen_in_train": sum(bool(x) and x in a for x in right), "val_by_class": byclass}


def write_cohort(path, rows, native=False):
    report = write_rows(path, rows)
    if native:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for key in (*EXTRA_FIELDS, "episodes"):
            arrays[key] = np.asarray([row["extra"][key] for row in rows])
        for key in ("native_speed", "full_native_speed"):
            if all(key in row for row in rows):
                arrays[key] = np.concatenate([row[key] for row in rows])
        temporary = path.with_name(path.stem + ".meta.tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        report["sha256"] = sha256(path)
    return report


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("*.npz")) or any((args.output / name).exists() for name in ("prepared.json", "failed.json")):
        raise FileExistsError("Preparation output already contains a cohort; do not overwrite or resample")
    encoding = json.loads(args.encoding.read_text())
    original_partitions = original_id_partitions(args.cnr_splits)
    cohorts = {domain: {} for domain in DOMAINS}
    receipt = {"scope": "Fixed three-source development diagnosis; no pilot or frozen test trajectories/labels, and no generation; original ID/split manifest used only for entity isolation",
               "classes": CLASSES, "seed": SEED, "encoding": encoding,
               "minimum_continuous_span_s": 5., "continuity": "Saved longest continuous .25-second core is retained exactly",
               "sources": {"encoding_metadata_sha256": sha256(args.encoding), "g29_raw_files": {},
                           "cnr_split_manifest_sha256": sha256(args.cnr_splits)},
               "available": {}, "count_matched": {}, "native": {},
               "label_semantics": {"native": "Declared vClass; source vType uber maps to taxi_or_uber; moped excluded",
                                   "cnr_correspondence": "Original class2 and native source taxonomy are not proven identical"},
               "selection": {"entity_partition": "Exact CNR test IDs excluded; known CNR train/val IDs inherit original membership; unknown IDs use SHA256([30042,entity-partition,base_id]) below 2^256/5 for val, otherwise train",
                             "episode_partitions": EPISODES,
                             "representative": "Lowest SHA256([30042,representative,base_id,episode,entity_tag]) within assigned episode partition",
                             "matched_sampling": "Lowest SHA256([30042,cohort,domain,split,observation_id]) separately within each class",
                             "class_caps": {key: value.tolist() for key, value in CAPS.items()}},
               "interpretation": "Common demand files across native seeds; one eligible base entity per cohort, not new OD demand."
               }
    for domain in ("cnr", "sumo"):
        for split in SPLITS:
            path = args.g29_raw / f"{domain}-{split}.npz"
            cohorts[domain][split] = rows_from_archive(path)
            if domain == "cnr" and any(original_partitions.get(row["id"]) != split for row in cohorts[domain][split]):
                raise ValueError("The existing CNR cohort disagrees with its original split manifest")
            receipt["sources"]["g29_raw_files"][f"{domain}-{split}"] = {"sha256": sha256(path)}
    chosen, source_hashes, extraction_counts, exclusions = native_candidates(args.episode_root, original_partitions)
    receipt["sources"]["native_extractions"] = source_hashes
    receipt["native"]["extraction_aggregates"] = extraction_counts
    receipt["native"]["entity_selection"] = exclusions
    for split in SPLITS:
        byfile = defaultdict(list)
        for row in chosen[split]:
            byfile[(row["path"], row["episode"])].append(row["index"])
        rows = []
        for (path, episode), indices in sorted(byfile.items(), key=lambda item: item[0][1]):
            rows.extend(rows_from_archive(path, sorted(indices), native=True, episode=episode))
        if len({row["extra"]["base_ids"] for row in rows}) != len(rows):
            raise ValueError("Native base entity was selected more than once")
        cohorts["native"][split] = rows
    for domain in DOMAINS:
        if set(row["id"] for row in cohorts[domain]["train"]) & set(row["id"] for row in cohorts[domain]["val"]):
            raise ValueError("Source observation IDs cross the fixed train/val partitions")
        for split in SPLITS:
            rows = cohorts[domain][split]
            receipt["available"][f"{domain}-{split}"] = {"rows": len(rows), "class_counts": class_counts(rows)}
    train_bases = {row["extra"]["base_ids"] for row in cohorts["native"]["train"]}
    val_bases = {row["extra"]["base_ids"] for row in cohorts["native"]["val"]}
    if train_bases & val_bases:
        raise ValueError("Native base entities cross train/validation partitions")
    receipt["native"]["base_entity_overlap"] = 0
    original_sets = {split: {key for key, value in original_partitions.items() if value == split}
                     for split in ("train", "val", "test")}
    overlap = {"native_train_with_cnr_val": len(train_bases & original_sets["val"]),
               "native_val_with_cnr_train": len(val_bases & original_sets["train"]),
               "native_any_with_cnr_test": len((train_bases | val_bases) & original_sets["test"])}
    if any(overlap.values()):
        raise ValueError("Native selection violates the original CNR entity split")
    receipt["native"]["cross_source_excluded_partition_overlap"] = overlap
    receipt["native"]["cross_source_same_partition_overlap"] = {
        "native_train_with_cnr_train": len(train_bases & original_sets["train"]),
        "native_val_with_cnr_val": len(val_bases & original_sets["val"])}
    receipt["native"]["independence_limit"] = "Same-partition shared base IDs and route/line families are correlated demand templates; native versus CNR is not an independent-entity replication."
    counts, missing = {}, []
    for split in SPLITS:
        counts[split] = np.minimum(CAPS[split], np.min([class_counts(cohorts[domain][split]) for domain in DOMAINS], axis=0))
        for label in np.flatnonzero(counts[split] == 0):
            missing.append({"split": split, "class": CLASSES[label], "available_by_source": {
                domain: class_counts(cohorts[domain][split])[label] for domain in DOMAINS}})
    if missing:
        receipt["complete"] = False
        receipt["failure"] = "A declared common class has no eligible support; no class substitution or fitting is allowed"
        receipt["missing_class_support"] = missing
        save_json(args.output / "failed.json", receipt)
        raise ValueError("A declared common class lacks support; aggregate failure receipt saved")
    matched = {domain: {} for domain in DOMAINS}
    for domain in DOMAINS:
        for split in SPLITS:
            rows = cohorts[domain][split]
            selected = set()
            for label, count in enumerate(counts[split]):
                candidates = [row for row in rows if row["label"] == label]
                ordered = sorted(candidates, key=lambda row: (rank("cohort", domain, split, row["id"]), row["id"]))
                selected.update(row["id"] for row in ordered[:int(count)])
            # Keep source ordering; an unchanged G29 population remains exactly
            # compatible with its saved input order as well as its ID set.
            matched[domain][split] = [row for row in rows if row["id"] in selected]
            receipt["count_matched"][f"{domain}-{split}"] = write_cohort(
                args.output / f"{domain}-{split}.npz", matched[domain][split], native=domain == "native")
    receipt["matched_class_counts"] = {split: counts[split].tolist() for split in SPLITS}
    receipt["native"]["route_metadata_scope"] = "Final emitted route fingerprints and lines for audits only; not pre-simulation planned routes or generation conditions"
    receipt["native"]["matched_route_overlap"] = {key: route_overlap(matched["native"]["train"], matched["native"]["val"], key)
                                                  for key in ("route_hashes", "lines")}
    receipt["native"]["available_route_overlap"] = {key: route_overlap(cohorts["native"]["train"], cohorts["native"]["val"], key)
                                                    for key in ("route_hashes", "lines")}
    receipt["g29_population_unchanged"] = {domain: {split: len(matched[domain][split]) == len(cohorts[domain][split])
                                                  for split in SPLITS} for domain in ("cnr", "sumo")}
    receipt["complete"] = True
    receipt["script_sha256"] = sha256(Path(__file__))
    save_json(args.output / "prepared.json", receipt)
    print(json.dumps({"complete": True, "class_counts": receipt["matched_class_counts"],
                      "native_entity_overlap": 0, "output": str(args.output)}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", type=Path, default=Path("g30-data/native"))
    parser.add_argument("--g29-raw", type=Path, default=Path("../g29-job/g29-data/raw"))
    parser.add_argument("--cnr-splits", type=Path, default=Path("../data/processed/most64/dataset.json"))
    parser.add_argument("--encoding", type=Path, default=Path("../data/processed/original-hilbert64grid224/dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("g30-data/raw"))
    prepare(parser.parse_args())
