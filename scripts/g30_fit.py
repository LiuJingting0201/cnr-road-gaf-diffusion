"""Fixed source comparison, reusing G29 encoding/training without changing its runs."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from g29_encode import read_raw, write_case, digest
from g29_train import train, data, network, evaluate

ROOT = Path("g30-data")
DOMAINS = ("cnr", "sumo", "native")


def encode():
    root = ROOT/"encoded"
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = Path("../data/processed/original-hilbert64grid224/dataset.json")
    metadata = json.loads(metadata_path.read_text())
    hashes = {f"{domain}-{split}": digest(ROOT/"raw"/f"{domain}-{split}.npz")
              for domain in DOMAINS for split in ("train", "val")}
    if (root/"complete.json").exists():
        previous = json.loads((root/"complete.json").read_text())
        assert previous["raw_sha256"] == hashes and previous["metadata_sha256"] == digest(metadata_path)
        return
    reports = {}
    for domain in DOMAINS:
        for split in ("train", "val"):
            rows = read_raw(ROOT/"raw"/f"{domain}-{split}.npz")
            key = f"{domain}/full/{split}"
            reports[key] = write_case(root/key, rows, np.arange(len(rows["ids"])), "full", metadata, 6)
    result = {"complete": True, "raw_sha256": hashes, "metadata_sha256": digest(metadata_path),
              "scope": "G30 matched six-class continuous observations, inherited frozen G29 encoding",
              "reports": reports, "frozen_metadata": {k: metadata[k] for k in
                  ("xy_min", "xy_max", "hilbert_eps", "channel_min", "channel_max")}}
    (root/"complete.tmp.json").write_text(json.dumps(result, indent=2))
    (root/"complete.tmp.json").replace(root/"complete.json")


def fit(domain, seed):
    prepared = json.loads((ROOT/"raw/prepared.json").read_text())
    unchanged = prepared["g29_population_unchanged"].get(domain, {})
    output = ROOT/"models"/f"{domain}-full-gaf-{seed}"
    if unchanged and all(unchanged.values()):
        previous = Path("../g29-job/g29-data")
        for split in ("train", "val"):
            current = ROOT/"encoded"/domain/"full"/split
            old = previous/"encoded"/domain/"full"/split
            with np.load(current/"features.npz") as a, np.load(old/"features.npz") as b:
                assert np.array_equal(a["ids"], b["ids"]) and np.array_equal(a["labels"], b["labels"])
            assert digest(current/"images.npy") == digest(old/"images.npy")
        model = (previous/"models"/output.name).resolve()
        assert (model/"complete.json").exists()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            assert output.resolve() == model
        else:
            output.symlink_to(model, target_is_directory=True)
        (output.parent/f"reused-{domain}-{seed}.json").write_text(json.dumps({
            "reused": True, "source": str(model), "identical_ids_labels_and_image_bytes": True}))
        return
    args = SimpleNamespace(domain=domain, seed=seed, view="full", arm="gaf", encoded=ROOT/"encoded",
                           output=ROOT/"models", pretrained=Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    # G29 implementation accepts native through the programmatic interface.
    # Its fixed cross-domain loop already covers cnr/sumo for native fits.
    train(args)


@torch.no_grad()
def transfer():
    torch.set_num_threads(2)
    result = {"complete": True, "scope": "fixed final-epoch scores; frozen G29 CNR teacher and new G30 models",
              "models": {}}
    jobs = [(f"{source}-{seed}", ROOT/"models"/f"{source}-full-gaf-{seed}")
            for source in DOMAINS for seed in (42, 43)]
    jobs += [(f"frozen-g29-cnr-{seed}", Path("../g29-job/g29-data/models")/f"cnr-full-gaf-{seed}")
             for seed in (42, 43)]
    out = ROOT/"predictions"
    out.mkdir(exist_ok=True)
    for name, folder in jobs:
        if not (folder/"complete.json").exists():
            raise RuntimeError(f"Incomplete fit: {name}")
        config = json.loads((folder/"config.json").read_text())
        model = network("gaf", Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
        model.load_state_dict(torch.load(folder/"model.pt", map_location="cuda", weights_only=True))
        mean = torch.tensor(config["normalization_mean"], dtype=torch.float32, device="cuda")
        standard = torch.tensor(config["normalization_std"], dtype=torch.float32, device="cuda")
        result["models"][name] = {}
        for domain in DOMAINS:
            rows = data(ROOT/"encoded", domain, "full", "val", "gaf")
            scores, probability = evaluate(model, rows, "gaf", mean, standard)
            result["models"][name][domain] = scores
            np.savez_compressed(out/f"{name}-on-{domain}.npz", ids=rows["ids"], labels=rows["labels"], probabilities=probability)
        del model
    (ROOT/"transfer.tmp.json").write_text(json.dumps(result, indent=2))
    (ROOT/"transfer.tmp.json").replace(ROOT/"transfer.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("encode", "fit", "transfer"))
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--seed", type=int, choices=(42, 43))
    args = parser.parse_args()
    if args.action == "encode":
        encode()
    elif args.action == "fit":
        assert args.domain is not None and args.seed is not None
        fit(args.domain, args.seed)
    else:
        transfer()
