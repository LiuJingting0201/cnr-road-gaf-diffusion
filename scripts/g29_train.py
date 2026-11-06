"""Fixed-budget source/encoding diagnosis, with no validation-based model choice."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
from cnr.representations import SequenceClassifier

ARMS = ("xy", "grid_xy", "native_h", "squeezed_h", "unit_h", "gaf")


def metrics(truth, probability):
    predicted = probability.argmax(1)
    confusion = np.bincount(6*truth + predicted, minlength=36).reshape(6, 6)
    f1 = 2*np.diag(confusion)/np.maximum(confusion.sum(0)+confusion.sum(1), 1)
    return {"accuracy": float(np.mean(predicted == truth)), "macro_f1": float(f1.mean()),
            "per_class_f1": f1.tolist(),
            "per_class_recall": (np.diag(confusion)/np.maximum(confusion.sum(1), 1)).tolist(),
            "confusion": confusion.tolist(), "rows": len(truth)}


def network(arm, pretrained):
    if arm != "gaf":
        return SequenceClassifier(2).cuda()
    model = resnet18(weights=None)
    # Preserve E1's initialization order: ImageNet1000-class model, then new6-class head.
    model.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True))
    model.fc = nn.Linear(512, 6)
    return model.cuda()


def data(root, domain, view, split, arm):
    directory = root/domain/view/split
    if not (directory/"complete.json").exists():
        raise RuntimeError(f"Encoding incomplete: {directory}")
    with np.load(directory/"features.npz") as file:
        rows = {"ids": file["ids"], "labels": file["labels"]}
        if arm != "gaf":
            rows["input"] = file[arm]
    if arm == "gaf":
        rows["input"] = np.load(directory/"images.npy", mmap_mode="r")
    return rows


def features(values, arm, mean, standard):
    x = torch.as_tensor(np.asarray(values), device="cuda").float()
    if arm == "gaf":
        x = x.permute(0, 3, 1, 2)/255
    return (x-mean)/standard


@torch.no_grad()
def evaluate(model, rows, arm, mean, standard):
    model.eval()
    probabilities = []
    for start in range(0, len(rows["ids"]), 64):
        x = features(rows["input"][start:start+64], arm, mean, standard)
        probabilities.append(model(x).softmax(1).cpu().numpy())
    probability = np.concatenate(probabilities)
    return metrics(rows["labels"], probability), probability


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    name = f"{args.domain}-{args.view}-{args.arm}-{args.seed}"
    output = args.output/name
    output.mkdir(parents=True, exist_ok=True)
    if (output/"complete.json").exists():
        return
    if not (args.encoded/"complete.json").exists():
        raise RuntimeError("Matched-source preparation incomplete")
    train = data(args.encoded, args.domain, args.view, "train", args.arm)
    validation = data(args.encoded, args.domain, args.view, "val", args.arm)
    if set(train["ids"]) & set(validation["ids"]):
        raise ValueError("Train/validation ID overlap")
    if np.unique(train["labels"]).tolist() != list(range(6)):
        raise ValueError("All six classes must be represented in training")
    if args.arm == "gaf":
        mean_np = np.array([.485, .456, .406])[None, :, None, None]
        std_np = np.array([.229, .224, .225])[None, :, None, None]
    else:
        mean_np = train["input"].mean(axis=(0, 2), keepdims=True)
        std_np = train["input"].std(axis=(0, 2), keepdims=True)
        # The padded zero channel stays zero with a unit denominator.
        std_np = np.where(std_np > 1e-6, std_np, 1.)
    mean = torch.tensor(mean_np, dtype=torch.float32, device="cuda")
    standard = torch.tensor(std_np, dtype=torch.float32, device="cuda")
    model = network(args.arm, args.pretrained)
    labels = torch.as_tensor(train["labels"], device="cuda")
    counts = torch.bincount(labels, minlength=6).float()
    weight = counts.rsqrt()
    weight /= weight.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1000)
    config = {"domain": args.domain, "view": args.view, "arm": args.arm, "seed": args.seed,
              "epochs": 50, "batch_size": 64, "learning_rate": 2e-4, "weight_decay": 1e-4,
              "selection": "fixed final epoch50, no monitor-based selection",
              "class_weights": weight.cpu().tolist(), "train_counts": counts.int().cpu().tolist(),
              "train_rows": len(train["ids"]), "validation_rows": len(validation["ids"]),
              "normalization_mean": mean_np.tolist(), "normalization_std": std_np.tolist(),
              "input_scope": "ordered positions or position-derived Hilbert encodings only; no explicit velocity, acceleration or duration",
              "backbone": "ImageNet ResNet18" if args.arm == "gaf" else "same two-channel SequenceClassifier for all1D arms"}
    (output/"config.json").write_text(json.dumps(config, indent=2))
    begin = time.monotonic()
    with (output/"history.jsonl").open("w", buffering=1) as stream:
        for epoch in range(1, 51):
            order = torch.randperm(len(labels), device="cuda", generator=generator)
            loss_sum = 0.
            model.train()
            for indices in order.split(64):
                positions = indices.cpu().numpy()
                x = features(train["input"][positions], args.arm, mean, standard)
                loss = nn.functional.cross_entropy(model(x), labels[indices], weight=weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite classifier loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                loss_sum += float(loss.detach())*len(indices)
            if epoch == 1 or epoch % 10 == 0:
                train_score, _ = evaluate(model, train, args.arm, mean, standard)
                val_score, _ = evaluate(model, validation, args.arm, mean, standard)
                record = {"epoch": epoch, "seconds": time.monotonic()-begin,
                          "train_weighted_ce": loss_sum/len(labels),
                          "train": train_score, "validation": val_score}
                stream.write(json.dumps(record)+"\n")
                print(name, epoch, train_score["macro_f1"], val_score["macro_f1"], flush=True)
    train_score, train_probability = evaluate(model, train, args.arm, mean, standard)
    validation_score, val_probability = evaluate(model, validation, args.arm, mean, standard)
    for split, rows, probability in [("train", train, train_probability),
                                     ("val", validation, val_probability)]:
        np.savez_compressed(output/(split+".npz"), ids=rows["ids"], labels=rows["labels"], probabilities=probability)
    torch.save(model.state_dict(), output/"model.tmp")
    (output/"model.tmp").replace(output/"model.pt")
    cross = {}
    for domain in ("cnr", "sumo"):
        if domain == args.domain:
            continue
        alternative = data(args.encoded, domain, args.view, "val", args.arm)
        score, probability = evaluate(model, alternative, args.arm, mean, standard)
        cross[domain] = score
        np.savez_compressed(output/f"cross-{domain}.npz", ids=alternative["ids"],
                            labels=alternative["labels"], probabilities=probability)
    summary = {"complete": True, "epochs": 50, "source": args.domain, "view": args.view,
               "arm": args.arm, "seed": args.seed, "train": train_score, "validation": validation_score,
               "cross_domain_validation": cross, "seconds": time.monotonic()-begin,
               "scope": "diagnostic only, fixed training/validation partitions, no generation or thesis claims"}
    temporary = output/"complete.tmp.json"
    temporary.write_text(json.dumps(summary, indent=2))
    temporary.replace(output/"complete.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=("cnr", "sumo", "sumo-duration"))
    parser.add_argument("view", choices=("full", "window512"))
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("seed", type=int, choices=(42, 43))
    parser.add_argument("--encoded", type=Path, default=Path("g29-data/encoded"))
    parser.add_argument("--output", type=Path, default=Path("g29-data/models"))
    parser.add_argument("--pretrained", type=Path, default=Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    train(parser.parse_args())
