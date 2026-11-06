"""Fit matched spatial-only and speed-only eight-class representation teachers."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from g23_modes import speed_images


def classifier(seed, pretrained=None):
    torch.manual_seed(seed)
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 8)
    if pretrained is not None:
        state = torch.load(pretrained, weights_only=True, map_location="cpu")
        del state["fc.weight"], state["fc.bias"]
        model.load_state_dict(state, strict=False)
    return model.cuda()


def image_inputs(view, inputs):
    values = torch.as_tensor(inputs, device="cuda")
    if view == "spatial":
        image = values.float().permute(0, 3, 1, 2) / 255
    else:
        image = speed_images(values.float())
        image[:, :2] = (image[:, :2] + 1) / 2
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    return (image - mean) / std


def metrics(labels, logits):
    predicted = logits.argmax(1)
    matrix = np.bincount(8 * labels + predicted, minlength=64).reshape(8, 8)
    f1 = 2 * matrix.diagonal() / np.maximum(matrix.sum(0) + matrix.sum(1), 1)
    motorized = [1, 2, 3, 5, 6, 7]
    motor_matrix = matrix.copy()
    motor_matrix[[0, 4]] = 0
    motor_f1 = (2 * motor_matrix.diagonal()
                / np.maximum(motor_matrix.sum(0) + motor_matrix.sum(1), 1))
    return {"accuracy": float(np.mean(predicted == labels)), "macro_f1": float(f1.mean()),
            "per_class_f1": f1.tolist(), "confusion": matrix.tolist(),
            "per_class_recall": (matrix.diagonal() / np.maximum(matrix.sum(1), 1)).tolist(),
            "six_motorized_macro_f1": float(motor_f1[motorized].mean())}


@torch.no_grad()
def predictions(model, view, values):
    model.eval()
    return np.concatenate([model(image_inputs(view, values[start:start + 64])).cpu().numpy()
                           for start in range(0, len(values), 64)])


@torch.no_grad()
def extract(model, view, data, values, out):
    model.eval()
    head = model.fc
    model.fc = nn.Identity()
    vectors, logits = [], []
    for start in range(0, len(values), 64):
        feature = model(image_inputs(view, values[start:start + 64]))
        vectors.append(feature.cpu().numpy())
        logits.append(head(feature).cpu().numpy())
    model.fc = head
    features = np.concatenate(vectors)
    prediction = np.concatenate(logits)
    np.savez_compressed(out, ids=data["ids"], labels=data["labels"],
                        features=features, logits=prediction)
    return features, metrics(data["labels"], prediction)


def train(args):
    torch.set_num_threads(2)
    out = args.output / f"{args.view}-{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    prepared = json.loads((args.cache / "prepared.json").read_text())
    data = {split: dict(np.load(args.cache / f"{split}.npz")) for split in ("train", "monitor")}
    values = {split: np.load(args.cache / f"{split}-images.npy", mmap_mode="r")
              if args.view == "spatial" else rows["velocity"] for split, rows in data.items()}
    if not all(rows["valid"].all() for rows in data.values()):
        raise ValueError("Both teachers must use the same fully encoded IDs")
    model = classifier(args.seed, args.pretrained)
    beginning = time.monotonic()
    if not (out / "fit.json").exists():
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 18, eta_min=2e-5)
        generator = np.random.default_rng(args.seed + 28000)
        labels = torch.as_tensor(data["train"]["labels"], device="cuda")
        best, chosen, history = -1., 0, []
        with (out / "history.jsonl").open("w", buffering=1) as stream:
            for epoch in range(1, 19):
                model.train()
                order = generator.permutation(len(labels))
                for start in range(0, len(order), 64):
                    indices = order[start:start + 64]
                    logits = model(image_inputs(args.view, values["train"][indices]))
                    loss = F.cross_entropy(logits, labels[indices], label_smoothing=.02)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite teacher loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                result = metrics(data["monitor"]["labels"], predictions(model, args.view, values["monitor"]))
                result.update(epoch=epoch, seconds=time.monotonic() - beginning)
                history.append(result)
                if result["macro_f1"] > best:
                    best, chosen = result["macro_f1"], epoch
                    torch.save(model.state_dict(), out / "model.tmp")
                    (out / "model.tmp").replace(out / "model.pt")
                stream.write(json.dumps(result) + "\n")
                print(args.view, args.seed, epoch, result["accuracy"], result["macro_f1"], flush=True)
        fit = {"view": args.view, "seed": args.seed, "epochs": 18, "selected_epoch": chosen,
               "selection": "maximum monitor macro F1, earliest tie", "history": history,
               "preparation": prepared, "spatial_no_speed_input": True,
               "speed_only_input": True, "labels": "eight participant classes",
               "speed_image_input": "128x128 float GASF/GADF/MTF resized to 224, shared ImageNet normalization"}
        (out / "fit.json").write_text(json.dumps(fit, indent=2))
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cuda"))
    result = {}
    for split in ("train", "monitor"):
        features, score = extract(model, args.view, data[split], values[split], out / f"{split}.npz")
        result[split] = score
        if split == "train":
            # Only training features determine the reusable latent normalization.
            np.savez(out / "normalizer.npz", mean=features.mean(0),
                     std=np.maximum(features.std(0), 1e-4))
    result.update(view=args.view, seed=args.seed, seconds=time.monotonic() - beginning,
                  scope="train and monitor only, no evaluation observations")
    (out / "complete.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("view", choices=("spatial", "speed"))
    parser.add_argument("seed", type=int, choices=(42, 43))
    parser.add_argument("--cache", type=Path, default=Path("g28-data/cache"))
    parser.add_argument("--output", type=Path, default=Path("g28-data/teachers"))
    parser.add_argument("--pretrained", type=Path,
                        default=Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    train(parser.parse_args())
