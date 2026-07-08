# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Train + evaluate a DINOv3 + UPerNet damage-segmentation model on xView2 (xBD).

The xView2 ``targets/*_post_disaster_target.png`` masks label every pixel with
``0`` background, ``1`` no-damage, ``2`` minor-damage, ``3`` major-damage, ``4``
destroyed. This script collapses the four damage grades into a single
``damaged`` class in one of three ways (``--grouping``), producing a 3-class
segmentation problem ``{0: background, 1: undamaged, 2: damaged}``:

* ``any``       damaged = minor + major + destroyed   (undamaged = no-damage)
* ``major``     damaged = major + destroyed           (undamaged = no-damage + minor)
* ``destroyed`` damaged = destroyed                   (undamaged = no-damage + minor + major)

It trains the repo's :class:`bda.trainers.CustomSemanticSegmentationTask` with
``model="upernet"`` (so the checkpoint loads directly in ``inference.py``) on the
train folder and reports pixel IoU / precision / recall / F1 per class on the
**full** test folder.

Example::

    python scripts/train_eval_xview2_dinov3_upernet.py \
        --xview2-root ~/data/XView2 --grouping any --gpu 0 \
        --output-dir outputs/xview2_dinov3_upernet_any
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bda.trainers import CustomSemanticSegmentationTask  # noqa: E402

# xView2 damage grade -> {0 background, 1 undamaged, 2 damaged} for each grouping.
GROUPINGS = {
    "any": [0, 1, 2, 2, 2],
    "major": [0, 1, 1, 2, 2],
    "destroyed": [0, 1, 1, 1, 2],
}
CLASS_NAMES = ["background", "undamaged", "damaged"]
NUM_CLASSES = 3

# ImageNet normalization (DINOv3 backbones expect it); imagery is 8-bit RGB.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def _post_images(root: str, split: str) -> list[str]:
    fns = sorted(glob.glob(os.path.join(root, split, "images", "*_post_disaster.png")))
    if not fns:
        raise SystemExit(f"No post-disaster images under {root}/{split}/images/.")
    return fns


def _target_for(image_fn: str) -> str:
    d = os.path.dirname(os.path.dirname(image_fn))
    base = os.path.basename(image_fn).replace(".png", "_target.png")
    return os.path.join(d, "targets", base)


class XView2SegDataset(Dataset):
    """Post-disaster RGB images + remapped damage masks.

    Training samples random ``crop_size`` windows (biased toward building pixels);
    evaluation returns the full image so the whole test footprint is scored.
    """

    def __init__(
        self,
        image_fns: list[str],
        grouping: str,
        crop_size: int | None = 512,
        crops_per_image: int = 1,
        train: bool = True,
        seed: int = 0,
    ) -> None:
        self.image_fns = image_fns
        self.lut = np.array(GROUPINGS[grouping], dtype=np.uint8)
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image if train else 1
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.image_fns) * self.crops_per_image

    def _load(self, idx: int):
        image_fn = self.image_fns[idx % len(self.image_fns)]
        image = np.asarray(Image.open(image_fn).convert("RGB"), dtype=np.float32)
        mask = np.asarray(Image.open(_target_for(image_fn)))
        mask = self.lut[np.clip(mask, 0, 4)].astype(np.int64)
        return image, mask

    def _random_crop(self, image: np.ndarray, mask: np.ndarray):
        h, w = mask.shape
        cs = self.crop_size
        if h <= cs or w <= cs:
            return image, mask
        # Bias 80% of crops toward a window that contains building pixels.
        best = None
        for attempt in range(10):
            y = self.rng.randint(0, h - cs)
            x = self.rng.randint(0, w - cs)
            m = mask[y : y + cs, x : x + cs]
            if attempt == 0 or self.rng.random() > 0.8 or (m > 0).any():
                return image[y : y + cs, x : x + cs], m
            best = (image[y : y + cs, x : x + cs], m)
        return best

    def __getitem__(self, idx: int):
        image, mask = self._load(idx)
        if self.train and self.crop_size is not None:
            image, mask = self._random_crop(image, mask)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask = torch.from_numpy(mask.copy()).long()
        return {"image": image, "mask": mask}


def compute_class_weights(image_fns: list[str], grouping: str, sample: int = 300):
    """Inverse-sqrt-frequency class weights (normalized to mean 1) for weighted CE."""
    lut = np.array(GROUPINGS[grouping], dtype=np.uint8)
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    rng = random.Random(0)
    fns = image_fns if len(image_fns) <= sample else rng.sample(image_fns, sample)
    for fn in fns:
        m = lut[np.clip(np.asarray(Image.open(_target_for(fn))), 0, 4)]
        b = np.bincount(m.reshape(-1), minlength=NUM_CLASSES)
        counts += b[:NUM_CLASSES]
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(freq + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), counts


@torch.inference_mode()
def evaluate(task, image_fns: list[str], grouping: str, device, batch_size: int = 2):
    """Accumulate a 3x3 confusion matrix over the full images -> per-class metrics."""
    ds = XView2SegDataset(image_fns, grouping, crop_size=None, train=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=8)
    model = task.model.eval().to(device)
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].numpy().reshape(-1)
        pred = model(x).argmax(1).cpu().numpy().reshape(-1)
        conf += np.bincount(
            y * NUM_CLASSES + pred, minlength=NUM_CLASSES**2
        ).reshape(NUM_CLASSES, NUM_CLASSES)
    metrics = {"confusion_matrix": conf.tolist(), "per_class": {}}
    ious, f1s = [], []
    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
        metrics["per_class"][CLASS_NAMES[c]] = {
            "iou": float(iou), "precision": float(prec),
            "recall": float(rec), "f1": float(f1),
            "support": int(conf[c, :].sum()),
        }
        ious.append(iou)
        f1s.append(f1)
    metrics["mean_iou"] = float(np.nanmean(ious))
    metrics["damaged_f1"] = metrics["per_class"]["damaged"]["f1"]
    metrics["overall_accuracy"] = float(np.trace(conf) / conf.sum())
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xview2-root", default=os.path.expanduser("~/data/XView2"))
    p.add_argument("--grouping", choices=list(GROUPINGS), required=True)
    p.add_argument("--backbone", default="dinov3_vits16")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--crops-per-image", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap #images per split (smoke-testing only).")
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_fns = _post_images(args.xview2_root, "train")
    rng = random.Random(args.seed)
    rng.shuffle(train_fns)
    n_val = int(len(train_fns) * args.val_fraction)
    val_fns, tr_fns = train_fns[:n_val], train_fns[n_val:]
    test_fns = _post_images(args.xview2_root, "test")
    if args.limit:
        tr_fns, val_fns, test_fns = (
            tr_fns[: args.limit], val_fns[: max(2, args.limit // 5)],
            test_fns[: args.limit],
        )
    print(
        f"grouping={args.grouping}  train={len(tr_fns)} val={len(val_fns)} "
        f"test={len(test_fns)} images"
    )

    class_weights, counts = compute_class_weights(tr_fns, args.grouping)
    print(f"class pixel counts {counts.astype(int)}  weights {class_weights.numpy()}")

    train_ds = XView2SegDataset(
        tr_fns, args.grouping, crop_size=args.crop_size,
        crops_per_image=args.crops_per_image, train=True, seed=args.seed,
    )
    val_ds = XView2SegDataset(
        val_fns, args.grouping, crop_size=args.crop_size, train=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, num_workers=args.num_workers,
    )

    task = CustomSemanticSegmentationTask(
        model="upernet",
        backbone=args.backbone,
        weights=True,  # pretrained DINOv3 backbone
        in_channels=3,
        num_classes=NUM_CLASSES,
        loss="ce",
        class_weights=class_weights,
        ignore_index=255,  # nothing ignored; background (0) is a real class
        lr=args.lr,
        patience=5,
    )

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss", dirpath=ckpt_dir, save_top_k=1, save_last=True,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=[args.gpu],
        precision="16-mixed",
        callbacks=[checkpoint_cb],
        logger=pl.loggers.CSVLogger(args.output_dir, name="logs"),
        log_every_n_steps=20,
    )
    tic = time.time()
    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"trained in {(time.time()-tic)/60:.1f} min; best={checkpoint_cb.best_model_path}")

    best = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
    eval_task = CustomSemanticSegmentationTask.load_from_checkpoint(best, map_location="cpu")
    metrics = evaluate(eval_task, test_fns, args.grouping, device)
    metrics["grouping"] = args.grouping
    metrics["backbone"] = args.backbone
    metrics["checkpoint"] = best
    metrics["n_test_images"] = len(test_fns)
    out_json = os.path.join(args.output_dir, "test_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n== TEST metrics ({args.grouping}) ==")
    for c in CLASS_NAMES:
        m = metrics["per_class"][c]
        print(f"  {c:11s} IoU={m['iou']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"  mIoU={metrics['mean_iou']:.3f}  damaged_F1={metrics['damaged_f1']:.3f} "
          f"OA={metrics['overall_accuracy']:.3f}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
