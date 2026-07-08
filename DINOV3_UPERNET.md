# DINOv3 + UPerNet segmentation model

This repo can train and run a [DINOv3](https://ai.meta.com/dinov3/) ViT backbone
with a [UPerNet](https://arxiv.org/abs/1807.10221) decode head as a drop-in
alternative to the `segmentation_models_pytorch` U-Net / DeepLabV3+ models. It is
wired into `CustomSemanticSegmentationTask`, so a trained checkpoint runs through
`inference.py` unchanged.

## Architecture

`bda/dinov3_upernet.py` implements `DINOv3UPerNet`:

* **Backbone** — a pretrained DINOv3 ViT (loaded from the gated `facebook/dinov3-*`
  Hugging Face repos). Four equally spaced transformer blocks are read out; the
  class + register tokens are dropped and the patch tokens are reshaped back to
  feature maps.
* **Neck** — the four same-resolution (stride-16) maps are turned into a
  `{stride 4, 8, 16, 32}` pyramid with transposed-conv / identity / max-pool
  branches (the BEiT/MAE ViT-UPerNet recipe).
* **Head** — a UPerNet head (Pyramid Pooling Module + FPN fusion, GroupNorm) that
  produces per-pixel logits, upsampled to the input resolution.

It maps a `(B, in_channels, H, W)` ImageNet-normalized tensor to `(B, num_classes,
H, W)` logits, exactly like the other segmentation models.

Available backbones (`backbone=...`): `dinov3_vits16` (21 M), `dinov3_vitb16`,
`dinov3_vitl16`, `dinov3_vitl16_sat` (satellite-pretrained SAT-493M).

## Requirements

The DINOv3 backbone needs the optional `transformers` dependency (already added to
`environment.yml`) and access to the **gated** DINOv3 weights:

```bash
pip install "transformers>=4.56"
huggingface-cli login          # after requesting access on the model page
```

`H` and `W` must be divisible by the patch size (16), e.g. a power-of-two
`inference.patch_size` such as 512.

## Using it in `inference.py`

No code changes are needed — the architecture is rebuilt from the checkpoint's
hyperparameters. Point the standard inference config at an `upernet` checkpoint:

```bash
python inference.py --config configs/my_config.yml \
    --inference.checkpoint_fn checkpoints/last.ckpt \
    --imagery.raw_fn path/to/image.tif
```

## Training through `fine_tune.py`

`fine_tune.py` now reads `training.model`, `training.backbone`, and
`training.weights` from the config (defaulting to the previous
`unet` / `resnext50_32x4d`). To fine-tune DINOv3 + UPerNet, add:

```yaml
training:
  model: upernet
  backbone: dinov3_vits16
  weights: true        # load the pretrained DINOv3 backbone
  # ...existing training keys...
```

## Example: xView2 (xBD) damage segmentation

`scripts/train_eval_xview2_dinov3_upernet.py` trains and evaluates the model on
the [xView2 / xBD](https://xview2.org) dataset, collapsing the four damage grades
into a single `damaged` class in one of three ways
(`{0: background, 1: undamaged, 2: damaged}`):

| grouping | undamaged | damaged |
| --- | --- | --- |
| `any` | no-damage | minor + major + destroyed |
| `major` | no-damage + minor | major + destroyed |
| `destroyed` | no-damage + minor + major | destroyed |

```bash
python scripts/train_eval_xview2_dinov3_upernet.py \
    --xview2-root ~/data/XView2 --grouping any --gpu 0 \
    --output-dir outputs/xview2_dinov3_upernet_any
```

It trains on the `train/` folder (post-disaster RGB, 512-px crops, class-weighted
cross-entropy for the heavy background imbalance) and reports pixel IoU /
precision / recall / F1 per class on the **full** `test/` folder. Results are in
[`RESULTS.md`](scripts/xview2_dinov3_upernet_RESULTS.md).
