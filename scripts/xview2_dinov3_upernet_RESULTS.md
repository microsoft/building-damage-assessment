# xView2 DINOv3 + UPerNet results

Fully-supervised DINOv3 ViT-S/16 + UPerNet (`scripts/train_eval_xview2_dinov3_upernet.py`),
trained on the xView2 (xBD) `train/` folder and evaluated on the **full**
`test/` folder (933 post-disaster images, per-pixel metrics accumulated over a
3×3 confusion matrix). Each row is a separate model that collapses the four xBD
damage grades into one `damaged` class differently; every model is a 3-class
`{background, undamaged, damaged}` segmentation.

**Training config:** backbone `dinov3_vits16` (LVD-1689M, pretrained, fine-tuned
end-to-end), 512-px random crops (80% biased to contain building pixels),
batch 16, class-weighted cross-entropy (inverse-sqrt-frequency), AdamW lr 1e-4,
15 epochs, mixed precision, ImageNet normalization. Best-`val_loss` checkpoint.

## Test-set metrics (per-pixel)

| Grouping (damaged =) | Class | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| **any** (minor+major+destroyed) | background | 0.954 | 0.994 | 0.960 | 0.976 |
| | undamaged | 0.542 | 0.601 | 0.847 | 0.703 |
| | damaged | 0.354 | 0.391 | 0.791 | 0.523 |
| **major** (major+destroyed) | background | 0.958 | 0.994 | 0.963 | 0.978 |
| | undamaged | 0.563 | 0.610 | 0.879 | 0.720 |
| | damaged | 0.377 | 0.424 | 0.774 | 0.548 |
| **destroyed** (destroyed only) | background | 0.958 | 0.994 | 0.964 | 0.979 |
| | undamaged | 0.587 | 0.629 | 0.898 | 0.739 |
| | damaged | 0.271 | 0.298 | 0.749 | 0.426 |

| Grouping | mean IoU | damaged F1 | overall accuracy |
| --- | --- | --- | --- |
| any | 0.617 | 0.523 | 0.952 |
| major | 0.632 | 0.548 | 0.957 |
| destroyed | 0.605 | 0.426 | 0.959 |

**Notes.** The damaged class is rare (≈0.3–2% of pixels) and gets rarer from
`any` → `destroyed`, which is why its IoU drops accordingly. The class-weighted
loss trades precision for recall (damaged recall 0.75–0.79), i.e. the models find
most damaged pixels at the cost of some false positives — a reasonable operating
point for triage. A larger backbone (`dinov3_vitb16` / `dinov3_vitl16_sat`),
longer training, or post-hoc thresholding would push these further.

Reproduce:

```bash
for g in any major destroyed; do
  python scripts/train_eval_xview2_dinov3_upernet.py \
      --xview2-root ~/data/XView2 --grouping $g --gpu 0 \
      --output-dir outputs/xview2_dinov3_upernet_$g
done
```
