# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DINOv3 + UPerNet semantic-segmentation model.

A plain ViT (DINOv3) backbone paired with a UPerNet (PPM + FPN) decode head, in
the style of the BEiT / MAE segmentation recipe: four equally spaced transformer
blocks are read out, reshaped from token sequences back to feature maps, and
turned into a 4-level ``{stride 4, 8, 16, 32}`` pyramid by a small neck of
(de)convolutions before the UPerNet head fuses them into per-pixel logits.

The backbone weights come from the gated ``facebook/dinov3-*`` Hugging Face hub
repos (request access + ``huggingface-cli login`` once). ``transformers`` is an
optional dependency of this repo -- it is only imported when a ``upernet`` model
is requested, so the rest of the toolkit works without it.

The module exposes a single ``nn.Module`` (:class:`DINOv3UPerNet`) that maps a
``(B, in_channels, H, W)`` ImageNet-normalized tensor to ``(B, num_classes, H,
W)`` logits, so it is a drop-in replacement for the ``segmentation_models_pytorch``
models used elsewhere in :class:`bda.trainers.CustomSemanticSegmentationTask`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Short backbone name -> gated Hugging Face hub id.
DINOV3_HF_IDS = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vitl16_sat": "facebook/dinov3-vitl16-pretrain-sat493m",
}


def _norm(num_channels: int) -> nn.GroupNorm:
    """Batch-size-independent normalization (robust to 1x1 PPM maps / small batches)."""
    num_groups = 32
    while num_channels % num_groups:
        num_groups //= 2
    return nn.GroupNorm(num_groups, num_channels)


def _conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
        _norm(out_ch),
        nn.ReLU(inplace=True),
    )


class DINOv3Backbone(nn.Module):
    """DINOv3 ViT that returns four intermediate token maps as feature maps.

    The Hugging Face ``DINOv3ViTModel`` prepends ``1`` class token and
    ``num_register_tokens`` register tokens to the ``(H/p)*(W/p)`` patch tokens;
    those prefix tokens are stripped before each selected layer's tokens are
    reshaped to ``(B, C, H/p, W/p)``.
    """

    def __init__(
        self,
        name: str = "dinov3_vits16",
        pretrained: bool = True,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'upernet' model needs the optional 'transformers' package. "
                "Install it (see environment.yml) and request access to the gated "
                f"'{DINOV3_HF_IDS.get(name, name)}' Hugging Face repo."
            ) from e

        if name not in DINOV3_HF_IDS:
            raise ValueError(
                f"Unknown DINOv3 backbone '{name}'. "
                f"Choose one of {sorted(DINOV3_HF_IDS)}."
            )
        hf_id = DINOV3_HF_IDS[name]
        config = AutoConfig.from_pretrained(hf_id)
        if pretrained:
            self.vit = AutoModel.from_pretrained(hf_id)
        else:
            self.vit = AutoModel.from_config(config)

        self.patch_size = int(config.patch_size)
        self.embed_dim = int(config.hidden_size)
        self.num_prefix_tokens = 1 + int(getattr(config, "num_register_tokens", 0))

        n_layers = int(config.num_hidden_layers)
        if out_indices is None:
            # Four equally spaced blocks; hidden_states[0] is the embedding output
            # and hidden_states[i] is the output of block i, so valid ids are 1..n.
            out_indices = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers]
        self.out_indices = out_indices

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        b, _, h, w = x.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"Input {h}x{w} must be divisible by the patch size "
                f"{self.patch_size}."
            )
        gh, gw = h // self.patch_size, w // self.patch_size
        outputs = self.vit(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        feats = []
        for idx in self.out_indices:
            tokens = hidden_states[idx][:, self.num_prefix_tokens :, :]
            feat = tokens.transpose(1, 2).reshape(b, self.embed_dim, gh, gw)
            feats.append(feat.contiguous())
        return feats


class _PPM(nn.ModuleList):
    """Pyramid Pooling Module (the PSPNet context module used by UPerNet)."""

    def __init__(self, pool_scales, in_dim: int, channels: int) -> None:
        super().__init__()
        for scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_dim, channels, 1, bias=False),
                    _norm(channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for block in self:
            y = block(x)
            out.append(
                F.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)
            )
        return out


class UPerHead(nn.Module):
    """UPerNet decode head: PPM on the coarsest level + an FPN fuse of all levels."""

    def __init__(
        self,
        in_channels: list[int],
        channels: int = 256,
        num_classes: int = 3,
        pool_scales=(1, 2, 3, 6),
    ) -> None:
        super().__init__()
        # PPM over the last (coarsest) feature map.
        self.ppm = _PPM(pool_scales, in_channels[-1], channels)
        self.ppm_bottleneck = _conv_bn_relu(
            in_channels[-1] + len(pool_scales) * channels, channels, 3
        )
        # FPN lateral + output convs for every level except the last.
        self.lateral_convs = nn.ModuleList(
            [_conv_bn_relu(c, channels, 1) for c in in_channels[:-1]]
        )
        self.fpn_convs = nn.ModuleList(
            [_conv_bn_relu(channels, channels, 3) for _ in in_channels[:-1]]
        )
        self.fpn_bottleneck = _conv_bn_relu(len(in_channels) * channels, channels, 3)
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(feats[i]) for i, conv in enumerate(self.lateral_convs)]
        psp = torch.cat([feats[-1], *self.ppm(feats[-1])], dim=1)
        laterals.append(self.ppm_bottleneck(psp))

        # Top-down pathway.
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear",
                align_corners=False,
            )

        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear",
                align_corners=False,
            )
        out = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        return self.classifier(self.dropout(out))


class DINOv3UPerNet(nn.Module):
    """DINOv3 ViT backbone + UPerNet head for dense semantic segmentation.

    Args:
        backbone: DINOv3 short name (see :data:`DINOV3_HF_IDS`).
        in_channels: number of input channels; a 1x1 stem maps to the 3 channels
            the pretrained patch embedding expects when ``in_channels != 3``.
        num_classes: number of output classes.
        pretrained: load the gated Hugging Face pretrained backbone weights.
        channels: UPerNet fusion width.
    """

    def __init__(
        self,
        backbone: str = "dinov3_vits16",
        in_channels: int = 3,
        num_classes: int = 3,
        pretrained: bool = True,
        channels: int = 256,
    ) -> None:
        super().__init__()
        self.stem = (
            nn.Conv2d(in_channels, 3, 1) if in_channels != 3 else nn.Identity()
        )
        self.backbone = DINOv3Backbone(backbone, pretrained=pretrained)
        dim = self.backbone.embed_dim
        # Turn the four same-resolution (stride-16) ViT maps into a
        # {stride 4, 8, 16, 32} pyramid, as in BEiT/MAE UPerNet segmentation.
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, stride=2),
            _norm(dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, 2, stride=2),
        )
        self.fpn2 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decode_head = UPerHead(
            in_channels=[dim, dim, dim, dim],
            channels=channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x = self.stem(x)
        feats = self.backbone(x)
        feats = [self.fpn1(feats[0]), self.fpn2(feats[1]), self.fpn3(feats[2]), self.fpn4(feats[3])]
        logits = self.decode_head(feats)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
