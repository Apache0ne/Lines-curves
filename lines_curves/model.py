from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def smish(x: torch.Tensor) -> torch.Tensor:
    """Activation used by the original TEED implementation."""
    return x * torch.tanh(torch.log1p(torch.sigmoid(x)))


class Smish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return smish(x)


def _weight_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DenseLayer(nn.Sequential):
    def __init__(self, input_features: int, out_features: int) -> None:
        super().__init__()
        # Preserve the original TEED state-dict names exactly.
        self.add_module(
            "conv1",
            nn.Conv2d(input_features, out_features, 3, stride=1, padding=2, bias=True),
        )
        self.add_module("smish1", Smish())
        self.add_module(
            "conv2",
            nn.Conv2d(out_features, out_features, 3, stride=1, bias=True),
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]):
        x1, x2 = inputs
        new_features = super().forward(smish(x1))
        return 0.5 * (new_features + x2), x2


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers: int, input_features: int, out_features: int) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1}", DenseLayer(input_features, out_features))
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features: int, up_scale: int) -> None:
        super().__init__()
        all_pads = [0, 0, 1, 3, 7]
        layers: list[nn.Module] = []
        for i in range(up_scale):
            kernel_size = 2**up_scale
            out_features = 1 if i == up_scale - 1 else 16
            layers.extend(
                [
                    nn.Conv2d(in_features, out_features, 1),
                    Smish(),
                    nn.ConvTranspose2d(
                        out_features,
                        out_features,
                        kernel_size,
                        stride=2,
                        padding=all_pads[up_scale],
                    ),
                ]
            )
            in_features = out_features
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int, use_ac: bool = False) -> None:
        super().__init__()
        self.use_ac = use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        self.activation = Smish() if use_ac else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class DoubleConvBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        mid_features: int,
        out_features: int | None = None,
        stride: int = 1,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        out_features = mid_features if out_features is None else out_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.activation = Smish()
        self.use_act = use_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x) if self.use_act else x


class DoubleFusion(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.DWconv1 = nn.Conv2d(
            in_channels, in_channels * 8, 3, stride=1, padding=1, groups=in_channels
        )
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(24, 24, 3, stride=1, padding=1, groups=24)
        self.AF = Smish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.PSconv1(self.DWconv1(self.AF(x)))
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))
        return smish((attn2 + attn).sum(1, keepdim=True))


class CurveContext(nn.Module):
    """Very small anisotropic context block for long curved structures."""

    def __init__(self, channels: int = 48) -> None:
        super().__init__()
        self.h = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels)
        self.v = nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels)
        self.d = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels)
        self.mix = nn.Conv2d(channels, channels, 1)
        self.act = Smish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(self.act(self.h(x) + self.v(x) + self.d(x)))


class TEEDCurves(nn.Module):
    """TEED-compatible shared encoder with independent edge and curve heads.

    The edge modules retain the original TEED state-dict names, allowing the
    public ``5_model.pth`` checkpoint to initialize the shared encoder and edge
    head. The curve head starts from copied edge-head weights by default.
    """

    CURVE_MODULE_PREFIXES = ("curve_context", "curve_up_block_", "curve_block_cat")

    def __init__(self, use_curve_context: bool = True) -> None:
        super().__init__()
        # Original TEED-compatible names.
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = DenseBlock(1, 32, 48)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)

        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)
        self.block_cat = DoubleFusion(3)

        self.curve_context = CurveContext(48) if use_curve_context else nn.Identity()
        self.curve_up_block_1 = UpConvBlock(16, 1)
        self.curve_up_block_2 = UpConvBlock(32, 1)
        self.curve_up_block_3 = UpConvBlock(48, 2)
        self.curve_block_cat = DoubleFusion(3)

        self.apply(_weight_init)
        # Start the optional context block as an exact identity mapping.
        if isinstance(self.curve_context, CurveContext):
            nn.init.zeros_(self.curve_context.mix.weight)
            if self.curve_context.mix.bias is not None:
                nn.init.zeros_(self.curve_context.mix.bias)

    @staticmethod
    def _resize_to(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if x.shape[-2:] == (height, width):
            return x
        return F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        return block_1, block_2, block_3

    def _edge_decode(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor], out_hw: tuple[int, int]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        b1, b2, b3 = features
        sides = [self.up_block_1(b1), self.up_block_2(b2), self.up_block_3(b3)]
        sides = [self._resize_to(side, *out_hw) for side in sides]
        fused = self.block_cat(torch.cat(sides, dim=1))
        return sides, fused

    def _curve_decode(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor], out_hw: tuple[int, int]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        b1, b2, b3 = features
        b3 = self.curve_context(b3)
        sides = [
            self.curve_up_block_1(b1),
            self.curve_up_block_2(b2),
            self.curve_up_block_3(b3),
        ]
        sides = [self._resize_to(side, *out_hw) for side in sides]
        fused = self.curve_block_cat(torch.cat(sides, dim=1))
        return sides, fused

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB input, got {tuple(x.shape)}")
        out_hw = (x.shape[-2], x.shape[-1])
        features = self.encode(x)
        edge_sides, edge = self._edge_decode(features, out_hw)
        curve_sides, curve = self._curve_decode(features, out_hw)
        return {"edge": edge, "curve": curve, "edge_sides": edge_sides, "curve_sides": curve_sides}

    def freeze_for_stage(self, stage: int) -> None:
        if stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2, or 3")
        for parameter in self.parameters():
            parameter.requires_grad = stage != 1
        if stage == 1:
            for name, parameter in self.named_parameters():
                if name.startswith(self.CURVE_MODULE_PREFIXES):
                    parameter.requires_grad = True

    def copy_edge_head_to_curve_head(self) -> None:
        pairs = [
            (self.up_block_1, self.curve_up_block_1),
            (self.up_block_2, self.curve_up_block_2),
            (self.up_block_3, self.curve_up_block_3),
            (self.block_cat, self.curve_block_cat),
        ]
        for source, target in pairs:
            target.load_state_dict(source.state_dict(), strict=True)

    def load_teed_checkpoint(self, checkpoint: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        raw: Any
        if isinstance(checkpoint, Mapping):
            raw = checkpoint
        else:
            raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if isinstance(raw, Mapping):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in raw and isinstance(raw[key], Mapping):
                    raw = raw[key]
                    break
        if not isinstance(raw, Mapping):
            raise TypeError("Checkpoint does not contain a state dictionary")
        cleaned = {str(k).removeprefix("module."): v for k, v in raw.items()}
        compatible = {
            k: v
            for k, v in cleaned.items()
            if k in self.state_dict() and self.state_dict()[k].shape == v.shape
        }
        result = self.load_state_dict(compatible, strict=False)
        self.copy_edge_head_to_curve_head()
        return {
            "loaded_keys": len(compatible),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }

    def parameter_report(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
