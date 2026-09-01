"""2D U-Net modelini yaratish."""
from __future__ import annotations

from monai.networks.nets import UNet


def build_model(cfg: dict):
    m = cfg["model"]
    return UNet(
        spatial_dims=2,
        in_channels=int(m["in_channels"]),
        out_channels=int(m["out_channels"]),
        channels=tuple(m["channels"]),
        strides=tuple(m["strides"]),
        num_res_units=int(m["num_res_units"]),
        dropout=float(m.get("dropout", 0.0)),
    )
