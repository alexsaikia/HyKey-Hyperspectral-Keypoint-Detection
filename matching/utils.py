import numpy as np
import torch


def to_pseudo_rgb(chw: torch.Tensor) -> np.ndarray:
    """(C, H, W) HSI tensor → (H, W, 3) uint8 pseudo-RGB for display."""
    from matching.evaluation_wrappers import hsi_to_pseudo_rgb
    hwc = chw.permute(1, 2, 0).cpu().numpy()
    rgb = hsi_to_pseudo_rgb(hwc)
    rgb = rgb - rgb.min()
    rgb = rgb / (rgb.max() + 1e-8)
    return (rgb * 255).astype(np.uint8)
