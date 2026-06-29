import torch


def norm_to_px(keypoints_xy_norm: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert normalized [-1,1] keypoints to pixel coordinates (x,y) with image size (W,H).
    If input appears already in pixels (outside [-1,1] range), returns input unchanged.
    """
    if keypoints_xy_norm.numel() == 0:
        return keypoints_xy_norm
    if keypoints_xy_norm.min() >= -1.0 and keypoints_xy_norm.max() <= 1.0:
        scale = keypoints_xy_norm.new_tensor([float(width - 1), float(height - 1)])
        return (keypoints_xy_norm + 1.0) / 2.0 * scale
    return keypoints_xy_norm


