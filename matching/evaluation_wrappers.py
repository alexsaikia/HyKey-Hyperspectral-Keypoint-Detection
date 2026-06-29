import os
import sys
from pathlib import Path
import torch
import numpy as np
from typing import List, Optional

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)



class HSIModelWrapper(torch.nn.Module):
    """Base wrapper for HSI-specific models that don't fit the standard matcher interface"""

    def __init__(self, device: str = 'cuda', **kwargs):
        super().__init__()
        self.device = device

    def extract_features_single(self, image: np.ndarray):
        """Extract features from a single image. To be implemented by subclasses."""
        raise NotImplementedError

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, *args, **kwargs):
        """Implement the base matcher interface by extracting features from both images."""
        img0_np = img0.permute(1, 2, 0).cpu().numpy()
        img1_np = img1.permute(1, 2, 0).cpu().numpy()

        kpts0, desc0 = self.extract_features_single(img0_np)
        kpts1, desc1 = self.extract_features_single(img1_np)

        if len(desc0) > 0 and len(desc1) > 0:
            from evaluate.utils import mnn_match_descriptors
            matches_idx, _ = mnn_match_descriptors(desc0, desc1)
            if matches_idx.size > 0:
                mkpts0 = kpts0[matches_idx[:, 0]]
                mkpts1 = kpts1[matches_idx[:, 1]]
            else:
                mkpts0 = np.empty((0, 2))
                mkpts1 = np.empty((0, 2))
        else:
            mkpts0 = np.empty((0, 2))
            mkpts1 = np.empty((0, 2))

        if args or kwargs:
            return {
                "num_inliers": 0,
                "H": None,
                "mkpts0": mkpts0,
                "mkpts1": mkpts1,
                "inliers0": mkpts0,
                "inliers1": mkpts1,
                "kpts0": kpts0,
                "kpts1": kpts1,
                "desc0": desc0,
                "desc1": desc1,
            }

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1


class HyKeyWrapper(HSIModelWrapper):
    """Wrapper for HyKey model with YAML-configured architecture."""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda', model_yaml: Optional[str] = None, **kwargs):
        super().__init__(device, **kwargs)

        from models.hykey import HyKey
        model_kwargs = {}

        if model_yaml is not None and os.path.exists(model_yaml):
            try:
                from train.utils import load_config as _load_train_cfg, get_hykey_model_kwargs
                train_cfg = _load_train_cfg(model_yaml)
                model_kwargs = get_hykey_model_kwargs(train_cfg)
            except Exception as e:
                print(f"Warning: failed to parse model_yaml {model_yaml}: {e}. Falling back to defaults/overrides.")

        override_keys = [
            'input_channels','learning_rate','debug_dir','use_identity_homography','im_size',
            'c1','c2','c3','dim','radius','top_k','scores_th','n_limit','scores_th_eval','n_limit_eval',
            'w_pk','w_rp','w_sp','w_ds','num_blocks','train_gt_th','eval_gt_th','sc_th',
            'temp_det','temp_des','temp_rel','norm','debug_losses','lr_scheduler','mask_min_avg','mask_max_avg'
        ]
        for k in override_keys:
            if k in kwargs and kwargs[k] is not None:
                model_kwargs[k] = kwargs[k]

        if not model_kwargs:
            model_kwargs = dict(
                input_channels=16,
                c1=16, c2=32, c3=64, dim=128,
                radius=2, top_k=500, scores_th=0.0, n_limit=0,
                scores_th_eval=0.2, n_limit_eval=5000, num_blocks=2,
            )

        self.model = HyKey(**model_kwargs)
        try:
            self.expected_input_channels = int(model_kwargs.get('input_channels')) if ('input_channels' in model_kwargs) else None
        except Exception:
            self.expected_input_channels = None

        if not model_path:
            raise ValueError(
                "HyKeyWrapper requires a checkpoint path via the `model_path` argument; "
                "none was provided."
            )

        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(model_path, map_location=device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Successfully loaded HyKey weights from: {model_path}")
        else:
            print(f"Warning: HyKey checkpoint not found at {model_path}, using untrained weights")

        self.model.to(device)
        self.model.eval()

    def extract_features_single(self, image: np.ndarray):
        with torch.no_grad():
            if len(image.shape) == 3:
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            else:
                image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(self.device)

            pred = self.model.extract(image_tensor)

            if len(pred['keypoints']) > 0 and len(pred['keypoints'][0]) > 0:
                keypoints = pred['keypoints'][0].cpu().numpy()
                descriptors = pred['descriptors'][0].cpu().numpy()

                h, w = image.shape[:2]  # keypoints are in [-1,1]; map back to pixel space
                keypoints = (keypoints / 2 + 0.5) * np.array([[w - 1, h - 1]])

                return keypoints, descriptors
            else:
                return np.empty((0, 2)), np.empty((0, 128))


def create_model_wrapper(model_name: str, device: str = 'cuda', **kwargs) -> HyKeyWrapper:
    """Factory function to create the HyKey model wrapper."""
    if model_name.lower() == 'hykey':
        return HyKeyWrapper(device=device, **kwargs)

    raise ValueError(
        f"Unknown model '{model_name}'. This is a HyKey-only release; "
        f"the only available model is 'hykey'."
    )


def get_available_models() -> List[str]:
    """Get list of all available models."""
    return ['hykey']


def hsi_to_pseudo_rgb(hsi_image: np.ndarray) -> np.ndarray:
    """Convert HSI to pseudo-RGB using specific bands"""
    if hsi_image.shape[2] < 3:
        return np.repeat(hsi_image[:, :, 0:1], 3, axis=2)

    B = hsi_image.shape[2]
    # Bands are ordered by ascending wavelength, so map the longest wavelength
    # (last band) to red and the shortest (first band) to blue for natural color.
    red_band = B - 1
    green_band = B // 2
    blue_band = 0

    rgb_image = np.stack([
        hsi_image[:, :, red_band],
        hsi_image[:, :, green_band],
        hsi_image[:, :, blue_band]
    ], axis=2)

    return rgb_image
