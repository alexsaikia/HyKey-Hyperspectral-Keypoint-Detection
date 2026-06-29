#!/usr/bin/env python3
"""
Evaluate HyKey on planar (homography) and pairwise-pose (epipolar) scenes.

Usage:
    python evaluate_hykey.py --config configs/evaluate_hykey.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import time

# WandB is optional
try:
    import wandb  # type: ignore
    _HAVE_WANDB = True
except Exception:
    _HAVE_WANDB = False

from pytorch_lightning.loggers import WandbLogger  # type: ignore

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from train.utils import load_config, get_dataloader
from data_processing.spectral_dataset import SpectralDataset
import yaml
from evaluate.eval_class import Evaluator, EvaluationConfig
from models.utils.geometry import fundamental_from_pose as _F_from_pose
from matching.evaluation_wrappers import create_model_wrapper, get_available_models
from evaluate.utils import derive_wrapper_kwargs, find_training_yaml, find_best_checkpoint, adapt_images_for_model, mnn_match_descriptors
from models.utils.utils import plot_keypoints, plot_matches_with_accuracy, warp

# Respect an externally provided CUDA_VISIBLE_DEVICES; default to GPU 0 only when the
# launcher did not specify one.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

DEFAULT_BAND_STATS_PATH = str(Path(project_root) / 'configs' / 'band_stats.npz')


def flatten_for_wandb(prefix: str, obj: Any) -> Dict[str, float]:
    """Flatten numeric eval outputs for WandB scalar logging."""
    out: Dict[str, float] = {}
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            obj = obj.detach().cpu().item()
        else:
            return out
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, (int, float, np.number)) and np.isfinite(float(obj)):
        out[prefix] = float(obj)
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'detector_name':
                continue
            key = f"{prefix}/{k}" if prefix else str(k)
            out.update(flatten_for_wandb(key, v))
    return out


def clean_eval_metrics(flat: Dict[str, float]) -> Dict[str, float]:
    """Collapse the double-aggregation suffixes into a clean, readable set.

    The eval pipeline aggregates twice (over pairs within a batch, then over
    batches), so every metric ends up with concatenated suffixes like
    ``X_mean_mean`` / ``X_mean_std`` / ``X_std_mean`` / ``X_std_std``. Only the
    first two carry useful information:
      - ``X_mean_mean``  -> ``X``      (point estimate / headline value)
      - ``X_mean_std``   -> ``X_std``  (between-batch variability = error bar)
      - ``X_mean_median``-> ``X_median``
      - ``X_std_mean`` / ``X_std_std`` are dropped (second-order noise)
    Single-suffix keys (e.g. ``pose_success_rate_mean``, ``planar_batches``)
    are passed through unchanged to avoid name collisions.
    """
    cleaned: Dict[str, float] = {}
    for key, val in flat.items():
        if key.endswith('_std_mean') or key.endswith('_std_std'):
            continue
        if key.endswith('_mean_mean'):
            cleaned[key[: -len('_mean_mean')]] = val
        elif key.endswith('_mean_std'):
            cleaned[key[: -len('_mean_std')] + '_std'] = val
        elif key.endswith('_mean_median'):
            cleaned[key[: -len('_mean_median')] + '_median'] = val
        else:
            cleaned[key] = val
    return cleaned


def drop_eval_modality_prefix(flat: Dict[str, float]) -> Dict[str, float]:
    """Move modality out of WandB metric paths.

    Evaluation configs are run per modality, and the modality is now encoded in
    the WandB run/model name. Keeping it in metric paths creates duplicate panel
    groups for the same metric (``detector_eval/HSI/...`` and
    ``detector_eval/RGB/...``), so strip that one path component before logging.
    """
    cleaned: Dict[str, float] = {}
    for key, val in flat.items():
        if key.startswith('detector_eval/HSI/'):
            cleaned['detector_eval/' + key[len('detector_eval/HSI/'):]] = val
        elif key.startswith('detector_eval/RGB/'):
            cleaned['detector_eval/' + key[len('detector_eval/RGB/'):]] = val
        else:
            cleaned[key] = val
    return cleaned


ROBUST_CANONICAL_SUFFIXES = (
    'pose_t_deg',
    'pose_R_deg',
    'pose_success_rate',
    'processed_pairs',
    'failed_pairs',
)


def canonicalize_robust_metrics(flat: Dict[str, float]) -> Dict[str, float]:
    """Use the robust metric definitions as the canonical public names.

    Some evaluator outputs include both ``metric`` and ``metric_robust``. The
    robust version is the correct one for this project, so expose it as
    ``metric`` and remove the older duplicate from logging.
    """
    canonical: Dict[str, float] = {}
    robust_renames: Dict[str, float] = {}
    for key, val in flat.items():
        renamed = None
        for base in ROBUST_CANONICAL_SUFFIXES:
            marker = f'{base}_robust'
            if key.endswith(marker):
                renamed = key[: -len(marker)] + base
                break
            for suffix in ('_std', '_median'):
                marker_with_suffix = f'{base}_robust{suffix}'
                if key.endswith(marker_with_suffix):
                    renamed = key[: -len(marker_with_suffix)] + f'{base}{suffix}'
                    break
            if renamed is not None:
                break
        if renamed is None:
            canonical[key] = val
        else:
            robust_renames[renamed] = val

    # Robust values win over the old metric at the same canonical key.
    canonical.update(robust_renames)
    return canonical


def relative_pose_error_robust(R_est: torch.Tensor, t_est: torch.Tensor,
                                R_gt: torch.Tensor, t_gt: torch.Tensor, ignore_gt_t_thr=0.0) -> Tuple[float, float]:
    """Compute relative pose errors with a sign-invariant translation angle.

    The translation error uses abs(dot product), so the angle is always <=90 deg
    (the sign of the translation direction is ignored).

    Args:
        R_est, t_est: Estimated rotation matrix and translation vector
        R_gt, t_gt: Ground truth rotation matrix and translation vector

    Returns:
        t_err: Translation angle error in degrees (0-90°)
        R_err: Rotation error in degrees
    """
    if isinstance(R_est, torch.Tensor):
        R_est = R_est.detach().cpu().numpy()
    if isinstance(t_est, torch.Tensor):
        t_est = t_est.detach().cpu().numpy().reshape(-1)
    if isinstance(R_gt, torch.Tensor):
        R_gt = R_gt.detach().cpu().numpy()
    if isinstance(t_gt, torch.Tensor):
        t_gt = t_gt.detach().cpu().numpy().reshape(-1)
    t_mag = np.linalg.norm(t_est)
    t_gt_mag = np.linalg.norm(t_gt)

    if t_mag < 1e-5 or t_gt_mag < 1e-5:
        if t_gt_mag < ignore_gt_t_thr:  # pure rotation case
            t_err = 0
        else:
            # If one is small and the other isn't, use 90 degrees as the error
            t_err = 90.0
    else:
        cos_angle = np.abs(np.dot(t_est, t_gt)) / (t_mag * t_gt_mag)
        cos_angle = np.clip(cos_angle, 0.0, 1.0)
        t_err = np.rad2deg(np.arccos(cos_angle))

    cos = (np.trace(np.dot(R_est.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1, 1.0)
    R_err = np.rad2deg(np.abs(np.arccos(cos)))
    
    return float(t_err), float(R_err)

def compute_homography_from_pose_robust(K1, K2, R, t, normal=None, d=1.0):
    """Compute H = K2 (R - t n^T / d) K1^{-1} for a plane with given normal and depth."""
    if normal is None:
        normal = np.array([0, 0, 1]).reshape(3, 1)
    else:
        normal = np.array(normal).reshape(3, 1)
    
    if isinstance(K1, torch.Tensor):
        K1 = K1.detach().cpu().numpy()
    if isinstance(K2, torch.Tensor):
        K2 = K2.detach().cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    
    t = np.array(t).reshape(3, 1)

    H = K2 @ (R - (t @ normal.T) / d) @ np.linalg.inv(K1)
    return H / H[2, 2]

def homography_transfer_error_robust(H_gt, H_est, points, symmetric=True):
    """Mean pixel distance between H_gt and H_est applied to points; symmetric averages both directions."""
    n_points = len(points)
    pts_homog = np.hstack((points, np.ones((n_points, 1))))

    pts_gt_homog = (H_gt @ pts_homog.T).T
    pts_gt = pts_gt_homog[:, :2] / pts_gt_homog[:, 2:3]

    pts_est_homog = (H_est @ pts_homog.T).T
    pts_est = pts_est_homog[:, :2] / pts_est_homog[:, 2:3]

    errors = np.sqrt(np.sum((pts_gt - pts_est) ** 2, axis=1))
    
    if symmetric:
        try:
            inv_H_gt = np.linalg.inv(H_gt)
            inv_H_est = np.linalg.inv(H_est)

            back_pts_homog = np.hstack((pts_gt, np.ones((n_points, 1))))

            back_pts_gt_homog = (inv_H_gt @ back_pts_homog.T).T
            back_pts_gt = back_pts_gt_homog[:, :2] / back_pts_gt_homog[:, 2:3]

            back_pts_est_homog = (inv_H_est @ back_pts_homog.T).T
            back_pts_est = back_pts_est_homog[:, :2] / back_pts_est_homog[:, 2:3]

            back_errors = np.sqrt(np.sum((back_pts_gt - back_pts_est) ** 2, axis=1))

            errors = (errors + back_errors) / 2
        except Exception:
            pass

    return np.mean(errors)

def compute_homography_error_robust(K0, K1, R, t, points1, points2):
    """Estimate H from matched points via USAC_MAGSAC, compare to GT H derived from pose.
    Returns (h_error_px, H_gt, H_est, inlier_mask)."""
    try:
        # Compute ground truth homography (assuming frontal plane at average depth)
        normal = [0, 0, 1]  # Normal vector in camera 1's coordinate frame
        avg_depth = 1.0  # Use a normalized depth for simplicity
        
        H_gt = compute_homography_from_pose_robust(K0, K1, R, t, normal, avg_depth)
        
        # Compute estimated homography from matches
        H_est, mask = cv2.findHomography(
            points1, points2, cv2.USAC_MAGSAC, 
            3.0,  # reproj_thresh
            0.95, # confidence
            1000  # max_iters
        )
        
        if H_est is None:
            return float('inf'), H_gt, None, None
        
        # Convert mask to boolean array
        mask = mask.ravel().astype(bool)
        
        # Use inliers for error computation
        inlier_points1 = points1[mask] if mask.sum() > 0 else points1[:10]
        
        # Compute transfer error
        h_error = homography_transfer_error_robust(H_gt, H_est, inlier_points1)
        
        return h_error, H_gt, H_est, mask
    except Exception as e:
        return float('inf'), None, None, None

def to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Return 3-channel tensor from HSI or already-RGB image, preserving device/dtype.
    Expects input as (1, C, H, W) or (C, H, W); returns (C=3, H, W)."""
    if img.ndim == 4:
        img = img.squeeze(0)
    if img.shape[0] > 3:
        c = img.shape[0]
        return img[[-1, c // 2, 0]]
    return img

def kpts_to_pixel(kpts: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert keypoints from normalized [-1,1] to pixel if needed; else return as-is."""
    if not isinstance(kpts, torch.Tensor):
        kpts = torch.as_tensor(kpts, dtype=torch.float32)
    if kpts.numel() == 0:
        return kpts.reshape(0, 2)
    kmax = float(kpts.max()) if kpts.numel() > 0 else 0.0
    kmin = float(kpts.min()) if kpts.numel() > 0 else 0.0
    if kmax <= 1.0 and kmin >= -1.0:
        scale = kpts.new_tensor([max(width - 1, 1), max(height - 1, 1)])
        return (kpts + 1.0) / 2.0 * scale
    return kpts

def ensure_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def safe_name_from_paths(path0: Optional[str], path1: Optional[str]) -> str:
    """Make a filesystem-safe name from optional image paths."""
    def base(p: Optional[str]) -> str:
        if p is None:
            return "unknown"
        b = os.path.basename(p)
        b = b.replace(os.sep, "_")
        return b
    return f"{base(path0)}__{base(path1)}"

class CombinedDetectorEvaluator:
    def __init__(self, config_path: str, device: str = 'cuda', wandb_logger: Optional[WandbLogger] = None,
                 nonplanar_step: int = 1, output_dir: str = "hykey_eval"):
        self.config_path = config_path
        self.device = device
        self.wandb_logger = wandb_logger
        self.nonplanar_step = max(1, int(nonplanar_step))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = load_config(config_path)

        eval_cfg_for_thresholds = self.cfg.get('evaluation', {}) if isinstance(self.cfg, dict) else {}
        thresholds = eval_cfg_for_thresholds.get('thresholds')
        pose_auc_degs_raw = eval_cfg_for_thresholds.get('pose_auc_degs')
        pose_auc_degs = tuple(float(x) for x in pose_auc_degs_raw) if pose_auc_degs_raw else None
        
        eval_config = EvaluationConfig(
            device=device, 
            thresholds=thresholds or [1, 3, 5, 10, 20],
            pose_auc_degs=pose_auc_degs or (5.0, 10.0, 20.0)
        )
        self.unified_evaluator = Evaluator(eval_config)

        if thresholds is not None:
            self.thresholds = thresholds
        else:
            self.thresholds = [1, 3, 5, 10, 20]

        eval_cfg = self.cfg.get('evaluation', {}) if isinstance(self.cfg, dict) else {}
        self.write_per_pair_csv: bool = bool(eval_cfg.get('write_per_pair_csv', True))
        self.save_visualizations: bool = bool(eval_cfg.get('save_visualizations', True))
        self.save_success_curves: bool = bool(eval_cfg.get('save_success_curves', True))
        self.csv_buffer_size: int = int(eval_cfg.get('csv_buffer_size', 200))
        self.tqdm_disable: bool = bool(eval_cfg.get('tqdm_disable', False))
        self.unified_evaluator.config.enable_plots = bool(eval_cfg.get('enable_plots', True))
        self.use_autocast: bool = bool(eval_cfg.get('autocast', True))
        amp_dtype_str = str(eval_cfg.get('amp_dtype', 'bf16')).lower()
        self._amp_dtype = torch.bfloat16 if amp_dtype_str in ('bf16', 'bfloat16') else torch.float16
        self.jpeg_quality: int = int(eval_cfg.get('jpeg_quality', 85))
        self._csv_buffers: Dict[Path, List[Dict[str, Any]]] = {}
        self._csv_headers: Dict[Path, List[str]] = {}

        # Threading / matmul knobs (all optional); ignore unsupported settings silently
        opencv_threads = eval_cfg.get('opencv_num_threads', None)
        if isinstance(opencv_threads, int) and opencv_threads > 0:
            cv2.setNumThreads(int(opencv_threads))  # type: ignore
        torch_threads = eval_cfg.get('torch_num_threads', None)
        if isinstance(torch_threads, int) and torch_threads > 0:
            torch.set_num_threads(int(torch_threads))
        interop_threads = eval_cfg.get('torch_num_interop_threads', None)
        if isinstance(interop_threads, int) and interop_threads > 0:
            torch.set_num_interop_threads(int(interop_threads))
        if bool(eval_cfg.get('cudnn_benchmark', True)):
            torch.backends.cudnn.benchmark = True  # type: ignore
        # Allow TF32 for Ampere+ to speed matmuls
        if bool(eval_cfg.get('allow_tf32', True)):
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
            torch.backends.cudnn.allow_tf32 = True  # type: ignore
        mp = eval_cfg.get('matmul_precision', None)
        if isinstance(mp, str) and hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision(mp)  # type: ignore
        omp_threads = eval_cfg.get('omp_num_threads', None)
        if isinstance(omp_threads, int) and omp_threads > 0:
            os.environ['OMP_NUM_THREADS'] = str(int(omp_threads))
        mkl_threads = eval_cfg.get('mkl_num_threads', None)
        if isinstance(mkl_threads, int) and mkl_threads > 0:
            os.environ['MKL_NUM_THREADS'] = str(int(mkl_threads))
        openblas_threads = eval_cfg.get('openblas_num_threads', None)
        if isinstance(openblas_threads, int) and openblas_threads > 0:
            os.environ['OPENBLAS_NUM_THREADS'] = str(int(openblas_threads))
        numexpr_threads = eval_cfg.get('numexpr_num_threads', None)
        if isinstance(numexpr_threads, int) and numexpr_threads > 0:
            os.environ['NUMEXPR_NUM_THREADS'] = str(int(numexpr_threads))
        # Display color handling: dataset tensors are RGB. Keep true RGB by default;
        # only enable this legacy option for explicitly BGR sources.
        self.rgb_reverse_for_display: bool = bool(eval_cfg.get('rgb_reverse_channels_for_display', False))

        # Image-pairing options. HyKey is evaluated on HSI frames using synthetic
        # (warped) homography pairs for the planar metrics, so no RGB<->HSI registration
        # engine is needed here.
        self.reg_direction: str = 'hsi_to_rgb'
        self.pair_mode: str = 'auto'  # 'auto' | 'warped'
        eval_cfg = self.cfg.get('evaluation', {}) if isinstance(self.cfg, dict) else {}
        if isinstance(eval_cfg, dict):
            self.pair_mode = str(eval_cfg.get('pair_mode', self.pair_mode))
            self.reg_direction = str(eval_cfg.get('registration_direction', self.reg_direction))

        self._nonplanar_counter = 0

    def load_test_scenes(self) -> Dict[str, List[str]]:
        """Load test_set.yaml and return a dict with the 'spectral' scene list."""
        eval_cfg = self.cfg.get('evaluation', {}) if isinstance(self.cfg, dict) else {}
        default_path = 'configs/test_set.yaml'
        test_set_path = str(eval_cfg.get('test_set_path', default_path))
        if not os.path.exists(test_set_path):
            raise FileNotFoundError(f"test_set.yaml not found at {test_set_path}")
        with open(test_set_path, 'r') as f:
            ts_cfg = yaml.safe_load(f) or {}
        spectral = []
        if 'spectral' in ts_cfg and isinstance(ts_cfg['spectral'], dict):
            spectral = list(ts_cfg['spectral'].get('acquisition_folders', []) or [])
        return {'spectral': spectral}

    def create_dataloader_for_scene(self, scene_type: str, scene_name: str):
        """Create a per-scene DataLoader(batch_size=1) without shuffling.
        Uses SpectralDataset with the configured nonplanar_stride."""
        ds_cfg = self.cfg.get('dataset', {}) if isinstance(self.cfg, dict) else {}
        tr_cfg = self.cfg.get('training', {}) if isinstance(self.cfg, dict) else {}
        ev_cfg = self.cfg.get('evaluation', {}) if isinstance(self.cfg, dict) else {}
        band_stats_path = ds_cfg.get('band_stats_path', DEFAULT_BAND_STATS_PATH)
        # Synthetic-homography difficulty: 'paper' | 'harder' preset, or explicit dict.
        # Default None preserves generate_warped_image defaults (current/hard warps).
        from data_processing.utils.augmentation_utils import resolve_warp_params
        warp_params = resolve_warp_params(ev_cfg.get('warp_difficulty', ds_cfg.get('warp_params')))

        spectral_list: List[str] = [scene_name]
        spectral_root = ds_cfg['spectral_data_root']

        dataset = SpectralDataset(
            data_root=spectral_root,
            acquisition_folders=spectral_list,
            load_poses=ds_cfg.get('load_poses', True),
            load_rgb=ds_cfg.get('load_rgb', True),
            load_hsi=ds_cfg.get('load_hsi', True),
            aug_strength=0.0,
            registration=ds_cfg.get('registration'),
            calibration_dir=ds_cfg.get('calibration_dir'),
            target_space=ds_cfg.get('target_space', 'rgb'),
            undistort=ds_cfg.get('undistort', False),
            split='all',
            test_size=None,
            val_size=None,
            random_state=ds_cfg.get('random_state', 42),
            radiometric_calibration=ds_cfg.get('radiometric_calibration', True),
            warp_augmentation=ds_cfg.get('warp_augmentation', True),
            photometric_augmentation=False,
            photometric_augmentation_nonplanar=False,
            always_include_nonplanar=True,
            max_nonplanar_offset=200,
            nonplanar_stride=int(self.nonplanar_step),
            band_stats_path=band_stats_path,
            warp_params=warp_params,
        )
        dl = get_dataloader(
            dataset,
            batch_size=int(tr_cfg.get('batch_size', 1)),
            num_workers=int(tr_cfg.get('num_workers', 0)),
            persistent_workers=bool(tr_cfg.get('persistent_workers', False)),
            shuffle=False
        )
        return dl

    def prepare_detector_inputs(self, batch: Dict[str, Any], modality: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
        """
        For a batch, build a list of per-sample inputs for evaluation.
        Returns: (samples, meta, H, W)
          samples[i] = {
              'kpts0','kpts1','desc0','desc1', 'warp01_params' (planar optional)
          }
          meta[i] = optional meta with paths/scene
        """
        samples: List[Dict[str, Any]] = []
        meta: List[Dict[str, Any]] = []

        # Decide image tensors based strictly on requested modality
        have_hsi = ('hsi' in batch)
        have_rgb = ('rgb' in batch)
        have_warped_hsi = ('warped_hsi' in batch)
        have_warped_rgb = ('warped_rgb' in batch)
        have_np_hsi = ('hsi_np' in batch)
        have_np_rgb = ('rgb_np' in batch)

        if modality.upper() == 'HSI' and have_hsi and have_warped_hsi:
            image0 = batch['hsi']
            image1 = batch['warped_hsi']
            image1_np = batch['hsi_np'] if have_np_hsi else None
        elif modality.upper() == 'RGB' and have_rgb and have_warped_rgb:
            image0 = batch['rgb']
            image1 = batch['warped_rgb']
            image1_np = batch['rgb_np'] if have_np_rgb else None
        else:
            return samples, meta, 0, 0

        B, C, H, W = image0.shape
        device = torch.device(self.device)

        # Build per-sample inputs
        for i in range(B):
            img0 = image0[i:i+1].to(device)
            img1 = image1[i:i+1].to(device)
            img1_np = image1_np[i:i+1].to(device) if (image1_np is not None and i < image1_np.shape[0]) else None

            sdict: Dict[str, Any] = {
                'img0': img0.squeeze(0), 'img1': img1.squeeze(0),
                'H': int(H), 'W': int(W)
            }
            if img1_np is not None:
                sdict['img1_np'] = img1_np.squeeze(0)
            samples.append(sdict)

            # Attach meta if present (paths, scene)
            mi: Dict[str, Any] = {}
            # Prefer HSI paths for naming if available
            p0 = None
            p1 = None
            for k0, k1 in [('hsi_path', 'warped_hsi_path'), ('hsi_path', 'hsi_np_path'), ('rgb_path', 'warped_rgb_path'), ('rgb_path', 'rgb_np_path')]:
                if k0 in batch and isinstance(batch[k0], list) and i < len(batch[k0]):
                    p0 = batch[k0][i]
                if k1 in batch and isinstance(batch[k1], list) and i < len(batch[k1]):
                    p1 = batch[k1][i]
                if p0 is not None or p1 is not None:
                    break
            if p0 is not None:
                mi['image0_path'] = p0
            if p1 is not None:
                mi['image1_path'] = p1
            # Also record nonplanar image1 path when present
            try:
                if modality.upper() == 'HSI' and ('hsi_np_path' in batch) and isinstance(batch['hsi_np_path'], list) and i < len(batch['hsi_np_path']):
                    mi['image1_np_path'] = batch['hsi_np_path'][i]
                if modality.upper() == 'RGB' and ('rgb_np_path' in batch) and isinstance(batch['rgb_np_path'], list) and i < len(batch['rgb_np_path']):
                    mi['image1_np_path'] = batch['rgb_np_path'][i]
            except Exception:
                pass
            # Best-effort scene parsing from path
            try:
                if 'scene' not in mi and (p0 is not None):
                    pdir = os.path.dirname(os.path.dirname(p0))
                    mi['scene'] = os.path.basename(pdir)
            except Exception:
                pass
            meta.append(mi)

        # Add planar homography (synthetic/planar warped pairs)
        if 'H_mat' in batch:
            # Use per-batch synthetic homographies (planar warped pairs)
            H_mat = batch['H_mat']
            for i in range(min(len(samples), len(H_mat))):
                samples[i]['warp01_params'] = {
                    'homography_matrix': ensure_tensor(H_mat[i], device),
                    'width': samples[i]['W'], 'height': samples[i]['H']
                }

        return samples, meta, H, W

    def forward_detector(self, detector, img0_chw: torch.Tensor, img1_chw: torch.Tensor) -> Dict[str, Any]:
        # The wrappers typically accept CHW tensors; provide .squeeze(0) as used elsewhere
        return detector.forward(
            img0_chw, img1_chw,
            scale0=None, scale1=None,
            img0_orig=img0_chw, img1_orig=img1_chw
        )

    def img_chw_to_rgb_uint8(self, img_chw: torch.Tensor) -> np.ndarray:
        """Convert CHW tensor (any channels) to HxWx3 uint8 RGB image for visualization."""
        if img_chw.ndim == 4:
            img_chw = img_chw.squeeze(0)
        img3 = to_rgb(img_chw.detach().cpu())  # CxHxW, select bands if >3
        # Dataset tensors are RGB; this legacy override is only for explicitly BGR sources.
        try:
            if getattr(self, '_vis_modality', None) == 'RGB' and self.rgb_reverse_for_display and img3.shape[0] == 3:
                img3 = img3[[2, 1, 0], ...]
        except Exception:
            pass
        img3 = torch.clamp(img3, 0.0, 1.0)
        img_rgb = (img3 * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()  # HxWxC
        return img_rgb

    def save_score_concat_image(self, scores_map0: Optional[torch.Tensor], scores_map1: Optional[torch.Tensor], out_path: Path) -> None:
        try:
            if not self.save_visualizations:
                return
            if scores_map0 is None or scores_map1 is None:
                return
            s0 = scores_map0.detach()
            s1 = scores_map1.detach()
            if s0.dim() == 3 and s0.shape[0] == 1:
                s0 = s0.squeeze(0)
            if s1.dim() == 3 and s1.shape[0] == 1:
                s1 = s1.squeeze(0)
            s0u = (s0.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            s1u = (s1.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            s0c = cv2.applyColorMap(s0u, cv2.COLORMAP_MAGMA)
            s1c = cv2.applyColorMap(s1u, cv2.COLORMAP_MAGMA)
            # Keep RGB ordering for saving via OpenCV (convert to BGR)
            concat = cv2.hconcat([cv2.cvtColor(s0c, cv2.COLOR_BGR2RGB), cv2.cvtColor(s1c, cv2.COLOR_BGR2RGB)])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            params = []
            suf = out_path.suffix.lower()
            if suf in ('.jpg', '.jpeg'):
                try:
                    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
                except Exception:
                    params = []
            cv2.imwrite(str(out_path), cv2.cvtColor(concat, cv2.COLOR_RGB2BGR), params)
        except Exception as e:
            print(f"Warning: failed to save score concat image to {out_path}: {e}")

    def save_image_with_keypoints(self, img0_chw: torch.Tensor, img1_chw: torch.Tensor,
                                   kpts0_px: torch.Tensor, kpts1_px: torch.Tensor,
                                   out_path: Path) -> None:
        try:
            if not self.save_visualizations:
                return
            h, w = int(img0_chw.shape[-2]), int(img0_chw.shape[-1])
            # plot_keypoints expects (y, x) order per vis.py usage
            k0yxp = kpts0_px[:, [1, 0]] if kpts0_px.numel() > 0 else kpts0_px
            k1yxp = kpts1_px[:, [1, 0]] if kpts1_px.numel() > 0 else kpts1_px
            img0 = self.img_chw_to_rgb_uint8(img0_chw)
            img1 = self.img_chw_to_rgb_uint8(img1_chw)
            img0_k = plot_keypoints(img0, k0yxp, radius=1)
            img1_k = plot_keypoints(img1, k1yxp, radius=1)
            concat = cv2.hconcat([img0_k, img1_k])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            params = []
            suf = out_path.suffix.lower()
            if suf in ('.jpg', '.jpeg'):
                try:
                    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
                except Exception:
                    params = []
            cv2.imwrite(str(out_path), cv2.cvtColor(concat, cv2.COLOR_RGB2BGR), params)
        except Exception as e:
            print(f"Warning: failed to save image-with-keypoints to {out_path}: {e}")

    def save_matches_with_accuracy_planar(self, img0_chw: torch.Tensor, img1_chw: torch.Tensor,
                                           k0px: torch.Tensor, k1px: torch.Tensor,
                                           d0: torch.Tensor, d1: torch.Tensor,
                                           H01: torch.Tensor, out_path: Path,
                                           sim_threshold: float = 0.7, ratio_threshold: float = 0.9) -> None:
        try:
            if not self.save_visualizations:
                return
            mk0, mk1 = self.select_matches(d0, d1, k0px, k1px, sim_threshold=sim_threshold, ratio_threshold=ratio_threshold)
            # Compute warped overlay and accuracy mask like vis.py
            accuracy_mask = np.zeros((len(mk0),), dtype=bool)
            mk0_warped_full = None
            if len(mk0) > 0:
                try:
                    H = H01
                    h, w = int(img0_chw.shape[-2]), int(img0_chw.shape[-1])
                    warp01_params = {
                        'mode': 'homo',
                        'homography_matrix': H,
                        'width': w,
                        'height': h
                    }
                    # convert to torch tensors for warp helper
                    mk0_t = torch.as_tensor(mk0, dtype=torch.float32, device=H.device)
                    mk1_t = torch.as_tensor(mk1, dtype=torch.float32, device=H.device)
                    mkpts0_warped, mkpts01, ids0, _ = warp(mk0_t, warp01_params)
                    if ids0.numel() > 0:
                        mk1_valid = mk1_t[ids0]
                        dist = torch.sqrt(((mkpts01 - mk1_valid) ** 2).sum(dim=1))
                        valid_accuracy = (dist < 3.0)
                        acc_np = torch.zeros(mk0_t.shape[0], dtype=torch.bool, device=H.device)
                        acc_np[ids0] = valid_accuracy
                        accuracy_mask = acc_np.detach().cpu().numpy().astype(bool)
                        mk0_warped_full = np.full((mk0.shape[0], 2), np.nan, dtype=np.float32)
                        mk0_warped_full[ids0.detach().cpu().numpy()] = mkpts01[:, [1, 0]].detach().cpu().numpy()
                except Exception:
                    mk0_warped_full = None
            # Prepare images and plotting coords
            img0 = self.img_chw_to_rgb_uint8(img0_chw)
            img1 = self.img_chw_to_rgb_uint8(img1_chw)
            mk0_plot = mk0[:, [1, 0]] if len(mk0) > 0 else mk0
            mk1_plot = mk1[:, [1, 0]] if len(mk1) > 0 else mk1
            match_image = plot_matches_with_accuracy(
                img0, img1, mk0_plot, mk1_plot, mk0_warped_full, accuracy_mask,
                correct_color=(0, 255, 0), incorrect_color=(255, 0, 0)
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            params = []
            suf = out_path.suffix.lower()
            if suf in ('.jpg', '.jpeg'):
                try:
                    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
                except Exception:
                    params = []
            cv2.imwrite(str(out_path), cv2.cvtColor(match_image, cv2.COLOR_RGB2BGR), params)
        except Exception as e:
            print(f"Warning: failed to save planar matches image to {out_path}: {e}")

    def save_matches_with_accuracy_nonplanar(self, img0_chw: torch.Tensor, img1_chw: torch.Tensor,
                                              k0px: torch.Tensor, k1px: torch.Tensor,
                                              d0: torch.Tensor, d1: torch.Tensor,
                                              K0: torch.Tensor, K1: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
                                              out_path: Path, sim_threshold: float = 0.6, ratio_threshold: float = 0.8) -> None:
        try:
            if not self.save_visualizations:
                return
            mk0, mk1 = self.select_matches(d0, d1, k0px, k1px, sim_threshold=sim_threshold, ratio_threshold=ratio_threshold)
            acc_mask = self.accuracy_mask_nonplanar(mk0, mk1, K0, K1, R, t, thresh_px=5.0)
            img0 = self.img_chw_to_rgb_uint8(img0_chw)
            img1 = self.img_chw_to_rgb_uint8(img1_chw)
            mk0_plot = mk0[:, [1, 0]] if len(mk0) > 0 else mk0
            mk1_plot = mk1[:, [1, 0]] if len(mk1) > 0 else mk1
            match_image = plot_matches_with_accuracy(
                img0, img1, mk0_plot, mk1_plot, None, acc_mask,
                correct_color=(0, 255, 0), incorrect_color=(255, 0, 0)
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            params = []
            suf = out_path.suffix.lower()
            if suf in ('.jpg', '.jpeg'):
                try:
                    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
                except Exception:
                    params = []
            cv2.imwrite(str(out_path), cv2.cvtColor(match_image, cv2.COLOR_RGB2BGR), params)
        except Exception as e:
            print(f"Warning: failed to save nonplanar matches image to {out_path}: {e}")

    def save_epipolar_lines_image(self, img0_chw: torch.Tensor, img1_chw: torch.Tensor,
                                   k0px: torch.Tensor, k1px: torch.Tensor,
                                   d0: torch.Tensor, d1: torch.Tensor,
                                   K0: torch.Tensor, K1: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
                                   out_path: Path, sim_threshold: float = 0.6, ratio_threshold: float = 0.8) -> None:
        try:
            if not self.save_visualizations:
                return
            # Build F
            F = _F_from_pose(K0, K1, R, t).detach().cpu().numpy().astype(np.float32)
            # Select matches
            mk0, mk1 = self.select_matches(d0, d1, k0px, k1px, sim_threshold=sim_threshold, ratio_threshold=ratio_threshold)
            if len(mk0) == 0 or len(mk1) == 0:
                return
            img0 = self.img_chw_to_rgb_uint8(img0_chw)
            img1 = self.img_chw_to_rgb_uint8(img1_chw)
            draw0 = cv2.cvtColor(img0.copy(), cv2.COLOR_RGB2BGR)
            draw1 = cv2.cvtColor(img1.copy(), cv2.COLOR_RGB2BGR)
            h, w = draw0.shape[:2]
            # Keep up to 5 matches after simple border checks
            kept = []
            border = 10
            for k in range(min(len(mk0), len(mk1))):
                p0 = mk0[k]
                p1 = mk1[k]
                inside0 = (border <= p0[0] <= (w - 1 - border)) and (border <= p0[1] <= (h - 1 - border))
                inside1 = (border <= p1[0] <= (w - 1 - border)) and (border <= p1[1] <= (h - 1 - border))
                if inside0 and inside1:
                    kept.append(k)
                if len(kept) >= 5:
                    break
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
            K0n = K0.detach().cpu().numpy().astype(np.float32)
            K1n = K1.detach().cpu().numpy().astype(np.float32)
            for idx, k in enumerate(kept):
                color = colors[idx % len(colors)]
                p0 = mk0[k]
                p1 = mk1[k]
                cv2.circle(draw0, (int(round(p0[0])), int(round(p0[1]))), 4, color, thickness=-1)
                cv2.circle(draw1, (int(round(p1[0])), int(round(p1[1]))), 4, color, thickness=-1)
                # Epipolar lines in image1 for p0
                x0h = np.array([p0[0], p0[1], 1.0], dtype=np.float32)
                l1 = (F @ x0h.reshape(3, 1)).reshape(3)
                a1, b1, c1 = float(l1[0]), float(l1[1]), float(l1[2])
                # Intersections with borders
                pts = []
                if abs(a1) > 1e-8:
                    x0 = -c1 / a1
                    x1 = -(c1 + b1 * (h - 1)) / a1
                    if 0 <= x0 < w:
                        pts.append((int(round(x0)), 0))
                    if 0 <= x1 < w:
                        pts.append((int(round(x1)), h - 1))
                if abs(b1) > 1e-8:
                    y0 = -c1 / b1
                    y1 = -(c1 + a1 * (w - 1)) / b1
                    if 0 <= y0 < h:
                        pts.append((0, int(round(y0))))
                    if 0 <= y1 < h:
                        pts.append((w - 1, int(round(y1))))
                if len(pts) >= 2:
                    cv2.line(draw1, pts[0], pts[1], color, thickness=2)
            concat = cv2.hconcat([draw0, draw1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            params = []
            suf = out_path.suffix.lower()
            if suf in ('.jpg', '.jpeg'):
                try:
                    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
                except Exception:
                    params = []
            cv2.imwrite(str(out_path), concat, params)
        except Exception as e:
            print(f"Warning: failed to save epipolar lines image to {out_path}: {e}")

    def select_matches(self, d0: torch.Tensor, d1: torch.Tensor, kpts0_px: torch.Tensor, kpts1_px: torch.Tensor,
                         sim_threshold: float = 0.7, ratio_threshold: float = 0.9,
                         top_k: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """Mutual best + ratio test on cosine similarity of descriptors.
        Returns (mkpts0_px, mkpts1_px) as numpy float32 arrays Nx2 in pixel coords."""
        if d0.numel() == 0 or d1.numel() == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        # Limit to top_k by detection order
        d0s = d0[:top_k].detach()
        d1s = d1[:top_k].detach()
        k0 = kpts0_px[:top_k].detach()
        k1 = kpts1_px[:top_k].detach()
        S = torch.matmul(d0s, d1s.t())  # assume L2-normalized
        n0, n1 = S.shape
        if n1 >= 2:
            top2_vals, top2_idx = S.topk(k=2, dim=1)
            best_vals = top2_vals[:, 0]
            best_idx = top2_idx[:, 0]
            second_vals = top2_vals[:, 1]
            ratio_ok = best_vals >= ratio_threshold * (second_vals + 1e-6)
        else:
            best_vals, best_idx = S.max(dim=1)
            ratio_ok = torch.ones_like(best_vals, dtype=torch.bool)
        rev_best_idx = S.argmax(dim=0)
        arange_i = torch.arange(n0, device=S.device)
        mutual_ok = arange_i.eq(rev_best_idx[best_idx])
        thresh_ok = best_vals >= sim_threshold
        keep = mutual_ok & ratio_ok & thresh_ok
        sel_i = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if sel_i.numel() == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        sel_j = best_idx[sel_i]
        mk0 = k0[sel_i].detach().cpu().numpy().astype(np.float32)
        mk1 = k1[sel_j].detach().cpu().numpy().astype(np.float32)
        return mk0, mk1

    def accuracy_mask_nonplanar(self, mk0_px: np.ndarray, mk1_px: np.ndarray,
                                 K0: torch.Tensor, K1: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
                                 thresh_px: float = 5.0) -> np.ndarray:
        if mk0_px.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        # Build F from pose
        F = _F_from_pose(K0, K1, R, t)
        F = F.detach().cpu().numpy().astype(np.float32)
        # Sampson distance per-point
        N = mk0_px.shape[0]
        x0 = np.concatenate([mk0_px, np.ones((N, 1), dtype=np.float32)], axis=1)
        x1 = np.concatenate([mk1_px, np.ones((N, 1), dtype=np.float32)], axis=1)
        Ex0 = (F @ x0.T).T
        Etx1 = (F.T @ x1.T).T
        num = np.abs(np.sum(x1 * Ex0, axis=1))
        den = Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2 + 1e-12
        d = num / np.sqrt(den)
        return d < float(thresh_px)

    def append_csv_row(self, file_path: Path, row: Dict[str, Any]) -> None:
        """Buffered append of a row to CSV; caches header to avoid per-append reads."""
        try:
            if not self.write_per_pair_csv:
                return
            file_path.parent.mkdir(parents=True, exist_ok=True)
            buf = self._csv_buffers.setdefault(file_path, [])
            buf.append(row)
            if len(buf) >= max(1, int(self.csv_buffer_size)):
                self.flush_csv_buffer(file_path)
        except Exception as e:
            print(f"Warning: failed to buffer row for {file_path}: {e}")

    def flush_csv_buffer(self, file_path: Path) -> None:
        try:
            if file_path not in self._csv_buffers or len(self._csv_buffers[file_path]) == 0:
                return
            rows = self._csv_buffers[file_path]
            self._csv_buffers[file_path] = []
            file_exists = file_path.exists()
            df = pd.DataFrame(rows)
            # Standardize leading columns
            lead_cols = [c for c in ['Detector', 'Modality', 'Planarity'] if c in df.columns]
            remaining = [c for c in df.columns if c not in lead_cols]
            df = df[lead_cols + remaining]
            # Align columns to cached header if file exists
            header_cols = None
            if file_exists:
                header_cols = self._csv_headers.get(file_path, None)
                if header_cols is None:
                    try:
                        header_df = pd.read_csv(file_path, nrows=0)
                        header_cols = list(header_df.columns)
                        self._csv_headers[file_path] = header_cols
                    except Exception:
                        header_cols = None
                if header_cols is not None:
                    for col in header_cols:
                        if col not in df.columns:
                            df[col] = None
                    df = df[header_cols]
            else:
                # Cache header from first write
                self._csv_headers[file_path] = list(df.columns)
            df.to_csv(file_path, mode='a', header=(not file_exists), index=False)
        except Exception as e:
            print(f"Warning: failed to flush CSV buffer to {file_path}: {e}")

    def flush_all_csv_buffers(self) -> None:
        for fp in list(self._csv_buffers.keys()):
            self.flush_csv_buffer(fp)

    def result_to_row(self, res: Dict[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                row[k] = v.item() if v.numel() == 1 else v.detach().cpu().numpy().tolist()
            elif isinstance(v, (int, float, str)):
                row[k] = v
        return row

    def evaluate_detector(self, detector_name: str, max_num_keypoints: int = 5000,
                          enable_planar: bool = True, enable_nonplanar: bool = True,
                          overrides_for_model: Optional[Dict[str, Any]] = None,
                          display_name: Optional[str] = None) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        label_name = str(display_name) if (display_name is not None) else str(detector_name)
        print(f"Evaluating detector: {label_name}")
        print(f"{'='*60}")

        # Build per-model overrides (checkpoint, model yaml, hyperparams)
        overrides = dict(overrides_for_model or {})
        # Auto-configure from checkpoint directory if provided
        ckpt_dir = overrides.pop('checkpoint_dir', None)
        if ckpt_dir:
            try:
                ckpt_dir_path = Path(str(ckpt_dir))
                train_yaml = find_training_yaml(ckpt_dir_path)
                best_ckpt = find_best_checkpoint(ckpt_dir_path)
                if train_yaml:
                    try:
                        from evaluate.utils import load_yaml as _load_yaml
                        train_cfg = _load_yaml(Path(train_yaml))
                        auto_kwargs = derive_wrapper_kwargs(detector_name, train_cfg)
                        overrides.update(auto_kwargs)
                    except Exception as e:
                        print(f"Warning: failed to parse training yaml at {train_yaml}: {e}")
                if best_ckpt and ('model_path' not in overrides):
                    overrides['model_path'] = str(best_ckpt)
                    print(f"Using checkpoint for {detector_name}: {best_ckpt}")
            except Exception as e:
                print(f"Warning: checkpoint auto-config failed for {detector_name}: {e}")

        # If a model_yaml path is explicitly provided, parse it to derive wrapper kwargs
        # (e.g., input_channels) so the wrapper receives the correct channel config.
        try:
            model_yaml_path = overrides.get('model_yaml', None)
            if isinstance(model_yaml_path, str) and os.path.isfile(model_yaml_path):
                try:
                    from evaluate.utils import load_yaml as _load_yaml
                    train_cfg = _load_yaml(Path(model_yaml_path))
                    auto_kwargs = derive_wrapper_kwargs(detector_name, train_cfg)
                    # Do not override explicitly provided overrides
                    for k, v in auto_kwargs.items():
                        overrides.setdefault(k, v)
                except Exception as e:
                    print(f"Warning: failed to parse model_yaml at {model_yaml_path}: {e}")
        except Exception:
            pass

        # Use max_num_keypoints from YAML config, don't override detector-specific settings
        overrides['max_num_keypoints'] = max_num_keypoints
        # For custom detectors (e.g., HyKey), align detector caps with evaluation setting
        overrides['top_k'] = max_num_keypoints
        overrides['n_limit_eval'] = max_num_keypoints
        overrides['n_limit'] = max_num_keypoints

        # Create detector wrapper with overrides
        detector = create_model_wrapper(
            detector_name, device=self.device, **overrides
        )

        # Output directories root for this detector
        base_dir = self.output_dir / label_name
        base_dir.mkdir(parents=True, exist_ok=True)

        scenes = self.load_test_scenes()
        # Modalities to evaluate (default both)
        try:
            eval_modalities = self.cfg.get('evaluation', {}).get('modalities', ['RGB', 'HSI'])
            if not isinstance(eval_modalities, list) or len(eval_modalities) == 0:
                eval_modalities = ['RGB', 'HSI']
        except Exception:
            eval_modalities = ['RGB', 'HSI']

        final: Dict[str, Any] = {'detector_name': label_name}
        mod_state: Dict[str, Dict[str, Any]] = {}
        for modality in eval_modalities:
            try:
                if str(detector_name).lower().startswith('hykey') and modality == 'RGB':
                    print("Skipping RGB modality for hykey as requested")
                    continue
            except Exception:
                pass
            # Guard against invalid cross-modality evaluation: a learned detector with a
            # fixed input-channel count must only run on the modality whose native channel
            # count matches. Otherwise conform_channels would, e.g., pad a 3-band RGB image
            # to 16 channels (replicating blue) and feed it to a 16-band HSI model, or slice
            # an HSI cube into a 3-band pseudo-RGB for an RGB model -- both meaningless.
            # exp_c is None for classical detectors, which legitimately handle adaptation.
            try:
                exp_c = getattr(detector, 'expected_input_channels', None)
                if isinstance(exp_c, int) and exp_c > 0:
                    is_rgb_modality = (str(modality).upper() == 'RGB')
                    model_is_rgb = (exp_c <= 3)
                    if is_rgb_modality != model_is_rgb:
                        print(f"Skipping {modality} modality for {detector_name} "
                              f"(model expects {exp_c} input channels; modality channel count mismatch)")
                        continue
            except Exception:
                pass
            mod_dir = base_dir / modality
            planar_dir = mod_dir / 'planar'
            nonplanar_dir = mod_dir / 'nonplanar'
            planar_dir.mkdir(parents=True, exist_ok=True)
            nonplanar_dir.mkdir(parents=True, exist_ok=True)
            mod_state[modality] = {
                'planar_dir': planar_dir,
                'nonplanar_dir': nonplanar_dir,
                'planar_batch_metrics': [],
                'nonplanar_batch_metrics': [],
                'saved_vis_planar_scenes': set(),
                'saved_vis_nonplanar_scenes': set(),
            }

        for scene_type in ['spectral']:
            for scene_name in scenes.get(scene_type, []):
                scene_dl = self.create_dataloader_for_scene(scene_type, scene_name)
                scene_safe = scene_name.replace(os.sep, '_')
                # Per-scene accumulators for summary per modality
                scene_planar_acc: Dict[str, List[Dict[str, Any]]] = {m: [] for m in mod_state.keys()}
                scene_nonplanar_acc: Dict[str, List[Dict[str, Any]]] = {m: [] for m in mod_state.keys()}
                # Optional cap batches per scene for quick debug runs via env EVAL_MAX_BATCHES
                max_batches_env = os.environ.get('EVAL_MAX_BATCHES', None)
                try:
                    max_batches_cfg = self.cfg.get('evaluation', {}).get('max_batches_per_scene', None)
                except Exception:
                    max_batches_cfg = None
                max_batches = None
                if max_batches_cfg is not None:
                    try:
                        max_batches = int(max_batches_cfg)
                    except Exception:
                        max_batches = None
                if (max_batches is None) and (max_batches_env is not None) and str(max_batches_env).isdigit():
                    max_batches = int(max_batches_env)
                num_batches = len(scene_dl)
                for batch_idx, batch in enumerate(tqdm(scene_dl, desc=f"Evaluating {detector_name} [{scene_type}:{scene_name}]", disable=self.tqdm_disable)):
                    if (max_batches is not None) and (batch_idx >= max_batches):
                        break
                    for modality in list(mod_state.keys()):
                        # Visualization modality context so display functions handle channel order
                        self._vis_modality = modality
                        planar_dir = mod_state[modality]['planar_dir']
                        nonplanar_dir = mod_state[modality]['nonplanar_dir']
                        planar_csv_path = planar_dir / 'planar.csv'
                        nonplanar_csv_path = nonplanar_dir / 'nonplanar.csv'
                        planar_imgs_dir = planar_dir / 'imgs' / scene_safe
                        nonplanar_imgs_dir = nonplanar_dir / 'imgs' / scene_safe
                        saved_vis_planar_scenes = mod_state[modality]['saved_vis_planar_scenes']
                        saved_vis_nonplanar_scenes = mod_state[modality]['saved_vis_nonplanar_scenes']
                        samples, meta, H, W = self.prepare_detector_inputs(batch, modality)
                        if len(samples) == 0:
                            continue

                        # Build lists for planar and nonplanar evaluation per this batch
                        pred_planar = {'kpts0': [], 'kpts1': [], 'desc0': [], 'desc1': [], 'warp01_params': []}
                        pred_nonplanar = {'kpts0': [], 'kpts1': [], 'desc0': [], 'desc1': [], 'imgs0': [], 'imgs1': []}
                        cam_pack = {'K0': [], 'K1': [], 'R_01': [], 't_01': []}
                        np_meta_indices: List[int] = []  # indices of selected samples for nonplanar
                        np_forward_times: List[float] = []  # ms for nonplanar forward per appended item

                        # Determine availability flags per-sample
                        have_planar = ('H_mat' in batch)
                        have_nonplanar = all(k in batch for k in ['K0', 'K1', 'R_01', 't_01'])

                        # Per-sample processing
                        for i, s in enumerate(samples):
                            # Forward detector
                            img0_chw = s['img0']
                            img1_chw = s['img1']
                            # Adapt channel layout based on model (keeps full channels for custom models)
                            img0_adapt, img1_adapt = adapt_images_for_model(detector_name, img0_chw, img1_chw)
                            # If wrapper expects a specific number of channels, conform HSI by selecting bands or replicating
                            try:
                                exp_c = getattr(detector, 'expected_input_channels', None)
                            except Exception:
                                exp_c = None
                            if isinstance(exp_c, int) and exp_c > 0:
                                def conform_channels(x: torch.Tensor, target_c: int) -> torch.Tensor:
                                    c, h, w = x.shape
                                    if c == target_c:
                                        return x
                                    if c > target_c:
                                        mean = x.mean(dim=0, keepdim=True)
                                        if target_c == 1:
                                            return mean
                                        if target_c == 3:
                                            return x[[-1, c // 2, 0], :, :]
                                        return mean.expand(target_c, h, w)
                                    reps = [x]
                                    while sum(t.shape[0] for t in reps) < target_c:
                                        reps.append(x[-1:].expand(1, h, w))
                                    y = torch.cat(reps, dim=0)
                                    return y[:target_c]
                                img0_adapt = conform_channels(img0_adapt, exp_c)
                                img1_adapt = conform_channels(img1_adapt, exp_c)
                            _t0 = time.perf_counter()
                            with torch.inference_mode():
                                # Some classical/immatch models don't support autocast dtypes; allow wrappers to disable it
                                disable_amp = bool(getattr(detector, 'disable_autocast', False))
                                if (self.use_autocast and not disable_amp) and (self.device.startswith('cuda') or self.device == 'cuda'):
                                    with torch.autocast(device_type='cuda', dtype=self._amp_dtype):
                                        det_out = self.forward_detector(detector, img0_adapt, img1_adapt)
                                else:
                                    det_out = self.forward_detector(detector, img0_adapt, img1_adapt)
                            _t1 = time.perf_counter()
                            elapsed_forward_ms = float((_t1 - _t0) * 1000.0)
                            k0 = det_out.get('kpts0', torch.empty(0, 2, device=self.device))
                            k1 = det_out.get('kpts1', torch.empty(0, 2, device=self.device))
                            d0 = det_out.get('desc0', torch.empty(0, 128, device=self.device))
                            d1 = det_out.get('desc1', torch.empty(0, 128, device=self.device))
                            scores_map0 = det_out.get('scores_map0', None)
                            scores_map1 = det_out.get('scores_map1', None)

                            # Ensure tensors on device
                            k0 = ensure_tensor(k0, torch.device(self.device))
                            k1 = ensure_tensor(k1, torch.device(self.device))
                            d0 = ensure_tensor(d0, torch.device(self.device))
                            d1 = ensure_tensor(d1, torch.device(self.device))

                            # To pixel coords if needed (use each sample's dims if available)
                            W_i = int(s.get('W', W)); H_i = int(s.get('H', H))
                            k0px = kpts_to_pixel(k0, W_i, H_i)
                            k1px = kpts_to_pixel(k1, W_i, H_i)

                            # Meta naming
                            p0 = meta[i].get('image0_path') if i < len(meta) else None
                            p1 = meta[i].get('image1_path') if i < len(meta) else None
                            safe_name = safe_name_from_paths(p0, p1)

                            # Planar path
                            if enable_planar and have_planar and ('warp01_params' in s):
                                pred_planar['kpts0'].append(k0px)
                                pred_planar['kpts1'].append(k1px)
                                pred_planar['desc0'].append(d0)
                                pred_planar['desc1'].append(d1)
                                pred_planar['warp01_params'].append(s['warp01_params'])
                                # Save vis images for first batch per scene
                                # if (scene_name not in saved_vis_planar_scenes):
                                try:
                                    # Score maps (if available)
                                    if scores_map0 is not None and scores_map1 is not None:
                                        self.save_score_concat_image(scores_map0, scores_map1, planar_imgs_dir / 'score_0.jpg')
                                    # Image with keypoints
                                    self.save_image_with_keypoints(img0_chw, img1_chw, k0px, k1px, planar_imgs_dir / 'image_0.jpg')
                                    # Matches with accuracy using homography
                                    H01 = s['warp01_params']['homography_matrix']
                                    self.save_matches_with_accuracy_planar(img0_chw, img1_chw, k0px, k1px, d0, d1, H01, planar_imgs_dir / 'matches_0.jpg')
                                except Exception as e:
                                    print(f"Warning: failed to save planar visualization for scene {scene_name}: {e}")
                                    # saved_vis_planar_scenes.add(scene_name)

                            # Nonplanar path: use explicit nonplanar image if provided by dataloader
                            if enable_nonplanar and have_nonplanar:
                                if 'img1_np' not in s:
                                    raise RuntimeError(f"Nonplanar sample missing img1_np (scene={scene_name}, b={batch_idx}, i={i})")
                                else:
                                    img1_np_chw = s['img1_np']
                                    img0_np_adapt, img1_np_adapt = adapt_images_for_model(detector_name, img0_chw, img1_np_chw)
                                    # Conform channels for nonplanar path as well
                                    try:
                                        exp_c = getattr(detector, 'expected_input_channels', None)
                                    except Exception:
                                        exp_c = None
                                    if isinstance(exp_c, int) and exp_c > 0:
                                        def conform_channels(x: torch.Tensor, target_c: int) -> torch.Tensor:
                                            c, h, w = x.shape
                                            if c == target_c:
                                                return x
                                            if c > target_c:
                                                mean = x.mean(dim=0, keepdim=True)
                                                if target_c == 1:
                                                    return mean
                                                if target_c == 3:
                                                    return x[[-1, c // 2, 0], :, :]
                                                return mean.expand(target_c, h, w)
                                            reps = [x]
                                            while sum(t.shape[0] for t in reps) < target_c:
                                                reps.append(x[-1:].expand(1, h, w))
                                            y = torch.cat(reps, dim=0)
                                            return y[:target_c]
                                        img0_np_adapt = conform_channels(img0_np_adapt, exp_c)
                                        img1_np_adapt = conform_channels(img1_np_adapt, exp_c)
                                    _tn0 = time.perf_counter()
                                    with torch.inference_mode():
                                        disable_amp = bool(getattr(detector, 'disable_autocast', False))
                                        if (self.use_autocast and not disable_amp) and (self.device.startswith('cuda') or self.device == 'cuda'):
                                            with torch.autocast(device_type='cuda', dtype=self._amp_dtype):
                                                det_out_np = self.forward_detector(detector, img0_np_adapt, img1_np_adapt)
                                        else:
                                            det_out_np = self.forward_detector(detector, img0_np_adapt, img1_np_adapt)
                                    _tn1 = time.perf_counter()
                                    np_forward_times.append(float((_tn1 - _tn0) * 1000.0))
                                    k0_np = ensure_tensor(det_out_np.get('kpts0', torch.empty(0, 2, device=self.device)), torch.device(self.device))
                                    k1_np = ensure_tensor(det_out_np.get('kpts1', torch.empty(0, 2, device=self.device)), torch.device(self.device))
                                    d0_np = ensure_tensor(det_out_np.get('desc0', torch.empty(0, 128, device=self.device)), torch.device(self.device))
                                    d1_np = ensure_tensor(det_out_np.get('desc1', torch.empty(0, 128, device=self.device)), torch.device(self.device))
                                    scores_map0_np = det_out_np.get('scores_map0', None)
                                    scores_map1_np = det_out_np.get('scores_map1', None)
                                    k0px_np = kpts_to_pixel(k0_np, W_i, H_i)
                                    k1px_np = kpts_to_pixel(k1_np, W_i, H_i)
                                    pred_nonplanar['kpts0'].append(k0px_np)
                                    pred_nonplanar['kpts1'].append(k1px_np)
                                    pred_nonplanar['desc0'].append(d0_np)
                                    pred_nonplanar['desc1'].append(d1_np)
                                    pred_nonplanar['imgs0'].append(img0_chw)
                                    pred_nonplanar['imgs1'].append(img1_np_chw)
                                K0i = ensure_tensor(batch['K0'][i], k0px.device)
                                K1i = ensure_tensor(batch['K1'][i], k0px.device)
                                R01i = ensure_tensor(batch['R_01'][i], k0px.device)
                                t01i = ensure_tensor(batch['t_01'][i], k0px.device)
                                cam_pack['K0'].append(K0i)
                                cam_pack['K1'].append(K1i)
                                cam_pack['R_01'].append(R01i)
                                cam_pack['t_01'].append(t01i)
                                np_meta_indices.append(i)
                                # Save vis images for first batch per scene (nonplanar)
                                if (scene_name not in saved_vis_nonplanar_scenes) and ('img1_np' in s):
                                    try:
                                        # Score maps (if available)
                                        if scores_map0_np is not None and scores_map1_np is not None:
                                            self.save_score_concat_image(scores_map0_np, scores_map1_np, nonplanar_imgs_dir / 'score_0.jpg')
                                        # Image with keypoints
                                        self.save_image_with_keypoints(img0_chw, img1_np_chw, k0px_np, k1px_np, nonplanar_imgs_dir / 'image_0.jpg')
                                        # Matches with epipolar accuracy
                                        self.save_matches_with_accuracy_nonplanar(img0_chw, img1_np_chw, k0px_np, k1px_np, d0_np, d1_np, K0i, K1i, R01i, t01i, nonplanar_imgs_dir / 'matches_0.jpg')
                                        # Epipolar lines visualization
                                        self.save_epipolar_lines_image(img0_chw, img1_np_chw, k0px_np, k1px_np, d0_np, d1_np, K0i, K1i, R01i, t01i, nonplanar_imgs_dir / 'epi_0.jpg')
                                    except Exception as e:
                                        print(f"Warning: failed to save nonplanar visualization for scene {scene_name}: {e}")
                                    saved_vis_nonplanar_scenes.add(scene_name)

                            if enable_planar and have_planar and ('warp01_params' in s):
                                try:
                                    res = self.unified_evaluator.evaluate(k0px, k1px, d0, d1,
                                                                          s['warp01_params']['homography_matrix'],
                                                                          (H_i, W_i))
                                    # Per-pair CSV
                                    row = self.result_to_row(res)
                                    row['forward_ms'] = float(elapsed_forward_ms)
                                    row['_batch_idx'] = batch_idx
                                    row['_sample_idx'] = i
                                    row['_safe_name'] = safe_name
                                    # Standardize leading identifiers
                                    row['Detector'] = label_name
                                    row['Modality'] = modality
                                    row['Planarity'] = 'planar'
                                    row['scene_type'] = scene_type
                                    row['scene'] = scene_name
                                    if 'scene' in meta[i]:
                                        row['scene'] = meta[i]['scene']
                                    if p0 is not None:
                                        row['image0_path'] = p0
                                    if p1 is not None:
                                        row['image1_path'] = p1
                                    # Append row immediately to per-scene CSV
                                    self.append_csv_row(planar_csv_path, row)
                                except Exception as e:
                                    print(f"Warning: planar per-pair eval failed (b={batch_idx}, i={i}): {e}")

                        # Batch-level aggregation (planar)
                        if enable_planar and have_planar and len(pred_planar['kpts0']) > 0:
                            try:
                                pack = self.unified_evaluator.evaluate_batch_individual(pred_planar, (H, W))
                                mod_state[modality]['planar_batch_metrics'].append(pack['batch_metrics'])
                                scene_planar_acc[modality].append(pack['batch_metrics'])
                            except Exception as e:
                                print(f"Warning: planar batch eval failed (scene={scene_name}, b={batch_idx}, modality={modality}): {e}")

                        # Nonplanar aggregation and per-pair CSVs (already subsampled per-sample)
                        if enable_nonplanar and have_nonplanar and len(pred_nonplanar['kpts0']) > 0:
                            try:
                                pack_np = self.unified_evaluator.evaluate_nonplanar_batch_individual(
                                    pred_nonplanar, cam_pack, epi_thresholds_px= self.thresholds
                                )
                                try:
                                    if 'concat_sampson_px' in pack_np and pack_np['concat_sampson_px'].numel() > 0:
                                        scene_key = (modality, scene_type, scene_name)
                                        _se_store = getattr(self, '_scene_se_store', {})
                                        _se_store.setdefault(scene_key, []).append(pack_np['concat_sampson_px'].detach().cpu())
                                        self._scene_se_store = _se_store
                                    if 'concat_pose_err_deg' in pack_np and pack_np['concat_pose_err_deg'].numel() > 0:
                                        scene_key = (modality, scene_type, scene_name)
                                        _pe_store = getattr(self, '_scene_pose_store', {})
                                        _pe_store.setdefault(scene_key, []).append(pack_np['concat_pose_err_deg'].detach().cpu())
                                        self._scene_pose_store = _pe_store
                                except Exception:
                                    pass
                                # Debug: brief batch summary
                                try:
                                    bm = pack_np.get('batch_metrics', {})
                                    tau_list = [1,2,3,5]
                                    ninl = [float(bm.get(f'N_inlier_epi@{t}_mean', 0.0)) for t in tau_list]
                                    ms   = [float(bm.get(f'MS_epi@{t}_mean', 0.0)) for t in tau_list]
                                    # print(f"[eval np BATCH] {scene_desc} b={batch_idx} N_M_mean={bm.get('N_M_mean', 0.0)} inliers@{tau_list}={ninl} MS@{tau_list}={ms} se_mean={bm.get('sampson_mean_mean', 0.0)}")
                                except Exception:
                                    pass
                                # Save per-pair rows using mapping to original meta indices
                                for j, res in enumerate(pack_np['individual_results']):
                                    i_meta = np_meta_indices[j] if j < len(np_meta_indices) else None
                                    row = self.result_to_row(res)
                                    # attach per-item nonplanar forward time if available
                                    try:
                                        row['forward_ms'] = float(np_forward_times[j])
                                    except Exception:
                                        pass
                                    row['_batch_idx'] = batch_idx
                                    row['_sample_idx'] = i_meta if i_meta is not None else j
                                    p0 = meta[i_meta].get('image0_path') if (i_meta is not None and i_meta < len(meta)) else None
                                    p1 = meta[i_meta].get('image1_path') if (i_meta is not None and i_meta < len(meta)) else None
                                    # Prefer explicit NP path for image1 when available
                                    try:
                                        if ((i_meta is not None) and (i_meta < len(meta)) and ('image1_np_path' in meta[i_meta])):
                                            p1 = meta[i_meta]['image1_np_path']
                                    except Exception:
                                        pass
                                    row['_safe_name'] = safe_name_from_paths(p0, p1)
                                    # Standardize leading identifiers
                                    row['Detector'] = label_name
                                    row['Modality'] = modality
                                    row['Planarity'] = 'nonplanar'
                                    row['scene_type'] = scene_type
                                    row['scene'] = scene_name
                                    if (i_meta is not None) and ('scene' in meta[i_meta]):
                                        row['scene'] = meta[i_meta]['scene']
                                    if p0 is not None:
                                        row['image0_path'] = p0
                                    if p1 is not None:
                                        row['image1_path'] = p1
                                    # Add nonplanar step and original keypoint counts
                                    try:
                                        row['nonplanar_step'] = int(self.nonplanar_step)
                                    except Exception:
                                        row['nonplanar_step'] = int(1)
                                    try:
                                        row['num_keypoints_0'] = float(len(pred_nonplanar['kpts0'][j]))
                                    except Exception:
                                        row['num_keypoints_0'] = None
                                    try:
                                        row['num_keypoints_1'] = float(len(pred_nonplanar['kpts1'][j]))
                                    except Exception:
                                        row['num_keypoints_1'] = None
                                    # Ensure field exists for header stability prior to match computation
                                    if 'np_reason' not in row:
                                        row['np_reason'] = ''
                                    
                                    # Compute additional robust pose/homography metrics
                                    try:
                                        if j < len(pred_nonplanar['kpts0']) and j < len(pred_nonplanar['kpts1']):
                                            k0_j = pred_nonplanar['kpts0'][j]
                                            k1_j = pred_nonplanar['kpts1'][j]
                                            d0_j = pred_nonplanar['desc0'][j]
                                            d1_j = pred_nonplanar['desc1'][j]
                                            
                                            # Get camera parameters for this pair
                                            K0_j = cam_pack['K0'][j]
                                            K1_j = cam_pack['K1'][j]
                                            R_j = cam_pack['R_01'][j]
                                            t_j = cam_pack['t_01'][j]
                                            
                                            # Compute robust pose errors if pose was estimated
                                            if 'R_est' in res and 'R_gt' in res:
                                                try:
                                                    t_err_robust, R_err_robust = relative_pose_error_robust(
                                                        res['R_est'], res['t_est'], res['R_gt'], res['t_gt']
                                                    )
                                                    row['pose_t_deg_robust'] = float(t_err_robust)
                                                    row['pose_R_deg_robust'] = float(R_err_robust)
                                                except Exception:
                                                    row['pose_t_deg_robust'] = 90.0
                                                    row['pose_R_deg_robust'] = 180.0
                                            else:
                                                row['pose_t_deg_robust'] = 90.0
                                                row['pose_R_deg_robust'] = 180.0
                                            
                                            # Select matches using centralized MNN helper
                                            if len(d0_j) > 0 and len(d1_j) > 0:
                                                mi, _ = mnn_match_descriptors(
                                                    d0_j.detach().cpu().numpy(),
                                                    d1_j.detach().cpu().numpy(),
                                                    sim_threshold=None,
                                                    ratio_threshold=None
                                                )
                                                # Record number of matches for aggregation
                                                try:
                                                    row['N_M'] = int(mi.shape[0])
                                                except Exception:
                                                    row['N_M'] = int(mi.size) if hasattr(mi, 'size') else 0
                                                if mi.size > 0:
                                                    idx0 = torch.as_tensor(mi[:, 0], device=d0_j.device, dtype=torch.long)
                                                    idx1 = torch.as_tensor(mi[:, 1], device=d1_j.device, dtype=torch.long)
                                                    mk0_np = k0_j[idx0].detach().cpu().numpy().astype(np.float32)
                                                    mk1_np = k1_j[idx1].detach().cpu().numpy().astype(np.float32)
                                                else:
                                                    mk0_np = np.empty((0,2), dtype=np.float32)
                                                    mk1_np = np.empty((0,2), dtype=np.float32)
                                                try:
                                                    if int(row.get('N_M', 0)) == 0:
                                                        row['np_reason'] = 'no_matches'
                                                        print(f"Nonplanar: no matches for {detector_name} {modality} scene={scene_name} (b={batch_idx}, i={i_meta}).")
                                                    else:
                                                        row['np_reason'] = ''
                                                except Exception:
                                                    pass
                                                
                                                # Compute homography error
                                                if len(mk0_np) >= 4:
                                                    h_err, H_gt, H_est, h_mask = compute_homography_error_robust(
                                                        K0_j, K1_j, R_j, t_j, mk0_np, mk1_np
                                                    )
                                                    row['homography_error_px'] = float(h_err)
                                                    if h_mask is not None:
                                                        row['homography_inlier_ratio'] = float(h_mask.sum()) / float(len(h_mask))
                                                    else:
                                                        row['homography_inlier_ratio'] = 0.0
                                                else:
                                                    row['homography_error_px'] = float('inf')
                                                    row['homography_inlier_ratio'] = 0.0
                                            else:
                                                row['homography_error_px'] = float('inf')
                                                row['homography_inlier_ratio'] = 0.0
                                    except Exception as e:
                                        row['homography_error_px'] = float('inf')
                                        row['homography_inlier_ratio'] = 0.0
                                        row['pose_t_deg_robust'] = 90.0
                                        row['pose_R_deg_robust'] = 180.0
                                    
                                    # Append row immediately to per-scene CSV
                                    self.append_csv_row(nonplanar_csv_path, row)

                                mod_state[modality]['nonplanar_batch_metrics'].append(pack_np['batch_metrics'])
                                scene_nonplanar_acc[modality].append(pack_np['batch_metrics'])
                            except Exception as e:
                                print(f"Error: nonplanar batch eval failed (scene={scene_name}, b={batch_idx}, modality={modality}): {e}")
                            
                        elif enable_nonplanar:
                            # Require K,R,t for pose evaluation; if missing, hard error (no silent zeros)
                            missing = [req for req in ['K0', 'K1', 'R_01', 't_01'] if req not in batch]
                            if missing:
                                raise RuntimeError(f"Nonplanar evaluation requires K0,K1,R_01,t_01 but missing: {missing} (scene={scene_name}, batch={batch_idx})")

                # After finishing a scene: write per-modality success curves
                for modality in list(mod_state.keys()):
                    try:
                        from models.utils.metrics import success_curve as _success_curve
                        from models.utils.plotting import save_success_curve_plot as _save_curve
                        scene_key = (modality, scene_type, scene_name)
                        nonplanar_dir = mod_state[modality]['nonplanar_dir']
                        nonplanar_imgs_dir = nonplanar_dir / 'imgs' / scene_safe
                        if self.save_success_curves and hasattr(self, '_scene_pose_store') and scene_key in getattr(self, '_scene_pose_store', {}):
                            P_list = self._scene_pose_store[scene_key]
                            if len(P_list) > 0:
                                P = torch.cat([torch.as_tensor(x, dtype=torch.float32) for x in P_list])
                                taus, suc = _success_curve(P, 45.0)
                                title = f"Pose success (scene={scene_name})"
                                out_path = (nonplanar_imgs_dir / 'nonplanar_pose_success_full.png')
                                _save_curve(taus, suc, title, xlabel='Threshold (deg)', ylabel='Success rate', out_path=str(out_path))
                        if self.save_success_curves and hasattr(self, '_scene_se_store') and scene_key in getattr(self, '_scene_se_store', {}):
                            S_list = self._scene_se_store[scene_key]
                            if len(S_list) > 0:
                                SE = torch.cat([torch.as_tensor(x, dtype=torch.float32) for x in S_list])
                                # Plot up to 20 px (full) and 5 px (standard)
                                taus_px, suc_px = _success_curve(SE, 20.0)
                                title = f"Epipolar success (scene={scene_name})"
                                out_path = (nonplanar_imgs_dir / 'nonplanar_epi_success_full.png')
                                _save_curve(taus_px, suc_px, title, xlabel='Threshold (px)', ylabel='Success rate', out_path=str(out_path))
                                try:
                                    taus_px5, suc_px5 = _success_curve(SE, 5.0)
                                    out_path5 = (nonplanar_imgs_dir / 'nonplanar_epi_success_5px.png')
                                    _save_curve(taus_px5, suc_px5, title + ' (≤5px)', xlabel='Threshold (px)', ylabel='Success rate', out_path=str(out_path5))
                                except Exception:
                                    pass
                    except Exception as e:
                        print(f"Warning: failed to save per-scene continuous AUC curves for {scene_name}: {e}")

                # Per-scene summary (Detector | Modality | Scene | Planarity | metrics)
                try:
                    def aggregate_plain(metrics_list: List[Dict[str, Any]]):
                        if not metrics_list:
                            return {}
                        agg: Dict[str, float] = {}
                        keys = set().union(*[m.keys() for m in metrics_list])
                        # Define keys for which we want median in addition to mean
                        median_keys = {
                            'pose_R_deg', 'pose_t_deg', 'pose_R_deg_robust', 'pose_t_deg_robust',
                            'sampson_mean', 'sampson_median',
                            'homography_error_px', 'tri_rmse_px'
                        }
                        for k in keys:
                            vals: List[float] = []
                            for m in metrics_list:
                                v = m.get(k, None)
                                if isinstance(v, torch.Tensor):
                                    vals.append(float(v.mean().item()) if v.numel() > 1 else float(v.item()))
                                elif isinstance(v, (int, float)):
                                    vals.append(float(v))
                            if vals:
                                agg[f'{k}_mean'] = float(np.mean(vals))
                                agg[f'{k}_std'] = float(np.std(vals))
                                # Add median for select metrics
                                if k in median_keys:
                                    agg[f'{k}_median'] = float(np.median(vals))
                        
                        # Compute pose estimation success rate.
                        # Failed pose estimation: pose_R_deg=180, pose_t_deg=90
                        if 'pose_R_deg' in keys:
                            pose_r_vals = []
                            for m in metrics_list:
                                v = m.get('pose_R_deg', None)
                                if isinstance(v, torch.Tensor):
                                    pose_r_vals.append(float(v.mean().item()) if v.numel() > 1 else float(v.item()))
                                elif isinstance(v, (int, float)):
                                    pose_r_vals.append(float(v))
                            if pose_r_vals:
                                # Count pairs where pose estimation succeeded (R_err < 180)
                                successful = sum(1 for r in pose_r_vals if r < 180.0)
                                agg['pose_success_rate'] = float(successful) / float(len(pose_r_vals))
                                agg['processed_pairs'] = int(successful)
                                agg['failed_pairs'] = int(len(pose_r_vals) - successful)
                        
                        # Also compute success rate for the robust pose variant
                        if 'pose_R_deg_robust' in keys:
                            pose_r_robust_vals = []
                            for m in metrics_list:
                                v = m.get('pose_R_deg_robust', None)
                                if isinstance(v, torch.Tensor):
                                    pose_r_robust_vals.append(float(v.mean().item()) if v.numel() > 1 else float(v.item()))
                                elif isinstance(v, (int, float)):
                                    pose_r_robust_vals.append(float(v))
                            if pose_r_robust_vals:
                                successful_robust = sum(1 for r in pose_r_robust_vals if r < 180.0)
                                agg['pose_success_rate_robust'] = float(successful_robust) / float(len(pose_r_robust_vals))
                                agg['processed_pairs_robust'] = int(successful_robust)
                                agg['failed_pairs_robust'] = int(len(pose_r_robust_vals) - successful_robust)

                        return agg
                    scene_summary_path = self.output_dir / 'scene_summary.csv'
                    for modality in list(mod_state.keys()):
                        if scene_planar_acc[modality]:
                            agg_scene_p = aggregate_plain(scene_planar_acc[modality])
                            row_p = {
                                'Detector': label_name,
                                'Modality': modality,
                                'Scene': scene_name,
                                'Planarity': 'planar',
                            }
                            row_p.update(agg_scene_p)
                            self.append_csv_row(scene_summary_path, row_p)
                        if scene_nonplanar_acc[modality]:
                            agg_scene_np = aggregate_plain(scene_nonplanar_acc[modality])
                            row_np = {
                                'Detector': label_name,
                                'Modality': modality,
                                'Scene': scene_name,
                                'Planarity': 'nonplanar',
                            }
                            row_np.update(agg_scene_np)
                            self.append_csv_row(scene_summary_path, row_np)
                except Exception as e:
                    print(f"Warning: failed to write scene summary for {scene_name}: {e}")

        # Final summaries per modality
        def aggregate(metrics_list: List[Dict[str, Any]], prefix: str):
            if not metrics_list:
                return {}
            agg: Dict[str, float] = {}
            keys = set().union(*[m.keys() for m in metrics_list])
            for k in keys:
                vals: List[float] = []
                for m in metrics_list:
                    v = m.get(k, None)
                    if isinstance(v, torch.Tensor):
                        vals.append(float(v.mean().item()) if v.numel() > 1 else float(v.item()))
                    elif isinstance(v, (int, float)):
                        vals.append(float(v))
                if vals:
                    agg[f'{prefix}/{k}_mean'] = float(np.mean(vals))
                    agg[f'{prefix}/{k}_std'] = float(np.std(vals))
            return agg

        for modality, st in mod_state.items():
            final[f'{modality}/planar_batches'] = len(st['planar_batch_metrics'])
            final[f'{modality}/nonplanar_batches'] = len(st['nonplanar_batch_metrics'])
            agg_p = aggregate(st['planar_batch_metrics'], f'{modality}/planar')
            agg_np = aggregate(st['nonplanar_batch_metrics'], f'{modality}/nonplanar')
            final.update(agg_p)
            final.update(agg_np)

        # No per-detector/modality CSVs here; a single root-level summary.csv is written in main()

        # Ensure any pending buffered CSV rows are flushed for this detector
        try:
            self.flush_all_csv_buffers()
        except Exception:
            pass
        
        # Print summary of key nonplanar metrics
        try:
            print(f"\n{'='*60}")
            print(f"Summary for {label_name}:")
            print(f"{'='*60}")
            for modality in eval_modalities:
                if modality not in mod_state:
                    continue
                print(f"\n{modality} Modality:")
                # Planar summary
                if f'{modality}/planar_batches' in final and final[f'{modality}/planar_batches'] > 0:
                    print(f"  Planar pairs: {final[f'{modality}/planar_batches']}")
                    for metric in ['Repeatability', 'MS@3', 'MMA']:
                        key_mean = f'{modality}/planar/{metric}_mean'
                        key_std = f'{modality}/planar/{metric}_std'
                        if key_mean in final:
                            mean_val = final[key_mean]
                            std_val = final.get(key_std, 0.0)
                            print(f"    {metric}: {mean_val:.4f} ± {std_val:.4f}")
                # Nonplanar summary (prioritize pose errors)
                if f'{modality}/nonplanar_batches' in final and final[f'{modality}/nonplanar_batches'] > 0:
                    print(f"  Nonplanar pairs: {final[f'{modality}/nonplanar_batches']}")

                    # Success rate: processed_pairs / total_pairs
                    success_rate_key = f'{modality}/nonplanar/pose_success_rate_mean'
                    processed_key = f'{modality}/nonplanar/processed_pairs_mean'
                    failed_key = f'{modality}/nonplanar/failed_pairs_mean'
                    if success_rate_key in final:
                        sr = final[success_rate_key] * 100
                        proc = int(final.get(processed_key, 0))
                        fail = int(final.get(failed_key, 0))
                        print(f"    Pose Success Rate: {sr:.1f}% ({proc} successful, {fail} failed)")
                    
                    # PRIMARY: Pose errors (most important)
                    print(f"    --- Pose Estimation Errors ---")
                    # Show both variants: default (eval_class) and robust
                    for metric, label in [('pose_t_deg', 'Translation Error'), ('pose_R_deg', 'Rotation Error')]:
                        key_mean = f'{modality}/nonplanar/{metric}_mean_mean'
                        key_median = f'{modality}/nonplanar/{metric}_mean_median'
                        key_std = f'{modality}/nonplanar/{metric}_mean_std'
                        # Robust variant
                        key_mean_robust = f'{modality}/nonplanar/{metric}_robust_mean_mean'
                        key_median_robust = f'{modality}/nonplanar/{metric}_robust_mean_median'

                        if key_mean in final and key_mean_robust in final:
                            mean_val = final[key_mean]
                            median_val = final.get(key_median, mean_val)
                            std_val = final.get(key_std, 0.0)
                            mean_val_robust = final[key_mean_robust]
                            median_val_robust = final.get(key_median_robust, mean_val_robust)
                            print(f"      {label} (eval): {mean_val:.2f}° ± {std_val:.2f}° (median: {median_val:.2f}°)")
                            print(f"      {label} (robust): {mean_val_robust:.2f}° (median: {median_val_robust:.2f}°)")
                        elif key_mean in final:
                            mean_val = final[key_mean]
                            median_val = final.get(key_median, mean_val)
                            std_val = final.get(key_std, 0.0)
                            print(f"      {label}: {mean_val:.2f}° ± {std_val:.2f}° (median: {median_val:.2f}°)")
                    
                    # Success rates (very important) - dynamically show configured thresholds
                    print(f"    --- Pose Success Rates ---")
                    # Get configured pose thresholds from evaluator config
                    try:
                        pose_threshs = self.unified_evaluator.config.pose_auc_degs
                    except Exception:
                        pose_threshs = (1.0, 3.0, 5.0, 10.0, 20.0)
                    
                    for thresh in pose_threshs:
                        t_int = int(thresh)
                        key = f'{modality}/nonplanar/pose_S@{t_int}_mean'
                        if key in final:
                            print(f"      Success@{t_int}°: {final[key]*100:.1f}%")
                    
                    # Inlier ratio (indicates pose estimation quality)
                    inlier_key = f'{modality}/nonplanar/pose_inlier_ratio_mean'
                    if inlier_key in final:
                        print(f"      Avg Inlier Ratio: {final[inlier_key]*100:.1f}%")
                    
                    # SECONDARY: Epipolar and supplementary metrics
                    print(f"    --- Epipolar Metrics ---")
                    for metric, label in [('sampson_mean', 'Sampson Error'), ('sampson_median', 'Sampson Median')]:
                        key_mean = f'{modality}/nonplanar/{metric}_mean_mean'
                        if key_mean in final:
                            print(f"      {label}: {final[key_mean]:.2f}px")
                    
                    # Homography (supplementary)
                    h_key = f'{modality}/nonplanar/homography_error_px_mean_mean'
                    if h_key in final:
                        print(f"    --- Supplementary ---")
                        print(f"      Homography Error: {final[h_key]:.2f}px")
        except Exception as e:
            print(f"Warning: failed to print summary: {e}")
        
        return final

def create_wandb_logger_for_detector(detector_name: str, config_path: str, project_name: str = "hykey_project", run_name: Optional[str] = None, group: Optional[str] = None) -> Optional[WandbLogger]:
    if not _HAVE_WANDB:
        return None
    cfg = load_config(config_path)
    wandb_logger = WandbLogger(
        project=project_name,
        name=(run_name or detector_name),
        group=group,
        tags=["detector-evaluation", "planar", "nonplanar", detector_name]
    )
    try:
        wandb_logger.log_hyperparams(cfg)
    except Exception:
        pass
    return wandb_logger

def main():
    parser = argparse.ArgumentParser(description="Evaluate HyKey on planar and nonplanar scenes")
    parser.add_argument("--config", type=str, default="configs/evaluate_hykey.yaml",
                        help="Path to evaluation configuration file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found")
        return

    cfg = load_config(args.config)
    eval_cfg = cfg.get('evaluation', {}) if isinstance(cfg, dict) else {}

    # Load runtime parameters entirely from YAML
    device = str(eval_cfg.get('device', ("cuda" if torch.cuda.is_available() else "cpu")))
    output_dir = str(eval_cfg.get('output_dir', 'hykey_eval'))
    nonplanar_step = int(eval_cfg.get('nonplanar_step', 1))
    enable_planar = bool(eval_cfg.get('enable_planar', True))
    enable_nonplanar = bool(eval_cfg.get('enable_nonplanar', True))
    max_keypoints = int(eval_cfg.get('max_keypoints', 5000))
    wandb_project = str(eval_cfg.get('wandb_project', 'hykey_project'))
    wandb_group = eval_cfg.get('wandb_group', None)
    if wandb_group is not None:
        wandb_group = str(wandb_group)
        # Belt-and-suspenders: env var is honoured by wandb for any run created in-process.
        os.environ['WANDB_RUN_GROUP'] = wandb_group
    use_wandb = bool(eval_cfg.get('use_wandb', True))
    list_models = bool(eval_cfg.get('list_models', False))
    models_to_evaluate = eval_cfg.get('models', ['hykey'])
    eval_modalities = eval_cfg.get('modalities', [])
    if not isinstance(eval_modalities, list):
        eval_modalities = []
    modality_slug = str(eval_modalities[0]).lower() if len(eval_modalities) == 1 else None
    # Optional: model -> list of checkpoint directories; auto-pick best ckpt and find training yaml
    model_checkpoint_dirs = eval_cfg.get('model_checkpoint_dirs', {}) if isinstance(eval_cfg.get('model_checkpoint_dirs', {}), dict) else {}

    if list_models:
        print("Available models:")
        for m in sorted(get_available_models()):
            print(f"  - {m}")
        return

    available = set(get_available_models())
    valid_models = [m for m in models_to_evaluate if m in available]
    if not valid_models:
        print("Error: No valid models found to evaluate")
        print(f"Available models: {sorted(available)}")
        return

    print(f"Will evaluate {len(valid_models)} models: {valid_models}")

    # Representative pass: use the step(s) from config. Default is a single pass at
    # eval_cfg['nonplanar_step']; optionally provide eval_cfg['nonplanar_steps'] (list) to sweep.
    nonplanar_steps = eval_cfg.get('nonplanar_steps') or [nonplanar_step]
    base_output_dir = output_dir
    for step_idx, nonplanar_step in enumerate(nonplanar_steps):
        output_dir = base_output_dir if len(nonplanar_steps) == 1 else f"{base_output_dir}_step_{nonplanar_step}"
        evaluator = CombinedDetectorEvaluator(
            args.config, device,
            None,  # per-detector loggers created per detector if enabled
            nonplanar_step=nonplanar_step,
            output_dir=output_dir,
        )

        all_results: Dict[str, Any] = {}

        for detector_name in valid_models:
            print(f"\n{'='*60}")
            print(f"Starting evaluation for: {detector_name}")
            print(f"{'='*60}")

            # Determine which checkpoint directories to run (if any)
            ckpt_dirs = []
            if detector_name in model_checkpoint_dirs and isinstance(model_checkpoint_dirs[detector_name], list):
                ckpt_dirs = [str(p) for p in model_checkpoint_dirs[detector_name]]
            # If none given, run a single variant without learned checkpoint (for classical models)
            if not ckpt_dirs:
                ckpt_dirs = [None]

            results_for_model = {}
            for variant_idx, ckpt_dir in enumerate(ckpt_dirs):
                variant_label = f"{detector_name}"
                train_yaml = None
                best_ckpt = None
                if ckpt_dir:
                    try:
                        dpath = Path(ckpt_dir)
                        train_yaml = find_training_yaml(dpath)
                        best_ckpt = find_best_checkpoint(dpath)
                        # Label each variant by its checkpoint directory name
                        variant_label = f"{detector_name}_{dpath.name}"
                    except Exception as e:
                        print(f"Warning: failed to inspect checkpoint dir {ckpt_dir}: {e}")
                elif len(ckpt_dirs) > 1:
                    variant_label = f"{detector_name}_v{variant_idx}"
                if modality_slug and not str(variant_label).lower().startswith(f'{modality_slug}_'):
                    variant_label = f"{modality_slug}_{variant_label}"

                wandb_logger = None
                if _HAVE_WANDB and use_wandb:
                    try:
                        log_name = variant_label if str(variant_label).startswith('eval_') else f"eval_{variant_label}"
                        wandb_logger = create_wandb_logger_for_detector(detector_name, args.config, wandb_project, run_name=log_name, group=wandb_group)
                        print(f"WandB logging enabled for {variant_label}: {wandb_logger.experiment.name}")
                    except Exception as e:
                        print(f"Warning: Failed to initialize WandB logging for {variant_label}: {e}")
                        print("Continuing without WandB logging...")

                evaluator.wandb_logger = wandb_logger

                # Build minimal overrides: point to best checkpoint and training yaml (wrapper will parse yaml for model)
                overrides_for_model = {}
                if best_ckpt:
                    overrides_for_model['model_path'] = str(best_ckpt)
                if train_yaml:
                    overrides_for_model['model_yaml'] = str(train_yaml)
                model_overrides = eval_cfg.get('model_overrides', {})
                if isinstance(model_overrides, dict):
                    specific_overrides = model_overrides.get(detector_name, {})
                    if isinstance(specific_overrides, dict):
                        overrides_for_model.update(specific_overrides)

                # Planar metrics are scene-geometry (not step-dependent), so compute them
                # only on the first pass to avoid redundant work when sweeping steps.
                run_planar = bool(enable_planar) and (step_idx == 0)
                result = evaluator.evaluate_detector(
                    detector_name,
                    max_num_keypoints=max_keypoints,
                    enable_planar=run_planar,
                    enable_nonplanar=enable_nonplanar,
                    overrides_for_model=overrides_for_model,
                    display_name=variant_label,
                )

                if wandb_logger and hasattr(wandb_logger, 'experiment'):
                    try:
                        flat_metrics = drop_eval_modality_prefix(
                            clean_eval_metrics(flatten_for_wandb('detector_eval', result))
                        )
                        flat_metrics = canonicalize_robust_metrics(flat_metrics)
                        if flat_metrics:
                            wandb_logger.experiment.log(flat_metrics)
                            wandb_logger.experiment.summary.update(flat_metrics)
                            print(f"Logged {len(flat_metrics)} scalar eval metrics to WandB for {variant_label}")
                    except Exception as e:
                        print(f"Warning: failed to log eval metrics to WandB for {variant_label}: {e}")

                results_for_model[variant_label] = result

                if wandb_logger and hasattr(wandb_logger, 'experiment'):
                    wandb_logger.experiment.finish()
                    print(f"Closed WandB run for {variant_label}")

            all_results.update(results_for_model)

        # Save top-level summary JSON + CSV
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / 'summary.json'
        csv_path = out_dir / 'summary.csv'

        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.detach().cpu().numpy().tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            return obj

        import json
        with open(json_path, 'w') as f:
            json.dump(convert(all_results), f, indent=2)
        print(f"Summary JSON saved to: {json_path}")

        # Flatten to CSV rows: one row per detector per modality, with leading columns
        flat_rows = []
        for det, res in all_results.items():
            for modality in ['HSI', 'RGB']:
                prefix = f"{modality}/"
                mod_fields = {k[len(prefix):]: convert(v) for k, v in res.items() if isinstance(k, str) and k.startswith(prefix)}
                if not mod_fields:
                    continue
                row = {'Detector': det, 'Modality': modality}
                row.update(mod_fields)
                flat_rows.append(row)
        df_summary = pd.DataFrame(flat_rows)
        # Ensure leading columns order
        if not df_summary.empty:
            lead_cols = [c for c in ['Detector', 'Modality'] if c in df_summary.columns]
            remaining = [c for c in df_summary.columns if c not in lead_cols]
            df_summary = df_summary[lead_cols + remaining]
        df_summary.to_csv(csv_path, index=False)
        print(f"Summary CSV saved to: {csv_path}")



if __name__ == "__main__":
    main()


