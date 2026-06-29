import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union
import random

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import pandas as pd

from data_processing.utils.hsi_utils import (
    radiometric_calibration, demosaic_aligned_ximea, normalize_hsi_bands, load_band_stats,
)
from data_processing.utils.registration_utils import warp_image
from data_processing.utils.utils import undistort_image_with_K, quaternion_to_rotation_matrix
from data_processing.utils.calibrate_utils import CameraCalibration, load_calibration_data
from data_processing.utils.augmentation_utils import (
    generate_warped_image, apply_photometric_augmentation,
    resolve_warp_params, WARP_PRESETS,
)

_NP_MIN_ROT_DEG = 0.0
_NP_MIN_T_NORM  = 0.000003   # metres; rig-scale dependent
_NP_MAX_T_NORM  = 1.50
_NP_TRY_CANDIDATES = 16
_NP_PREFER_LARGER_BASELINE = True


class SpectralDataset(Dataset):
    """HyKey Dataset loader for hyperspectral (and optional RGB) frame pairs.

    Each sample provides an HSI frame plus, when enabled, a synthetically warped
    copy (planar homography supervision) and a second temporal frame with the
    relative pose between them (epipolar supervision). RGB frames are optional and,
    when present, can be registered into the HSI frame using the precomputed
    registration homography shipped with the calibration data.
    """

    def __init__(
        self,
        data_root: str,
        acquisition_folders: Optional[List[str]] = None,
        load_poses: bool = False,
        load_rgb: bool = True,
        load_hsi: bool = True,
        aug_strength: float = 0.5,
        registration: Optional[str] = None,        # None or any truthy value -> register RGB<->HSI
        calibration_dir: Optional[str] = None,     # calibration XMLs + homographies/; defaults to <data_root>/calibration
        target_space: str = 'rgb',                 # 'rgb' or 'hsi'
        undistort: bool = False,
        split: str = 'all',
        test_size: Optional[float] = None,
        val_size: float = 0.2,
        random_state: int = 42,
        radiometric_calibration: bool = True,
        warp_augmentation: bool = True,            # planar synthetic-homography view
        photometric_augmentation: bool = True,
        photometric_augmentation_nonplanar: bool = False,
        max_nonplanar_offset: int = 100,           # max temporal offset in frames
        always_include_nonplanar: bool = True,
        nonplanar_stride: int = 1,
        band_stats_path: Optional[str] = None,
        warp_difficulty: Union[str, dict] = 'paper',   # 'paper' | 'harder' | dict of overrides
        warp_params: Optional[dict] = None,            # per-key overrides applied on top of the preset
    ):
        if not (load_rgb or load_hsi):
            raise ValueError("At least one of load_rgb or load_hsi must be True")

        self.data_root = Path(data_root)
        self.load_poses = load_poses
        self.load_rgb = load_rgb
        self.load_hsi = load_hsi
        self.aug_strength = aug_strength
        self.registration = registration
        self.target_space = target_space
        self.undistort = undistort
        self.radiometric_calibration = radiometric_calibration

        # Fixed dataset-level per-band normalization stats. Reflectance and raw HSI have
        # different intensity scales, so we pick the matching stat set. Falls back to
        # per-image min-max if the stats file/key is unavailable.
        self.band_stats_path = band_stats_path
        self._band_stats_key = 'spectral_reflectance' if radiometric_calibration else 'spectral_raw'
        self._band_stats = load_band_stats(band_stats_path, self._band_stats_key)
        self._warned_missing_band_stats = False
        if self._band_stats is None:
            print(
                f"WARNING: SpectralDataset could not load band stats "
                f"('{self._band_stats_key}' from {band_stats_path}); "
                f"falling back to per-image min-max normalization."
            )

        self.warp_augmentation = warp_augmentation
        self.warp_params = resolve_warp_params(warp_difficulty)
        if warp_params:
            self.warp_params.update(warp_params)
        self.photometric_augmentation = photometric_augmentation
        self.photometric_augmentation_nonplanar = photometric_augmentation_nonplanar
        self.max_nonplanar_offset = max_nonplanar_offset
        self.always_include_nonplanar = always_include_nonplanar
        self.nonplanar_stride = max(1, int(nonplanar_stride))

        self._np_seen = 0
        self._np_growth_steps = 5000
        self._np_start_min_off = 1
        self._np_start_max_off = max(5, min(15, int(self.max_nonplanar_offset // 6)))

        # Calibration ships inside the dataset: default to <data_root>/calibration so the
        # intrinsics + RGB<->HSI homographies always travel with the acquisition folders.
        self.calibration_dir = Path(calibration_dir) if calibration_dir is not None else self.data_root / 'calibration'

        self.rgb_calib = None
        self.hsi_calib = None
        rgb_calib_file = self.calibration_dir / 'RGB' / 'calibration_data.xml'
        hsi_calib_file = self.calibration_dir / 'HSI' / 'calibration_data.xml'
        if rgb_calib_file.exists() and self.load_rgb:
            self.rgb_calib = load_calibration_data(rgb_calib_file)
        if hsi_calib_file.exists() and self.load_hsi:
            self.hsi_calib = load_calibration_data(hsi_calib_file)
        self.K_rgb = self.rgb_calib['camera_matrix'] if isinstance(self.rgb_calib, dict) else None
        self.K_hsi = self.hsi_calib['camera_matrix'] if isinstance(self.hsi_calib, dict) else None

        self.homography = None
        if self.registration is not None and self.load_rgb:
            name = 'H_hsi2rgb.npy' if target_space == 'rgb' else 'H_rgb2hsi.npy'
            homography_path = self.calibration_dir / 'homographies' / name
            if not homography_path.exists():
                raise ValueError(f"No homography file found at {homography_path}")
            self.homography = np.load(homography_path)

        if acquisition_folders is None:
            raise ValueError("acquisition_folders must be provided. "
                             "Use training utilities to discover and filter folders.")
        self.acquisition_folders = acquisition_folders

        self.data_pairs = []
        self.rgb_frame_logs = {}
        self.hsi_frame_logs = {}
        self.frame_logs = {}
        self.white_dark_images = {}
        self.load_calibration_data()
        self.load_data_pairs()
        self.split_dataset(split, test_size, val_size, random_state)

    @staticmethod
    def debayer_rgb(raw_rgb: np.ndarray) -> np.ndarray:
        """Debayer raw RGB-camera frames as RGGB while preserving current orientation."""
        return cv2.flip(cv2.cvtColor(cv2.flip(raw_rgb, 1), cv2.COLOR_BayerRG2RGB), 1)

    def split_dataset(self, split, test_size, val_size, random_state):
        from data_processing.utils.dataset_utils import split_dataset_pairs
        self.data_pairs = split_dataset_pairs(self.data_pairs, split, test_size, val_size, random_state)

    def __getitem__(self, idx):
        data_pair = self.data_pairs[idx]
        folder = data_pair['folder']
        hsi_idx = data_pair['hsi_idx']
        rgb_idx = data_pair['rgb_idx']
        folder_path = self.data_root / folder

        hsi_data = None
        K0_new = None
        if self.load_hsi:
            hsi_path = folder_path / "HSI" / f"{hsi_idx:08d}.npy"
            hsi_data = self.preprocess_hsi(hsi_path, folder_path, hsi_idx)
            hsi_data = self.normalize_hsi(hsi_data)

            if self.undistort and self.hsi_calib is not None:
                hsi_data, K0_new = undistort_image_with_K(
                    hsi_data, self.hsi_calib['camera_matrix'], self.hsi_calib['dist_coeffs'])

        rgb_data = None
        if self.load_rgb:
            rgb_path = folder_path / "RGB" / f"{rgb_idx:08d}.npy"
            rgb_data = self.debayer_rgb(np.load(rgb_path)).astype(np.float32) / 255.0
            if self.undistort and self.rgb_calib is not None:
                rgb_data, _ = undistort_image_with_K(
                    rgb_data, self.rgb_calib['camera_matrix'], self.rgb_calib['dist_coeffs'])

        overlap_bbox = None
        if self.registration is not None and self.homography is not None and rgb_data is not None:
            if self.target_space == 'rgb' and hsi_data is not None:
                hsi_data = warp_image(hsi_data, self.homography, (rgb_data.shape[1], rgb_data.shape[0]))
            elif self.target_space == 'hsi':
                h, w = (hsi_data.shape[:2] if hsi_data is not None else (272, 512))
                overlap_bbox = self.get_overlap_bbox(rgb_data.shape[:2], (h, w))
                rgb_data = warp_image(rgb_data, self.homography, (w, h))

        hsi_tensor = None
        if hsi_data is not None:
            if hsi_data.ndim == 3:
                hsi_tensor = torch.from_numpy(hsi_data).permute(2, 0, 1)

        rgb_tensor = None
        if rgb_data is not None:
            if rgb_data.ndim == 3:
                rgb_tensor = torch.from_numpy(rgb_data).permute(2, 0, 1)
            elif rgb_data.ndim == 2:
                rgb_tensor = torch.from_numpy(np.stack([rgb_data] * 3, axis=2)).permute(2, 0, 1)

        if hsi_tensor is not None and torch.isnan(hsi_tensor).any():
            hsi_tensor = torch.nan_to_num(hsi_tensor, nan=0.0)
        if rgb_tensor is not None and torch.isnan(rgb_tensor).any():
            rgb_tensor = torch.nan_to_num(rgb_tensor, nan=0.0)

        hsi_pose = None
        rgb_pose = None
        if self.load_poses:
            if self.load_hsi:
                hsi_pose = self.pose_from_log(self.hsi_frame_logs[folder], hsi_idx)
            if self.load_rgb:
                rgb_pose = self.pose_from_log(self.rgb_frame_logs[folder], rgb_idx)

        warped_hsi_tensor = None
        warped_rgb_tensor = None
        H_mat = None
        K0 = K0_new if (self.undistort and self.hsi_calib is not None) else \
            (self.hsi_calib['camera_matrix'] if self.hsi_calib is not None else None)
        K1 = None
        R_01 = None
        t_01 = None

        if self.warp_augmentation:
            if hsi_tensor is not None:
                warped_hsi_tensor, H_mat = generate_warped_image(
                    hsi_tensor, sample_idx=int(idx) % (2**32 - 1), **self.warp_params)
            if rgb_tensor is not None:
                if H_mat is not None:
                    warped_rgb_tensor, _ = generate_warped_image(
                        rgb_tensor, H_mat, sample_idx=int(idx) % (2**32 - 1))
                else:
                    warped_rgb_tensor, H_mat = generate_warped_image(
                        rgb_tensor, sample_idx=int(idx) % (2**32 - 1), **self.warp_params)

        include_np = self.always_include_nonplanar
        hsi_np_tensor = None
        rgb_np_tensor = None
        if include_np:
            frame_log = self.hsi_frame_logs[folder]
            hsi_idx2, R_01, t_01 = self.pick_nonplanar_index_smart(
                frame_log, hsi_idx, min_off=1, max_off=max(1, self.max_nonplanar_offset))
            self._np_seen += 1

            hsi_path2 = folder_path / "HSI" / f"{hsi_idx2:08d}.npy"
            hsi2_data = self.normalize_hsi(
                self.preprocess_hsi(hsi_path2, folder_path, hsi_idx2)).astype(np.float32)
            K1_new = None
            if self.undistort and self.hsi_calib is not None:
                hsi2_data, K1_new = undistort_image_with_K(
                    hsi2_data, self.hsi_calib['camera_matrix'], self.hsi_calib['dist_coeffs'])
            if (self.registration is not None) and (self.target_space == 'rgb') and \
                    self.homography is not None and rgb_data is not None:
                hsi2_data = warp_image(hsi2_data, self.homography,
                                       (rgb_data.shape[1], rgb_data.shape[0]))
            hsi_np_tensor = torch.from_numpy(hsi2_data).permute(2, 0, 1).contiguous()

            K1 = K1_new if (self.undistort and self.hsi_calib is not None) else \
                (self.hsi_calib['camera_matrix'] if self.hsi_calib is not None else None)
            if K1 is not None:
                K1 = np.asarray(K1).astype(np.float32)

            if self.load_rgb:
                rgb_log = self.rgb_frame_logs.get(folder, None)
                rgb_idx2 = int(hsi_idx2)
                if rgb_log is not None and not rgb_log.empty:
                    ids = rgb_log['frame_id'].to_numpy().astype(int)
                    if ids.size > 0:
                        rgb_idx2 = int(ids[np.abs(ids - hsi_idx2).argmin()])
                rgb_np_path = folder_path / "RGB" / f"{rgb_idx2:08d}.npy"
                if rgb_np_path.exists():
                    rgb2 = self.debayer_rgb(np.load(rgb_np_path)).astype(np.float32) / 255.0
                    if self.undistort and self.rgb_calib is not None:
                        rgb2, _ = undistort_image_with_K(
                            rgb2, self.rgb_calib['camera_matrix'], self.rgb_calib['dist_coeffs'])
                    if (self.registration is not None) and (self.target_space == 'hsi') and \
                            self.homography is not None and hsi_data is not None:
                        rgb2 = warp_image(rgb2, self.homography,
                                          (hsi_data.shape[1], hsi_data.shape[0]))
                    rgb_np_tensor = torch.from_numpy(rgb2).permute(2, 0, 1).contiguous()

        if self.photometric_augmentation:
            if hsi_tensor is not None:
                hsi_tensor = apply_photometric_augmentation(hsi_tensor, aug_strength=self.aug_strength)
                if warped_hsi_tensor is not None:
                    warped_hsi_tensor = apply_photometric_augmentation(
                        warped_hsi_tensor, aug_strength=self.aug_strength)
            if rgb_tensor is not None:
                rgb_tensor = apply_photometric_augmentation(rgb_tensor, aug_strength=self.aug_strength)
                if warped_rgb_tensor is not None:
                    warped_rgb_tensor = apply_photometric_augmentation(
                        warped_rgb_tensor, aug_strength=self.aug_strength)
        if self.photometric_augmentation_nonplanar:
            if hsi_np_tensor is not None:
                hsi_np_tensor = apply_photometric_augmentation(
                    hsi_np_tensor, aug_strength=self.aug_strength)
            if rgb_np_tensor is not None:
                rgb_np_tensor = apply_photometric_augmentation(
                    rgb_np_tensor, aug_strength=self.aug_strength)

        if self.load_hsi and hsi_tensor is None:
            raise ValueError(f"Failed to load valid HSI data from {hsi_path}.")
        if self.load_rgb and rgb_tensor is None:
            raise ValueError(f"Failed to load valid RGB data from {rgb_path}.")

        sample = {}
        if self.load_hsi and hsi_tensor is not None:
            sample['hsi'] = hsi_tensor
            sample['hsi_path'] = str(hsi_path)
            if warped_hsi_tensor is not None:
                sample['warped_hsi'] = warped_hsi_tensor
            if hsi_np_tensor is not None:
                sample['hsi_np'] = hsi_np_tensor
                sample['hsi_np_path'] = str(hsi_path2)
            if self.load_poses and hsi_pose is not None:
                sample['hsi_pose'] = hsi_pose
        if self.load_rgb and rgb_tensor is not None:
            sample['rgb'] = rgb_tensor
            sample['rgb_path'] = str(rgb_path)
            if warped_rgb_tensor is not None:
                sample['warped_rgb'] = warped_rgb_tensor
            if rgb_np_tensor is not None:
                sample['rgb_np'] = rgb_np_tensor
            if self.load_poses and rgb_pose is not None:
                sample['rgb_pose'] = rgb_pose

        if H_mat is not None:
            sample['H_mat'] = H_mat
        if K0 is not None:
            sample['K0'] = torch.from_numpy(np.array(K0, copy=False)).float()
        if K1 is not None:
            sample['K1'] = torch.from_numpy(np.array(K1, copy=False)).float()
        if R_01 is not None:
            sample['R_01'] = torch.from_numpy(R_01).float()
        if t_01 is not None:
            sample['t_01'] = torch.from_numpy(t_01).float()
        if overlap_bbox is not None:
            sample['overlap_bbox'] = torch.tensor(overlap_bbox, dtype=torch.float32)

        return sample

    @staticmethod
    def pose_from_log(frame_log: pd.DataFrame, frame_id: int):
        row = frame_log[frame_log['frame_id'] == frame_id]
        if row.empty:
            return None
        return np.array([
            row['link_position_x'].values[0],
            row['link_position_y'].values[0],
            row['link_position_z'].values[0],
            row['link_orientation_x'].values[0],
            row['link_orientation_y'].values[0],
            row['link_orientation_z'].values[0],
            row['link_orientation_w'].values[0],
        ])

    def get_overlap_bbox(self, src_hw, dst_hw):
        """Cached valid-overlap bbox for warping RGB (src) into the HSI frame (dst)."""
        from data_processing.utils.registration_utils import compute_overlap_bbox, compute_overlap_mask
        key = (tuple(src_hw), tuple(dst_hw))
        cache = getattr(self, '_overlap_cache', None)
        if cache is None:
            cache = {}
            self._overlap_cache = cache
        if key not in cache:
            cache[key] = compute_overlap_bbox(self.homography, src_hw, dst_hw)
            self.overlap_mask = compute_overlap_mask(self.homography, src_hw, dst_hw)
        return cache[key]

    def __len__(self) -> int:
        return len(self.data_pairs)

    def load_calibration_data(self):
        """Load white/dark radiometric references and per-folder frame logs/poses."""
        for folder in self.acquisition_folders:
            folder_path = self.data_root / folder

            white_images = sorted(list(folder_path.glob('white_*.npy')))
            dark_images  = sorted(list(folder_path.glob('dark_*.npy')))
            if len(white_images) != 1 or len(dark_images) != 1:
                print(f"Warning: Expected 1 white and 1 dark image in {folder}, "
                      f"found {len(white_images)} white and {len(dark_images)} dark.")
                continue

            white_exposure_time = self.parse_exposure_time(white_images[0].name)
            dark_exposure_time  = self.parse_exposure_time(dark_images[0].name)
            if white_exposure_time != dark_exposure_time:
                print(f"Warning: White and dark images have different exposure times in {folder}.")
                continue

            frame_log_path = folder_path / 'frame_log.csv'
            hsi_pose_path  = folder_path / 'hsi_poses.csv'
            rgb_pose_path  = folder_path / 'rgb_poses.csv'
            if not frame_log_path.exists():
                print(f"Warning: No frame log found in {folder}")
                continue

            self.frame_logs[folder] = pd.read_csv(frame_log_path)
            if hsi_pose_path.exists():
                self.hsi_frame_logs[folder] = pd.read_csv(hsi_pose_path)
            else:
                print(f"Warning: No HSI pose file in {folder}; deriving from frame log via hand-eye transform")
                CameraCalibration().apply_hand_eye_transform(
                    str(frame_log_path), self.hsi_calib['hand_eye_rotation'],
                    self.hsi_calib['hand_eye_translation'], str(hsi_pose_path))
                self.hsi_frame_logs[folder] = pd.read_csv(hsi_pose_path)

            if rgb_pose_path.exists():
                self.rgb_frame_logs[folder] = pd.read_csv(rgb_pose_path)
            elif self.load_rgb:
                print(f"Warning: No RGB pose file in {folder}; deriving from frame log via hand-eye transform")
                CameraCalibration().apply_hand_eye_transform(
                    str(frame_log_path), self.rgb_calib['hand_eye_rotation'],
                    self.rgb_calib['hand_eye_translation'], str(rgb_pose_path))
                self.rgb_frame_logs[folder] = pd.read_csv(rgb_pose_path)

            self.white_dark_images[folder] = {
                'white_image':             np.load(white_images[0]),
                'dark_image':              np.load(dark_images[0]),
                'white_dark_exposure_time': white_exposure_time,
            }

    def load_data_pairs(self):
        """Enumerate matching HSI/RGB frame pairs across all acquisition folders."""
        for folder in self.acquisition_folders:
            folder_path = self.data_root / folder
            if folder not in self.white_dark_images:
                continue
            hsi_dir = folder_path / "HSI"
            rgb_dir = folder_path / "RGB"
            if not hsi_dir.exists() or not rgb_dir.exists():
                continue
            for hsi_file in sorted(f for f in hsi_dir.iterdir() if f.suffix == '.npy'):
                try:
                    hsi_idx = int(hsi_file.stem)
                except ValueError:
                    continue
                if not (rgb_dir / hsi_file.name).exists():
                    continue
                self.data_pairs.append({'folder': folder, 'hsi_idx': hsi_idx, 'rgb_idx': hsi_idx})

    @staticmethod
    def parse_exposure_time(filename: str) -> Optional[int]:
        """Extract exposure time from 'white_{exposure}_{info}.npy' / 'dark_{exposure}_{info}.npy'."""
        parts = filename.split('_')
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return None

    def preprocess_hsi(self, hsi_path: Path, folder_path: Path, frame_id: int) -> np.ndarray:
        """Demosaic the XIMEA mosaic and optionally apply radiometric calibration."""
        hsi_data = np.load(hsi_path)
        original_shape = hsi_data.shape
        try:
            hsi_data, _ = demosaic_aligned_ximea(hsi_data)
            if hsi_data.ndim != 3:
                raise ValueError(f"demosaic returned {hsi_data.ndim}D (shape {hsi_data.shape}) for {hsi_path}")
        except Exception as e:
            raise RuntimeError(
                f"demosaic_aligned_ximea failed for {hsi_path} (shape {original_shape})") from e

        if self.radiometric_calibration:
            calib = self.white_dark_images[folder_path.name]
            frame_log = self.frame_logs[folder_path.name]
            frame_row = frame_log[frame_log['frame_id'] == frame_id]
            if frame_row.empty:
                raise ValueError(f"No frame log entry for frame {frame_id}")
            frame_exposure_time = frame_row['camera_exposure_hsi'].values[0]
            white_image, _ = demosaic_aligned_ximea(calib['white_image'])
            dark_image,  _ = demosaic_aligned_ximea(calib['dark_image'])
            # renormalize=False keeps absolute reflectance; fixed per-band normalization owns scaling.
            hsi_data = radiometric_calibration(
                hsi_data, white_image, dark_image,
                frame_exposure_time, calib['white_dark_exposure_time'], renormalize=False)

        return hsi_data.astype(np.float32)

    def normalize_hsi(self, hsi_data: np.ndarray) -> np.ndarray:
        """Fixed per-band robust normalization (preferred) or per-image min-max fallback."""
        if self._band_stats is not None:
            lo, hi = self._band_stats
            if hsi_data.shape[-1] == lo.shape[0]:
                return normalize_hsi_bands(hsi_data, lo, hi)
            if not self._warned_missing_band_stats:
                print(f"WARNING: HSI has {hsi_data.shape[-1]} bands but band stats have "
                      f"{lo.shape[0]} bands; falling back to per-image min-max.")
                self._warned_missing_band_stats = True
        min_val, max_val = np.min(hsi_data), np.max(hsi_data)
        return (hsi_data - min_val) / (max_val - min_val + 1e-8)

    @staticmethod
    def custom_collate_fn(batch):
        from data_processing.utils.dataset_utils import spectral_collate_fn
        return spectral_collate_fn(batch)

    def relative_pose_from_logs(
        self, frame_log: pd.DataFrame, idx0: int, idx1: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return R_01, t_01 (cam0→cam1) from per-frame world poses."""
        row0 = frame_log[frame_log['frame_id'] == idx0]
        row1 = frame_log[frame_log['frame_id'] == idx1]
        if row0.empty or row1.empty:
            return None, None
        C0 = np.array([row0['link_position_x'].values[0], row0['link_position_y'].values[0],
                        row0['link_position_z'].values[0]], dtype=np.float32)
        q0 = np.array([row0['link_orientation_w'].values[0], row0['link_orientation_x'].values[0],
                        row0['link_orientation_y'].values[0], row0['link_orientation_z'].values[0]],
                       dtype=np.float32)
        C1 = np.array([row1['link_position_x'].values[0], row1['link_position_y'].values[0],
                        row1['link_position_z'].values[0]], dtype=np.float32)
        q1 = np.array([row1['link_orientation_w'].values[0], row1['link_orientation_x'].values[0],
                        row1['link_orientation_y'].values[0], row1['link_orientation_z'].values[0]],
                       dtype=np.float32)
        R_W_C0 = quaternion_to_rotation_matrix(q0).astype(np.float32)
        R_W_C1 = quaternion_to_rotation_matrix(q1).astype(np.float32)
        R_01 = (R_W_C1.T @ R_W_C0).astype(np.float32)
        t_01 = (R_W_C1.T @ (C0 - C1)).astype(np.float32)
        return R_01, t_01

    @staticmethod
    def rot_angle_deg_from_R(R: np.ndarray) -> float:
        tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(tr)))

    @staticmethod
    def score_np_quality(R_01: np.ndarray, t_01: np.ndarray) -> float:
        """Soft quality score in [0, 1] gating relative rotation and translation magnitude."""
        rot_deg = SpectralDataset.rot_angle_deg_from_R(R_01)
        t_norm  = float(np.linalg.norm(t_01))
        rot_ok = 1.0 / (1.0 + np.exp(-(rot_deg - _NP_MIN_ROT_DEG)))
        t_ok   = 1.0 / (1.0 + np.exp(-(t_norm  - _NP_MIN_T_NORM) * 10))
        t_cap  = 1.0 / (1.0 + np.exp((t_norm   - _NP_MAX_T_NORM) * 6))
        score  = (rot_ok * t_ok * t_cap) ** (1.0 / 3.0)
        return float(np.clip(score, 0.0, 1.0))

    def pick_nonplanar_index_smart(
        self, frame_log: pd.DataFrame, base_idx: int, min_off: int, max_off: int
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Pick a temporal partner frame by best quality score; always returns a valid index.

        A curriculum grows the temporal offset window over the first _np_growth_steps
        samples. Candidates are scored by relative rotation + translation magnitude.
        """
        min_id = int(frame_log['frame_id'].min())
        max_id = int(frame_log['frame_id'].max())

        p = max(0.0, min(1.0, float(self._np_seen) / max(float(self._np_growth_steps), 1.0)))
        eff_min_off = max(1, int(round(self._np_start_min_off
                                       + p * max(0, min_off - self._np_start_min_off))))
        eff_max_off = max(eff_min_off, int(round(self._np_start_max_off
                                                  + p * max(0, max_off - self._np_start_max_off))))

        offs  = np.random.randint(low=eff_min_off, high=eff_max_off + 1, size=_NP_TRY_CANDIDATES)
        signs = np.random.choice([-1, 1], size=_NP_TRY_CANDIDATES)
        cands = np.clip(base_idx + offs * signs, min_id, max_id).astype(int)
        cands = np.unique(np.concatenate([cands, np.array([
            np.clip(base_idx + eff_min_off, min_id, max_id),
            np.clip(base_idx - eff_min_off, min_id, max_id),
            np.clip(base_idx + max(2, eff_min_off + 1), min_id, max_id),
            np.clip(base_idx - max(2, eff_min_off + 1), min_id, max_id),
        ], dtype=int)]))

        stride = self.nonplanar_stride
        if stride > 1:
            filtered = np.array([c for c in cands
                                  if abs(int(c) - int(base_idx)) % stride == 0], dtype=int)
            if filtered.size == 0:
                step = min_off + (stride - (min_off % stride)) % stride
                filtered = np.unique(np.array([
                    np.clip(base_idx + step, min_id, max_id),
                    np.clip(base_idx - step, min_id, max_id)], dtype=int))
            cands = filtered

        best_idx, best_R, best_t = None, None, None
        best_score = -1.0
        for cand in cands:
            if cand == base_idx:
                continue
            R_01, t_01 = self.relative_pose_from_logs(frame_log, base_idx, cand)
            if R_01 is None or t_01 is None:
                continue
            score = self.score_np_quality(R_01, t_01)
            if _NP_PREFER_LARGER_BASELINE and p > 0.0:
                rot_deg = self.rot_angle_deg_from_R(R_01)
                t_norm  = float(np.linalg.norm(t_01))
                score += (0.05 * p) * (np.tanh((rot_deg - _NP_MIN_ROT_DEG) / 10.0)
                                        + np.tanh((t_norm - _NP_MIN_T_NORM)))
            if score > best_score:
                best_idx, best_R, best_t = int(cand), R_01, t_01
                best_score = float(score)

        if best_idx is None:
            step = max(1, min_off, stride if stride > 1 else 1)
            best_idx = min(max_id, base_idx + step)
            if best_idx == base_idx:
                best_idx = max(min_id, base_idx - step)
            R_01, t_01 = self.relative_pose_from_logs(frame_log, base_idx, best_idx)
            if R_01 is None or t_01 is None:
                R_01 = np.eye(3, dtype=np.float32)
                t_01 = np.array([1e-6, 0, 0], dtype=np.float32)
            best_R, best_t = R_01, t_01

        return best_idx, best_R, best_t
