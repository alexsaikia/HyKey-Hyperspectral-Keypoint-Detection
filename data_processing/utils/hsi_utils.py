import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import skimage as ski
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

WAVELENGTHS_TO_INDEX = {
    570.223306: 0,
    581.09613:  1,
    589.134045: 2,
    596.804142: 3,
    532.395702: 4,
    545.122226: 5,
    550.337997: 6,
    563.19443:  7,
    496.317441: 8,
    505.907749: 9,
    513.84068:  10,
    525.338102: 11,
    458.931626: 12,
    468.174975: 13,
    474.429656: 14,
    484.051574: 15,
}

_SORTED_WAVELENGTHS = sorted(WAVELENGTHS_TO_INDEX.keys())
_IDX_FOR_WV = [WAVELENGTHS_TO_INDEX[wv] for wv in _SORTED_WAVELENGTHS]

_ROW_OFFSETS_FLAT = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
_COL_OFFSETS_FLAT = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

_INTERP_MAP = {
    "nearest":  cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic":  cv2.INTER_CUBIC,
}

_WARP_MATS_CACHE = {}
_FULL_COORD_CACHE = {}
_TORCH_BASE_GRID_CACHE = {}


def demosaic_aligned_ximea(mosaic, h=1088, w=2048, method="bilinear",
                           fast_small=False, prefilter=False,
                           use_torch=True, device: str = "cpu",
                           align_corners: bool = True):
    """Demosaic a XIMEA raw mosaic with sub-pixel band alignment.

    Extracts each of the 16 spectral bands from the 4×4 macropixel mosaic and
    applies a fractional-pixel affine shift per band to cancel its physical
    macropixel offset. Output channels are sorted by wavelength (~459-597 nm).

    Args:
        mosaic:      Raw 2-D array of shape (h, w), h and w divisible by 4.
        method:      Interpolation: 'nearest', 'bilinear' (default), or 'bicubic'.
        fast_small:  If True, align directly on the small (h/4, w/4) grid: faster
                     but slightly less accurate than the full-res path.

    Returns:
        cube:             (h/4, w/4, 16) float32 array, bands sorted by wavelength.
        WAVELENGTHS_TO_INDEX: dict mapping wavelength (nm) → raw band index.
    """
    if method not in _INTERP_MAP:
        raise ValueError(f"Unknown method '{method}'")
    flag = _INTERP_MAP[method]

    new_h, new_w = h // 4, w // 4

    row_base = (np.arange(new_h) * 4)[:, None]
    col_base = (np.arange(new_w) * 4)[None, :]
    row_idx = row_base[:, :, None] + _ROW_OFFSETS_FLAT[None, None, :]
    col_idx = col_base[:, :, None] + _COL_OFFSETS_FLAT[None, None, :]
    small_flat = mosaic[row_idx, col_idx]
    small = small_flat[:, :, _IDX_FOR_WV]

    if fast_small:
        out = np.zeros((new_h, new_w, 16), dtype=np.float32)
        cache_key = (new_h, new_w, method)
        if cache_key not in _WARP_MATS_CACHE:
            mats = []
            for flat_idx in _IDX_FOR_WV:
                roff = float(_ROW_OFFSETS_FLAT[flat_idx]) / 4.0
                coff = float(_COL_OFFSETS_FLAT[flat_idx]) / 4.0
                M = np.array([[1.0, 0.0, -coff], [0.0, 1.0, -roff]], dtype=np.float32)
                mats.append(M)
            _WARP_MATS_CACHE[cache_key] = mats
        else:
            mats = _WARP_MATS_CACHE[cache_key]

        if use_torch and _TORCH_AVAILABLE:
            A = torch.from_numpy(np.stack(mats, axis=0))
            inp = torch.from_numpy(small.transpose(2, 0, 1)[:, None, :, :].astype(np.float32))
            if device != "cpu":
                inp = inp.to(device)
                A = A.to(device)
            grid = F.affine_grid(A, size=(16, 1, new_h, new_w), align_corners=align_corners)
            out_t = F.grid_sample(inp, grid,
                                  mode="bilinear" if flag != cv2.INTER_NEAREST else "nearest",
                                  align_corners=align_corners, padding_mode="reflection")
            out = out_t[:, 0].detach().cpu().numpy().transpose(1, 2, 0)
        else:
            for k, M in enumerate(mats):
                band = small[:, :, k].astype(np.float32, copy=False)
                out[:, :, k] = cv2.warpAffine(
                    band, M, (new_w, new_h), flags=flag,
                    borderMode=cv2.BORDER_REFLECT101)

        if prefilter:
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=0.5, sigmaY=0.5,
                                   borderType=cv2.BORDER_REFLECT101)

        if out.dtype != mosaic.dtype:
            if np.issubdtype(mosaic.dtype, np.integer):
                info = np.iinfo(mosaic.dtype)
                out = np.clip(out, info.min, info.max).astype(mosaic.dtype)
            else:
                out = out.astype(mosaic.dtype)

        return out, WAVELENGTHS_TO_INDEX

    # Accurate path: full-res interpolation then downsample
    if use_torch and _TORCH_AVAILABLE:
        inp = torch.from_numpy(
            small.transpose(2, 0, 1)[:, None, :, :].astype(np.float32))
        if device != "cpu":
            inp = inp.to(device)

        base_key = (h, w, new_h, new_w, align_corners)
        if base_key not in _TORCH_BASE_GRID_CACHE:
            ys = torch.linspace(0, h - 1, h)
            xs = torch.linspace(0, w - 1, w)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            xs_small = xx / 4.0
            ys_small = yy / 4.0
            if align_corners:
                x_norm = (2.0 * xs_small / max(new_w - 1, 1)) - 1.0
                y_norm = (2.0 * ys_small / max(new_h - 1, 1)) - 1.0
            else:
                x_norm = (xs_small + 0.5) / new_w * 2.0 - 1.0
                y_norm = (ys_small + 0.5) / new_h * 2.0 - 1.0
            base_grid = torch.stack([x_norm, y_norm], dim=-1).contiguous().float()
            _TORCH_BASE_GRID_CACHE[base_key] = base_grid
        else:
            base_grid = _TORCH_BASE_GRID_CACHE[base_key]

        base_grid = base_grid.to(inp.device)

        coffs = torch.tensor([float(_COL_OFFSETS_FLAT[idx]) for idx in _IDX_FOR_WV],
                             device=inp.device)
        roffs = torch.tensor([float(_ROW_OFFSETS_FLAT[idx]) for idx in _IDX_FOR_WV],
                             device=inp.device)
        if align_corners:
            dx = 2.0 * (coffs / 4.0) / max(new_w - 1, 1)
            dy = 2.0 * (roffs / 4.0) / max(new_h - 1, 1)
        else:
            dx = 2.0 * ((coffs / 4.0) / new_w)
            dy = 2.0 * ((roffs / 4.0) / new_h)

        grid = base_grid.unsqueeze(0).repeat(16, 1, 1, 1)
        grid[:, :, :, 0] = grid[:, :, :, 0] - dx.view(16, 1, 1)
        grid[:, :, :, 1] = grid[:, :, :, 1] - dy.view(16, 1, 1)

        mode = "bilinear" if flag != cv2.INTER_NEAREST else "nearest"
        full_t = F.grid_sample(inp, grid, mode=mode, align_corners=align_corners,
                               padding_mode="reflection")
        small_t = F.interpolate(full_t, size=(new_h, new_w), mode="bilinear",
                                align_corners=align_corners)
        out = small_t[:, 0].detach().cpu().numpy().transpose(1, 2, 0)

        if out.dtype != mosaic.dtype:
            if np.issubdtype(mosaic.dtype, np.integer):
                info = np.iinfo(mosaic.dtype)
                out = np.clip(out, info.min, info.max).astype(mosaic.dtype)
            else:
                out = out.astype(mosaic.dtype)

        return out, WAVELENGTHS_TO_INDEX

    else:
        # OpenCV fallback
        full_cube = np.zeros((h, w, 16), dtype=np.float32)
        coord_key = (h, w, method)
        if coord_key not in _FULL_COORD_CACHE:
            y_full, x_full = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            map_x_base = x_full.astype(np.float32) / 4.0
            map_y_base = y_full.astype(np.float32) / 4.0
            _FULL_COORD_CACHE[coord_key] = (map_x_base, map_y_base)
        else:
            map_x_base, map_y_base = _FULL_COORD_CACHE[coord_key]

        for k, flat_idx in enumerate(_IDX_FOR_WV):
            roff = float(_ROW_OFFSETS_FLAT[flat_idx])
            coff = float(_COL_OFFSETS_FLAT[flat_idx])
            map_x = map_x_base - (coff / 4.0)
            map_y = map_y_base - (roff / 4.0)
            band = small[:, :, k].astype(np.float32, copy=False)
            full_cube[:, :, k] = cv2.remap(band, map_x, map_y,
                                            interpolation=flag,
                                            borderMode=cv2.BORDER_REFLECT101)

        full_cube_small = cv2.resize(full_cube, (new_w, new_h),
                                     interpolation=cv2.INTER_LINEAR)

        if full_cube_small.dtype != mosaic.dtype:
            if np.issubdtype(mosaic.dtype, np.integer):
                info = np.iinfo(mosaic.dtype)
                full_cube_small = np.clip(full_cube_small, info.min, info.max).astype(mosaic.dtype)
            else:
                full_cube_small = full_cube_small.astype(mosaic.dtype)

        return full_cube_small, WAVELENGTHS_TO_INDEX


def radiometric_calibration(image, white_image, dark_image,
                             frame_exposure_time, white_dark_exposure_time,
                             renormalize=False):
    """Radiometric calibration: white/dark reference subtraction with exposure scaling.

    Args:
        image:                   Raw HSI frame (H, W, C).
        white_image:             White reference captured at white_dark_exposure_time.
        dark_image:              Dark reference captured at white_dark_exposure_time.
        frame_exposure_time:     Exposure of `image` (µs).
        white_dark_exposure_time: Exposure of references (µs).
        renormalize:             If True, stretch output to [0, 1] via per-image min-max.
                                 Disabled by default; downstream fixed per-band normalization
                                 owns scaling; re-stretching destroys absolute reflectance.

    Returns:
        Calibrated reflectance image (float32), clipped to [0, 1].
    """
    exposure_ratio = frame_exposure_time / white_dark_exposure_time
    adjusted_white_image = white_image * exposure_ratio
    adjusted_dark_image  = dark_image  * exposure_ratio

    adjusted_white_image = ski.filters.gaussian(adjusted_white_image, sigma=(2, 2),
                                                truncate=3, channel_axis=-1)
    adjusted_dark_image  = ski.filters.gaussian(adjusted_dark_image,  sigma=(2, 2),
                                                truncate=3, channel_axis=-1)

    # Floor the white reference at 100 counts to prevent division by near-zero in dark sensor regions.
    adjusted_white_image = adjusted_white_image.clip(min=100)

    numerator   = image.astype(np.float32) - adjusted_dark_image.astype(np.float32)
    denominator = adjusted_white_image.astype(np.float32) - adjusted_dark_image.astype(np.float32)
    calibrated_image = numerator / (denominator + 1e-6)
    calibrated_image = np.clip(calibrated_image, 0, 1)

    if renormalize:
        denom = calibrated_image.max() - calibrated_image.min()
        if denom > 0:
            calibrated_image = (calibrated_image - calibrated_image.min()) / denom

    return calibrated_image


# Cache of loaded band-statistics files (keyed by absolute path)
_BAND_STATS_FILE_CACHE = {}


def normalize_hsi_bands(hsi_data, lo, hi):
    """Fixed per-band robust normalization using precomputed dataset-level percentiles.

    Scales each spectral band independently:
        out[..., c] = clip((x[..., c] - lo[c]) / (hi[c] - lo[c]), 0, 1)

    Args:
        hsi_data: (H, W, C) HSI cube.
        lo:       Per-band low percentile, shape (C,).
        hi:       Per-band high percentile, shape (C,).

    Returns:
        Normalized float32 cube of shape (H, W, C).
    """
    hsi_data = np.asarray(hsi_data, dtype=np.float32)
    lo = np.asarray(lo, dtype=np.float32).reshape(1, 1, -1)
    hi = np.asarray(hi, dtype=np.float32).reshape(1, 1, -1)
    denom = hi - lo
    denom = np.where(denom <= 0, np.float32(1e-8), denom)
    out = (hsi_data - lo) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_band_stats(stats_path, key):
    """Load fixed per-band (lo, hi) percentile arrays from band_stats.npz.

    Args:
        stats_path: Path to the ``band_stats.npz`` file.
        key:        Stat set name: ``'spectral_raw'`` or ``'spectral_reflectance'``.

    Returns:
        (lo, hi) arrays of shape (C,), or None if the file or key is missing.
    """
    if stats_path is None:
        return None
    stats_path = str(Path(stats_path))

    cached = _BAND_STATS_FILE_CACHE.get(stats_path)
    if cached is None:
        if not Path(stats_path).exists():
            return None
        try:
            with np.load(stats_path, allow_pickle=True) as data:
                cached = {k: data[k] for k in data.files}
        except Exception as e:
            print(f"WARNING: failed to load band stats from {stats_path}: {e}")
            return None
        _BAND_STATS_FILE_CACHE[stats_path] = cached

    lo_key, hi_key = f"{key}_lo", f"{key}_hi"
    if lo_key not in cached or hi_key not in cached:
        return None
    return (np.asarray(cached[lo_key], dtype=np.float32),
            np.asarray(cached[hi_key], dtype=np.float32))
