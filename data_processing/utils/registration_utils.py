import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional, Union


def load_calibration_data(calib_xml_path: Union[str, Path]) -> Dict:
    """Load camera calibration data from an OpenCV XML file."""
    fs = cv2.FileStorage(str(calib_xml_path), cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('camera_matrix').mat()
    dist_coeff = fs.getNode('dist_coeff').mat()
    reprojection_error = fs.getNode('reprojection_error').real()
    fs.release()
    return {
        'camera_matrix': camera_matrix,
        'dist_coeff': dist_coeff,
        'reprojection_error': reprojection_error,
    }


def compute_homography(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Compute homography between two projection matrices."""
    n = np.array([0, 0, 1])
    d = 1.0
    R1 = P1[:, :3]; t1 = P1[:, 3]
    R2 = P2[:, :3]
    H = R2 @ (R1.T - (t1.reshape(3, 1) @ n.reshape(1, 3)) / d)
    return H


def warp_image(img: np.ndarray, H: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Warp image using homography matrix.

    Args:
        img:         Input image (H, W) or (H, W, C).
        H:           3×3 homography matrix.
        output_size: Desired output size as (width, height).

    Returns:
        Warped image of shape (output_height, output_width, C).
    """
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    warped_channels = []
    for c in range(img.shape[2]):
        warped = cv2.warpPerspective(
            img[..., c], H, output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_channels.append(warped)
    return np.stack(warped_channels, axis=-1)


def compute_overlap_mask(
    H: np.ndarray,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> np.ndarray:
    """Boolean mask in the destination frame where the warped source has valid content.

    When an RGB image is warped into the HSI frame with ``cv2.warpPerspective``
    (``BORDER_CONSTANT=0``), the regions the source does not cover become black bars.
    This returns a boolean mask, in the destination frame, that is True exactly where
    the warped source has valid content. Because the registration homography is fixed
    per dataset, this region is constant across frames.

    Args:
        H:      3×3 homography mapping source pixels → destination pixels.
        src_hw: (height, width) of the source image.
        dst_hw: (height, width) of the destination frame.

    Returns:
        Boolean array of shape ``dst_hw`` that is True inside the valid overlap region.
    """
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    ones = np.ones((src_h, src_w), dtype=np.uint8)
    warped = cv2.warpPerspective(
        ones, np.asarray(H, dtype=np.float64), (dst_w, dst_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def compute_overlap_bbox(
    H: np.ndarray,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Axis-aligned bounding box of the valid overlap region (see ``compute_overlap_mask``).

    Returns ``(x0, y0, x1, y1)`` in destination pixel coordinates where ``x1``/``y1`` are
    exclusive (so the region is ``mask[y0:y1, x0:x1]``). Falls back to the full destination
    frame if the warp produces no valid pixels.
    """
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    mask = compute_overlap_mask(H, src_hw, dst_hw)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, dst_w, dst_h)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


def register_intrinsic(rgb_img, hsi_img, rgb_calib, hsi_calib, target_space):
    """Register images using only intrinsic camera parameters.

    Args:
        rgb_img:      RGB image array.
        hsi_img:      HSI image array.
        rgb_calib:    RGB camera calibration dict (keys: camera_matrix, dist_coeffs, ...).
        hsi_calib:    HSI camera calibration dict.
        target_space: 'rgb' or 'hsi'.

    Returns:
        (registered_hsi, registered_rgb, H_hsi2rgb, H_rgb2hsi)
    """
    rgb_undist = cv2.undistort(rgb_img, rgb_calib['camera_matrix'], rgb_calib['dist_coeffs'])
    hsi_undist = cv2.undistort(hsi_img, hsi_calib['camera_matrix'], hsi_calib['dist_coeffs'])

    K_rgb = rgb_calib['camera_matrix']
    K_hsi = hsi_calib['camera_matrix']

    aspect_hsi = K_hsi[1, 1] / K_hsi[0, 0]
    aspect_rgb = K_rgb[1, 1] / K_rgb[0, 0]

    if aspect_hsi > aspect_rgb:
        scale = aspect_hsi / aspect_rgb
        S = np.array([[1, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    else:
        scale = aspect_rgb / aspect_hsi
        S = np.array([[1, 0, 0], [0, 1/scale, 0], [0, 0, 1]], dtype=np.float64)

    H_hsi2rgb = K_rgb @ S @ np.linalg.inv(K_hsi)
    H_rgb2hsi = K_hsi @ np.linalg.inv(S) @ np.linalg.inv(K_rgb)

    if target_space == 'rgb':
        registered_hsi = warp_image(hsi_undist, H_hsi2rgb, (rgb_img.shape[1], rgb_img.shape[0]))
        registered_rgb = rgb_undist
    else:
        registered_rgb = warp_image(rgb_undist, H_rgb2hsi, (hsi_img.shape[1], hsi_img.shape[0]))
        registered_hsi = hsi_undist

    return registered_hsi, registered_rgb, H_hsi2rgb, H_rgb2hsi
