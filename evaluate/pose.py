"""Relative-pose estimation from sparse matches (USAC-MAGSAC essential matrix)."""
import cv2
import numpy as np


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    """Recover relative pose (R, t) from matched keypoints and intrinsics.

    Returns (R, t, inlier_mask, E) for the essential-matrix hypothesis with the
    most inliers, or None if pose recovery fails.
    """
    if len(kpts0) < 5:
        return None

    # Normalize keypoints by the respective intrinsics.
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.USAC_MAGSAC)
    if E is None:
        return None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0, _E)
            best_num_inliers = n

    return ret
