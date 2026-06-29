import numpy as np
import torch


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.view(-1)
    M = v.new_zeros(3, 3)
    M[0,1], M[0,2] = -z,  y
    M[1,0], M[1,2] =  z, -x
    M[2,0], M[2,1] = -y,  x
    return M


def essential_from_pose(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return skew_symmetric(t.view(-1)) @ R


def fundamental_from_pose(K0: torch.Tensor, K1: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    E = essential_from_pose(R, t)
    return torch.linalg.inv(K1).T @ E @ torch.linalg.inv(K0)


def geodesic_deg(R_est: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    R_rel = R_est @ R_gt.T
    cos = torch.clamp((torch.trace(R_rel) - 1) / 2, min=-1.0, max=1.0)
    return torch.rad2deg(torch.arccos(cos))


def angle_between_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm() + 1e-12); b = b / (b.norm() + 1e-12)
    cos = torch.clamp(torch.dot(a, b), -1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def triangulate_linear(K0: torch.Tensor, K1: torch.Tensor, R: torch.Tensor, t: torch.Tensor, x0_px: torch.Tensor, x1_px: torch.Tensor):
    if x0_px.numel() == 0 or x1_px.numel() == 0:
        return x0_px.new_zeros(3,0), x0_px.new_zeros(0, dtype=torch.bool)
    N = min(x0_px.shape[0], x1_px.shape[0])
    x0 = x0_px[:N]; x1 = x1_px[:N]
    K0inv = torch.linalg.inv(K0); K1inv = torch.linalg.inv(K1)
    x0n = (K0inv @ torch.cat([x0, torch.ones(N,1, device=x0.device)], 1).T).T
    x1n = (K1inv @ torch.cat([x1, torch.ones(N,1, device=x1.device)], 1).T).T
    P0 = torch.eye(3,4, device=x0.device, dtype=x0.dtype)
    P1 = torch.cat([R, t.view(3,1)], 1)
    Xs = []
    for i in range(N):
        x0c = x0n[i]; x1c = x1n[i]
        A = torch.stack([
            x0c[0]*P0[2]-x0c[2]*P0[0],
            x0c[1]*P0[2]-x0c[2]*P0[1],
            x1c[0]*P1[2]-x1c[2]*P1[0],
            x1c[1]*P1[2]-x1c[2]*P1[1],
        ], dim=0)
        _, _, V = torch.linalg.svd(A)
        Xh = V[-1]; X = Xh[:3] / (Xh[3] + 1e-12)
        Xs.append(X)
    X = torch.stack(Xs, 1)
    z0 = X[2]
    X1 = R @ X + t.view(3,1)
    z1 = X1[2]
    mask = (z0 > 0) & (z1 > 0)
    return X, mask


def warp_points(pts_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a 3×3 homography H to (N, 2) [x, y] points."""
    if len(pts_xy) == 0:
        return pts_xy
    hom = np.concatenate([pts_xy, np.ones((len(pts_xy), 1))], axis=1)
    out = (H @ hom.T).T
    return out[:, :2] / out[:, 2:3]
