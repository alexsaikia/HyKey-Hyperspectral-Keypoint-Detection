import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import yaml

# -----------------------------
# General I/O and filesystem
# -----------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def find_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the BEST checkpoint to load (recursively).

    Lightning saves filenames that encode the monitored metric value, e.g.
    ``...val/nonplanar/metrics/mean=0.7427.ckpt``. We prefer the checkpoint with the
    highest such value (all checkpoints in a single run share the same monitor, and all
    monitors here are mode=max), so this returns each run's BEST epoch rather than the
    most-recently-written file (which would be ``last.ckpt``). Falls back to most-recent
    mtime when no metric value is encoded in any filename.
    """
    import re
    if not checkpoint_dir.exists():
        return None
    ckpts = list(checkpoint_dir.rglob('*.ckpt'))
    if not ckpts:
        return None
    scored = []
    for p in ckpts:
        m = re.search(r'mean=([0-9]+\.?[0-9]*)\.ckpt$', p.name)
        if m:
            scored.append((float(m.group(1)), p))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]


def find_training_yaml(checkpoint_dir: Path) -> Optional[Path]:
    """Heuristic: find a top-level *.yaml in the checkpoint dir; if multiple, pick the largest."""
    yamls = list(checkpoint_dir.glob('*.yaml'))
    if not yamls:
        yamls = list(checkpoint_dir.rglob('*.yaml'))
    if not yamls:
        return None
    yamls.sort(key=lambda p: p.stat().st_size, reverse=True)
    return yamls[0]


def derive_wrapper_kwargs(model_name: str, train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract common wrapper kwargs from a training config file."""
    kwargs: Dict[str, Any] = {}
    candidates = [train_cfg]
    if isinstance(train_cfg.get('model'), dict):
        candidates.append(train_cfg['model'])
    if isinstance(train_cfg.get('module'), dict):
        candidates.append(train_cfg['module'])
    keys_of_interest = ['input_channels']
    if model_name.lower() == 'hykey':
        keys_of_interest += ['c1', 'c2', 'c3', 'dim', 'radius', 'top_k', 'scores_th', 'n_limit',
                             'mask_min_avg', 'mask_max_avg']
    for cand in candidates:
        for k in keys_of_interest:
            if isinstance(cand, dict) and k in cand and cand[k] is not None:
                kwargs[k] = cand[k]
    return kwargs


def is_custom_model(name: str) -> bool:
    n = name.lower()
    return n in ("hykey", "hykey3d", "spn", "alike", "phssspn")


def adapt_images_for_model(model_name: str, img0: torch.Tensor, img1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapt channel layout for different models.
    - For classical detectors (LightGlue, etc.): if channels > 3, convert HSI to pseudo-RGB.
    - For custom models: keep original channels.
    """
    if is_custom_model(model_name):
        return img0, img1
    c0 = img0.shape[0]
    c1 = img1.shape[0]
    if c0 > 3:
        idx0 = torch.tensor([c0 - 1, c0 // 2, 0], device=img0.device)
        img0 = img0.index_select(0, idx0)
    if c1 > 3:
        idx1 = torch.tensor([c1 - 1, c1 // 2, 0], device=img1.device)
        img1 = img1.index_select(0, idx1)
    return img0, img1


# -----------------------------
# Matching utilities
# -----------------------------

def compute_mnn_matches(kpts0, kpts1, desc0, desc1):
    """MNN match two keypoint/descriptor sets; returns matched (M,2) arrays."""
    kpts0 = kpts0.cpu().numpy() if torch.is_tensor(kpts0) else kpts0
    kpts1 = kpts1.cpu().numpy() if torch.is_tensor(kpts1) else kpts1
    desc0 = desc0.cpu().numpy() if torch.is_tensor(desc0) else desc0
    desc1 = desc1.cpu().numpy() if torch.is_tensor(desc1) else desc1

    # Use centralized matcher (no thresholds by default)
    matches_idx, _ = mnn_match_descriptors(desc0, desc1, sim_threshold=None, ratio_threshold=None)
    if matches_idx.size == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    matched_kpts0 = kpts0[matches_idx[:, 0]]
    matched_kpts1 = kpts1[matches_idx[:, 1]]

    return matched_kpts0, matched_kpts1

def mnn_match_descriptors(
    desc0: np.ndarray,
    desc1: np.ndarray,
    sim_threshold: float | None = None,
    ratio_threshold: float | None = None,
):
    """Canonical cosine-similarity MNN matcher with optional thresholds.
    - L2-normalizes descriptors
    - Computes cosine similarity matrix
    - Mutual nearest neighbors by argmax both directions
    - Optional similarity threshold (keep sim >= sim_threshold)
    - Optional Lowe ratio test on similarities (top1/top2 >= ratio_threshold)
    Returns (matches_idx: (M,2), sims: (M,))
    """
    if isinstance(desc0, torch.Tensor):
        desc0 = desc0.detach().cpu().numpy()
    if isinstance(desc1, torch.Tensor):
        desc1 = desc1.detach().cpu().numpy()
    if desc0.size == 0 or desc1.size == 0:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

    # L2-normalize
    desc0 = desc0 / (np.linalg.norm(desc0, axis=1, keepdims=True) + 1e-8)
    desc1 = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)

    sim = desc0 @ desc1.T  # cosine similarity
    nn01 = sim.argmax(axis=1)
    nn10 = sim.argmax(axis=0)
    ids0 = np.arange(desc0.shape[0])
    mutual_mask = (ids0 == nn10[nn01])
    if not np.any(mutual_mask):
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

    i0 = ids0[mutual_mask]
    j0 = nn01[mutual_mask]
    sims0 = sim[i0, j0]

    # Build optional masks ALL relative to the mutual set (length n_mutual) and apply
    # them together exactly once. Previously the ratio mask was computed over the full
    # mutual set but applied after the sim_threshold had already shrunk i0/j0, causing a
    # "boolean index did not match" crash whenever both thresholds were supplied.
    keep = np.ones(i0.shape[0], dtype=bool)

    # Optional similarity threshold
    if sim_threshold is not None:
        keep &= (sims0 >= float(sim_threshold))

    # Optional ratio test on similarities (computed on the mutual set)
    if ratio_threshold is not None:
        eps = 1e-8
        sim_mut = sim[mutual_mask]  # (n_mutual, N1)
        if sim_mut.shape[1] >= 2:
            top2_idx = np.argpartition(sim_mut, -2, axis=1)[:, -2:]
            rows = np.arange(top2_idx.shape[0])
            top_vals = sim_mut[rows[:, None], top2_idx]
            top_vals.sort(axis=1)  # ascending
            top1 = top_vals[:, 1]
            top2 = top_vals[:, 0]
            ratio = top1 / (top2 + eps)
            keep &= (ratio >= float(ratio_threshold))

    i0, j0, sims0 = i0[keep], j0[keep], sims0[keep]
    if i0.size == 0:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

    matches = np.stack([i0, j0], axis=1)
    return matches, sims0

def to_pseudo_rgb(img: torch.Tensor) -> torch.Tensor:
    """Convert CxHxW to 3xHxW pseudo-RGB by selecting [last, mid, first] or replicating."""
    c = img.shape[0]
    if c == 3:
        return img
    if c == 1:
        return img.repeat(3, 1, 1)
    if c >= 3:
        idx = torch.tensor([c - 1, c // 2, 0], device=img.device)
        return img.index_select(0, idx)
    # c == 2: pad with mean channel
    mean = img.mean(dim=0, keepdim=True)
    return torch.cat([img, mean], dim=0)[:3]

def extract_pair_features(wrapper, img0_chw: torch.Tensor, img1_chw: torch.Tensor):
    """Run wrapper.forward(img0,img1) and return numpy kpts/desc and MNN indices."""
    with torch.inference_mode():
        _, _, kpts0, kpts1, desc0, desc1 = wrapper.forward(
            img0_chw.to(wrapper.device), img1_chw.to(wrapper.device)
        )

    def to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    kpts0 = to_np(kpts0)
    kpts1 = to_np(kpts1)
    desc0 = to_np(desc0)
    desc1 = to_np(desc1)

    if desc0.size > 0:
        desc0 = desc0 / (np.linalg.norm(desc0, axis=1, keepdims=True) + 1e-8)
    if desc1.size > 0:
        desc1 = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)

    mkpts0, mkpts1 = compute_mnn_matches(kpts0, kpts1, desc0, desc1)
    return kpts0, kpts1, desc0, desc1, mkpts0, mkpts1
