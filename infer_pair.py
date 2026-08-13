"""Run HyKey on a pair of hyperspectral frames and save a match visualization.

Two pairing modes:
  planar    - a frame and a synthetic homography warp of it (ground-truth H is known,
              so matches are coloured green/red by reprojection error).
  nonplanar - a frame and a later real frame from the same acquisition (true camera
              motion; no ground-truth correspondence, so matches are drawn neutrally).

Example:
    python infer_pair.py \
        --acquisition ./data/spectral/K4_surg_close_acquisition_2024-08-28_22-48-52 \
        --checkpoint ./checkpoints/hykey \
        --idx 0 --pair planar --output matches.png
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from data_processing.spectral_dataset import SpectralDataset
from matching.evaluation_wrappers import HyKeyWrapper
from matching.utils import to_pseudo_rgb
from models.utils.geometry import warp_points
from evaluate.utils import find_best_checkpoint, find_training_yaml
from evaluate.visualizer import plot_matches


def resolve_checkpoint(path: Path):
    """Accept either a .ckpt file or a directory; return (ckpt_path, yaml_path|None)."""
    if path.is_file():
        return path, find_training_yaml(path.parent)
    ckpt = find_best_checkpoint(path)
    if ckpt is None:
        raise FileNotFoundError(f"No .ckpt found under {path}")
    return ckpt, find_training_yaml(path)


def main():
    ap = argparse.ArgumentParser(description="HyKey keypoint matching demo on a frame pair.")
    ap.add_argument("--acquisition", required=True, help="Path to one acquisition folder.")
    ap.add_argument("--checkpoint", required=True, help="Path to a .ckpt file or a checkpoint dir.")
    ap.add_argument("--model-yaml", default=None, help="Training YAML (auto-discovered if omitted).")
    ap.add_argument("--idx", type=int, default=0, help="Frame index within the acquisition.")
    ap.add_argument("--pair", choices=["planar", "nonplanar"], default="planar")
    ap.add_argument("--calibration", default=None, help="Calibration directory (defaults to <acquisition parent>/calibration).")
    ap.add_argument("--band-stats", default="configs/band_stats.npz")
    ap.add_argument("--threshold", type=float, default=3.0, help="Planar correctness threshold (px).")
    ap.add_argument("--radiometric", action="store_true", help="Reflectance input (needs white/dark refs).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="matches.png")
    args = ap.parse_args()

    acq = Path(args.acquisition).resolve()
    ckpt, yaml = resolve_checkpoint(Path(args.checkpoint).resolve())
    if args.model_yaml:
        yaml = Path(args.model_yaml).resolve()
    device = args.device if torch.cuda.is_available() else "cpu"

    dataset = SpectralDataset(
        data_root=str(acq.parent),
        acquisition_folders=[acq.name],
        load_hsi=True, load_rgb=False, load_poses=(args.pair == "nonplanar"),
        registration=None, calibration_dir=args.calibration,
        target_space="hsi", undistort=False, split="all",
        radiometric_calibration=args.radiometric,
        band_stats_path=args.band_stats,
        warp_augmentation=(args.pair == "planar"),
        photometric_augmentation=False,
        always_include_nonplanar=(args.pair == "nonplanar"),
    )
    sample = dataset[args.idx]

    img0_chw = sample["hsi"]
    if args.pair == "planar":
        img1_chw = sample["warped_hsi"]
        H = sample["H_mat"].cpu().numpy()
    else:
        if "hsi_np" not in sample:
            raise RuntimeError("No nonplanar frame available for this index; try --pair planar.")
        img1_chw = sample["hsi_np"]
        H = None

    wrapper = HyKeyWrapper(model_path=str(ckpt), device=device,
                           model_yaml=str(yaml) if yaml else None)
    mkpts0, mkpts1, kpts0, kpts1, _, _ = wrapper.forward(img0_chw, img1_chw)

    mask, acc = None, None
    if H is not None and len(mkpts0):
        err = np.linalg.norm(warp_points(mkpts0, H) - mkpts1, axis=1)
        mask = err < args.threshold
        acc = float(mask.mean()) if len(mask) else 0.0

    title = (f"HyKey: {len(kpts0)}/{len(kpts1)} keypoints, {len(mkpts0)} matches"
             + (f", MMA@{args.threshold:g}px = {acc:.1%}" if acc is not None else " (real motion)"))
    plot_matches(to_pseudo_rgb(img0_chw), to_pseudo_rgb(img1_chw), mkpts0, mkpts1,
                 mask, args.output, title)
    print(title)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
