import torch
import cv2
import numpy as np
import wandb
from models.utils.utils import plot_keypoints, plot_matches_with_accuracy, warp
from evaluate.utils import mnn_match_descriptors
from models.utils.geometry import fundamental_from_pose as _F_from_pose


def vis_rgb_channels(num_channels):
    """Channel indices to render an input as RGB for wandb.Image (logged directly, no cv2 conversion).

    1 channel  -> grayscale (repeat)
    2 channels -> (first, first, last)
    3 channels -> native RGB order (true RGB input; previously reversed to BGR)
    >3 (HSI)   -> false color (high, mid, low wavelength) -> (R, G, B)
    """
    if num_channels == 1:
        return [0, 0, 0]
    if num_channels == 2:
        return [0, 0, 1]
    if num_channels == 3:
        return [0, 1, 2]
    return [-1, num_channels // 2, 0]


def log_image_and_score(logger, batch, pred, suffix, top_k=400, threshold=3.0, use_accuracy_coloring=True,
                       sim_threshold: float = 0.6, ratio_threshold: float = 0.8):
    """
    Log images with keypoints, score maps, and matches to WandB.
    
    Args:
        logger: PyTorch Lightning logger with experiment attribute
        batch: Batch dictionary containing 'image0' and 'image1' tensors
        pred: Prediction dictionary containing:
            - 'scores_map0', 'scores_map1': Score maps
            - 'kpts0', 'kpts1': Keypoints lists
            - 'desc0', 'desc1': Descriptors lists
        suffix: String prefix for logging keys
        top_k: Maximum number of keypoints to visualize
        threshold: Threshold in pixels for determining correct matches (default: 1.0)
        use_accuracy_coloring: Whether to color matches based on accuracy (default: True)
    """
    if logger is None or not hasattr(logger, 'experiment') or logger.experiment is None:
        return
    
    try:
        b, c, h, w = batch['image0'].shape

        # Determine a consistent visualization index: prefer first nonplanar item when present
        def find_nonplanar_index(batch):
            if 'pair_type' in batch:
                pt = batch['pair_type']
                if isinstance(pt, (list, tuple)):
                    for i, t in enumerate(pt):
                        if t == 'nonplanar':
                            return i
                elif isinstance(pt, str) and pt == 'nonplanar':
                    return 0
            return None
        idx_np = find_nonplanar_index(batch)
        vis_idx = 0 if idx_np is None else idx_np

        # Helper to fetch per-item tensors robustly (handles NP-subset tensors via orig_idx)
        def get_item_vis(t):
            if t is None:
                return None
            orig_idx = batch.get('orig_idx', None)
            def map_idx(n0):
                if n0 is None or n0 <= 0:
                    return vis_idx
                if n0 > vis_idx:
                    return vis_idx
                if torch.is_tensor(orig_idx) and orig_idx.numel() > vis_idx:
                    j = int(orig_idx[vis_idx].item())
                    if j < n0:
                        return j
                return min(vis_idx, max(0, n0 - 1))
            if isinstance(t, list):
                j = map_idx(len(t))
                return t[j]
            if torch.is_tensor(t):
                if t.dim() >= 3:
                    j = map_idx(t.shape[0])
                    return t[j]
                if t.dim() == 2:
                    j = map_idx(t.shape[0])
                    return t[j]
                return t
            if isinstance(t, np.ndarray):
                if t.ndim >= 3:
                    j = map_idx(t.shape[0])
                    return t[j]
                if t.ndim == 2:
                    j = map_idx(t.shape[0])
                    return t[j]
                return t
            return t

        # Handle different input channel formats
        img0 = batch['image0'][vis_idx]
        img1 = batch['image1'][vis_idx]
        
        # Select channels that render as RGB (true RGB kept native; HSI -> false color)
        num_channels = img0.shape[0]
        channels = vis_rgb_channels(num_channels)
        
        image0 = (img0[channels,:,:] * 255).to(torch.uint8).cpu().permute(1, 2, 0)
        image1 = (img1[channels,:,:] * 255).to(torch.uint8).cpu().permute(1, 2, 0)
        
        # Handle score maps - ensure they are properly shaped
        score_map0 = pred['scores_map0'][vis_idx].detach()
        score_map1 = pred['scores_map1'][vis_idx].detach()
        
        # Handle both single-channel and multi-channel score maps
        if score_map0.dim() == 3 and score_map0.shape[0] == 1:
            score_map0 = score_map0.squeeze(0)
        if score_map1.dim() == 3 and score_map1.shape[0] == 1:
            score_map1 = score_map1.squeeze(0)
        
        scores0 = (score_map0 * 255).to(torch.uint8).cpu().numpy()
        scores1 = (score_map1 * 255).to(torch.uint8).cpu().numpy()
        
        # Check if we have keypoints for this index
        if 0 < len(pred['kpts0']) and 0 < len(pred['kpts1']):
            kpts0, kpts1 = pred['kpts0'][vis_idx][:top_k].detach(), pred['kpts1'][vis_idx][:top_k].detach()
        else:
            # Create empty tensors if no keypoints
            device = image0.device if hasattr(image0, 'device') else torch.device('cpu')
            kpts0 = torch.empty((0, 2), device=device)
            kpts1 = torch.empty((0, 2), device=device)

        # Create a dictionary to hold all images
        log_dict = {}

        # =================== score map (concatenated)
        s0 = cv2.applyColorMap(scores0, cv2.COLORMAP_MAGMA)
        s0 = cv2.cvtColor(s0, cv2.COLOR_BGR2RGB)
        s1 = cv2.applyColorMap(scores1, cv2.COLORMAP_MAGMA)
        s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2RGB)
        try:
            s_concat = cv2.hconcat([s0, s1])
            log_dict[f'{suffix}score/{0}'] = wandb.Image(s_concat)
        except Exception:
            pass

        # =================== image with keypoints (ensure pixel coordinates)
        def to_pixel(k):
            if k.numel() == 0:
                return k
            if k.min() >= -1 and k.max() <= 1:
                return (k + 1) / 2 * torch.tensor([[w-1, h-1]], device=k.device)
            return k
        kpts0_px = to_pixel(kpts0)
        kpts1_px = to_pixel(kpts1)
        image0_kpts = plot_keypoints(image0, kpts0_px[:, [1, 0]], radius=1)
        image1_kpts = plot_keypoints(image1, kpts1_px[:, [1, 0]], radius=1)
        
        try:
            concat_img = cv2.hconcat([image0_kpts, image1_kpts])
            log_dict[f'{suffix}image/{0}'] = wandb.Image(concat_img)
        except Exception:
            pass

        # =================== matches
        if 0 < len(pred['desc0']) and 0 < len(pred['desc1']):
            desc0, desc1 = pred['desc0'][vis_idx][:top_k].detach(), pred['desc1'][vis_idx][:top_k].detach()
            if len(desc0) > 0 and len(desc1) > 0:
                try:
                    # Centralized MNN matching (utils)
                    mi, _ = mnn_match_descriptors(
                        desc0.detach().cpu().numpy(),
                        desc1.detach().cpu().numpy(),
                        sim_threshold=float(sim_threshold),
                        ratio_threshold=float(ratio_threshold)
                    )
                    if mi.size == 0:
                        mkpts0 = kpts0[:0]
                        mkpts1 = kpts1[:0]
                    else:
                        sel_i = torch.as_tensor(mi[:, 0], device=desc0.device, dtype=torch.long)
                        sel_j = torch.as_tensor(mi[:, 1], device=desc1.device, dtype=torch.long)
                        mkpts0, mkpts1 = kpts0[sel_i], kpts1[sel_j]
                    if len(mkpts0) > 0 and len(mkpts1) > 0:
                        # Convert to pixel coordinates if they're normalized (-1 to 1)
                        if mkpts0.min() >= -1 and mkpts0.max() <= 1:
                            mkpts0 = (mkpts0 + 1) / 2 * torch.tensor([[w-1, h-1]], device=mkpts0.device)
                            mkpts1 = (mkpts1 + 1) / 2 * torch.tensor([[w-1, h-1]], device=mkpts1.device)
                        
                        # Swap coordinates for plotting (y, x) -> (x, y)
                        mkpts0_plot = mkpts0[:, [1, 0]]
                        mkpts1_plot = mkpts1[:, [1, 0]]
                        
                        # Compute accuracy for each match using ground truth homography
                        accuracy_mask = torch.zeros(len(mkpts0), dtype=torch.bool, device=mkpts0.device)
                        mkpts0_warped = None
                        mkpts0_warped_full = None
                        
                        # Check if we have homography information in the batch
                        homography = None
                        if 'warp01_params' in batch and len(batch['warp01_params']) > 0:
                            warp_params = batch['warp01_params'][0]
                            if 'homography_matrix' in warp_params:
                                homography = warp_params['homography_matrix']
                        elif 'homography' in batch:
                            homography = batch['homography'][0] if isinstance(batch['homography'], list) else batch['homography']
                        elif 'H' in batch:
                            homography = batch['H'][0] if isinstance(batch['H'], list) else batch['H']
                        elif 'H_mat' in batch:
                            homography = batch['H_mat'][0] if isinstance(batch['H_mat'], list) else batch['H_mat']
                            if homography.dim() == 3:
                                homography = homography[0]
                        
                        if homography is not None:
                            try:
                                # Create warp parameters
                                warp01_params = {
                                    'mode': 'homo',
                                    'homography_matrix': homography,
                                    'width': w,
                                    'height': h
                                }
                                # Warp keypoints from image0 to image1
                                mkpts0_warped, mkpts01, ids0, _ = warp(mkpts0, warp01_params)
                                if len(ids0) > 0:
                                    # Get corresponding keypoints from image1
                                    mkpts1_valid = mkpts1[ids0]
                                    # Compute reprojection error
                                    dist = torch.sqrt(((mkpts01 - mkpts1_valid) ** 2).sum(axis=1))
                                    # Create accuracy mask for valid matches
                                    valid_accuracy = dist < threshold
                                    # Update accuracy mask for valid matches
                                    accuracy_mask[ids0] = valid_accuracy
                                    # Prepare warped overlay
                                    mkpts0_warped_full = torch.full((len(mkpts0), 2), float('nan'), device=mkpts0.device)
                                    mkpts0_warped_full[ids0] = mkpts01[:, [1, 0]]
                            except Exception as e:
                                # If computation fails, fall back to no warped overlay
                                mkpts0_warped_full = None
                        else:
                            # If no homography: for nonplanar, compute Sampson distance mask using K, R, t if available
                            mkpts0_warped_full = None
                            try:
                                K0 = get_item_vis(batch.get('K0', None))
                                K1 = get_item_vis(batch.get('K1', None))
                                R  = get_item_vis(batch.get('R_01', None))
                                t  = get_item_vis(batch.get('t_01', None))
                                if K0 is not None and K1 is not None and R is not None and t is not None and mkpts0.numel() > 0:
                                    # Ensure tensors on correct device/dtype
                                    dev = mkpts0.device
                                    dt = torch.float32
                                    if not torch.is_tensor(K0): K0 = torch.as_tensor(K0, device=dev, dtype=dt)
                                    if not torch.is_tensor(K1): K1 = torch.as_tensor(K1, device=dev, dtype=dt)
                                    if not torch.is_tensor(R):  R  = torch.as_tensor(R,  device=dev, dtype=dt)
                                    if not torch.is_tensor(t):  t  = torch.as_tensor(t,  device=dev, dtype=dt).view(-1)
                                # Fundamental via our geometry helper; Sampson distance squared in px^2
                                F = _F_from_pose(K0, K1, R, t)
                                # manual Sampson squared
                                x0h = torch.cat([mkpts0, torch.ones(mkpts0.shape[0],1, device=dev, dtype=dt)], 1)
                                x1h = torch.cat([mkpts1, torch.ones(mkpts1.shape[0],1, device=dev, dtype=dt)], 1)
                                Ex0  = (F @ x0h.t()).t()
                                Etx1 = (F.t() @ x1h.t()).t()
                                num  = (x1h * Ex0).sum(1).abs()
                                den  = Ex0[:,0].pow(2) + Ex0[:,1].pow(2) + Etx1[:,0].pow(2) + Etx1[:,1].pow(2) + 1e-12
                                d2 = (num / torch.sqrt(den + 1e-12))**2
                                if d2.numel() == 0:
                                    # No valid distances, skip epipolar accuracy coloring
                                    pass
                                valid = torch.isfinite(d2) & (d2 >= 0)
                                acc = torch.zeros_like(d2, dtype=torch.bool)
                                acc[valid] = torch.sqrt(d2[valid]) < 5.0
                                accuracy_mask = acc
                            except Exception:
                                pass
                        
                        # Create match visualization with accuracy colors
                        match_image = plot_matches_with_accuracy(
                            image0, image1, mkpts0_plot, mkpts1_plot, mkpts0_warped_full, accuracy_mask,
                            correct_color=(0, 255, 0),  # Green for correct matches
                            incorrect_color=(255, 0, 0)  # Red for incorrect matches
                        )
                        log_dict[f'{suffix}matches/{0}'] = wandb.Image(match_image)
                except Exception as e:
                    print(f"Warning: Could not create matches visualization: {e}")
                    # Skip matches visualization if it fails
        
        # =================== (nonplanar) epipolar lines for two matches
        try:
            # try to get intrinsics and relative pose to build E/F (nonplanar samples only)
            def get_per_item(t, idx):
                if t is None:
                    return None
                # Map subset index to original batch index when available
                orig_idx = batch.get('orig_idx', None)
                def map_idx(candidate_shape0):
                    if candidate_shape0 is None or candidate_shape0 <= 0:
                        return idx
                    if candidate_shape0 > idx:
                        return idx
                    if torch.is_tensor(orig_idx) and orig_idx.numel() > idx:
                        j = int(orig_idx[idx].item())
                        if j < candidate_shape0:
                            return j
                    return min(idx, candidate_shape0 - 1)
                # list
                if isinstance(t, list):
                    j = map_idx(len(t))
                    return t[j]
                # torch tensor
                if torch.is_tensor(t):
                    if t.dim() >= 3:
                        j = map_idx(t.shape[0])
                        return t[j]
                    if t.dim() == 2:
                        j = map_idx(t.shape[0])
                        return t[j]
                    return t
                # numpy array
                if isinstance(t, np.ndarray):
                    if t.ndim >= 3:
                        j = map_idx(t.shape[0])
                        return t[j]
                    if t.ndim == 2:
                        j = map_idx(t.shape[0])
                        return t[j]
                    return t
                return t

            # pick an index that is explicitly nonplanar (use first entry since this is an NP-only view often)
            idx_np = None
            if 'pair_type' in batch:
                if isinstance(batch['pair_type'], (list, tuple)):
                    for i, pt in enumerate(batch['pair_type']):
                        if pt == 'nonplanar':
                            idx_np = i
                            break
                elif isinstance(batch['pair_type'], str) and batch['pair_type'] == 'nonplanar':
                    idx_np = 0

            if idx_np is None:
                # no nonplanar item in this batch → skip epi viz silently
                raise RuntimeError('no_nonplanar')

            K0 = get_per_item(batch.get('K0', None), idx_np)
            K1 = get_per_item(batch.get('K1', None), idx_np)
            R  = get_per_item(batch.get('R_01', None), idx_np)
            t  = get_per_item(batch.get('t_01', None), idx_np)

            def skew(v):
                return np.array([[0.0, -v[2],  v[1]],
                                 [v[2],  0.0, -v[0]],
                                 [-v[1], v[0], 0.0]], dtype=np.float32)

            F = None
            if K0 is not None and K1 is not None and R is not None and t is not None:
                # ensure cpu numpy
                K0n = K0.detach().cpu().numpy().astype(np.float32)
                K1n = K1.detach().cpu().numpy().astype(np.float32)
                Rn  = R.detach().cpu().numpy().astype(np.float32)
                tn  = t.detach().cpu().numpy().astype(np.float32).reshape(3)
                E   = skew(tn) @ Rn
                try:
                    K0inv = np.linalg.inv(K0n)
                    K1inv = np.linalg.inv(K1n)
                    F = K1inv.T @ E @ K0inv
                except np.linalg.LinAlgError:
                    F = None

            def line_points_from_l(l, w, h):
                # l: [a,b,c], ax+by+c=0
                a, b, c = l
                pts = []
                # intersections with borders
                # y = 0 -> ax + c = 0
                if abs(a) > 1e-8:
                    x0 = -c / a
                    if 0 <= x0 < w:
                        pts.append((int(round(x0)), 0))
                # y = h-1
                if abs(a) > 1e-8:
                    x1 = -(c + b * (h - 1)) / a
                    if 0 <= x1 < w:
                        pts.append((int(round(x1)), h - 1))
                # x = 0 -> by + c = 0
                if abs(b) > 1e-8:
                    y0 = -c / b
                    if 0 <= y0 < h:
                        pts.append((0, int(round(y0))))
                # x = w-1
                if abs(b) > 1e-8:
                    y1 = -(c + a * (w - 1)) / b
                    if 0 <= y1 < h:
                        pts.append((w - 1, int(round(y1))))
                # pick two farthest
                if len(pts) < 2:
                    return None
                # unique points
                pts_unique = []
                for p in pts:
                    if p not in pts_unique:
                        pts_unique.append(p)
                if len(pts_unique) < 2:
                    return None
                # simple heuristic: take first two distinct
                return pts_unique[0], pts_unique[1]

            if F is not None and 0 < len(pred['kpts0']) and 0 < len(pred['kpts1']):
                # Recompute matches using centralized MNN
                desc0, desc1 = pred['desc0'][idx_np][:top_k].detach(), pred['desc1'][idx_np][:top_k].detach()
                kpts0, kpts1 = pred['kpts0'][idx_np][:top_k].detach(), pred['kpts1'][idx_np][:top_k].detach()
                if kpts0.numel() > 0 and kpts1.numel() > 0 and desc0.numel() > 0 and desc1.numel() > 0:
                    mi, _ = mnn_match_descriptors(
                        desc0.detach().cpu().numpy(),
                        desc1.detach().cpu().numpy(),
                        sim_threshold=float(sim_threshold),
                        ratio_threshold=float(ratio_threshold)
                    )
                    if mi.size == 0:
                        raise RuntimeError('no_epi_matches')
                    sel_i = torch.as_tensor(mi[:, 0], device=desc0.device, dtype=torch.long)
                    sel_j = torch.as_tensor(mi[:, 1], device=desc1.device, dtype=torch.long)
                    mk0 = kpts0[sel_i]
                    mk1 = kpts1[sel_j]
                    # Skip if no matches selected
                    if mk0.numel() == 0 or mk1.numel() == 0:
                        # No matches to draw epipolar lines; skip epi viz
                        raise RuntimeError('no_epi_matches')
                    if mk0.min() >= -1 and mk0.max() <= 1:
                        mk0 = (mk0 + 1) / 2 * torch.tensor([[w-1, h-1]], device=mk0.device)
                    if mk1.min() >= -1 and mk1.max() <= 1:
                        mk1 = (mk1 + 1) / 2 * torch.tensor([[w-1, h-1]], device=mk1.device)

                    # Main fundamental matrix only
                    F_main = F

                    # Apply masking consistent with EpipolarSampsonLoss and draw up to 5 color-coded matches
                    if mk0.shape[0] > 0 and mk1.shape[0] > 0:
                        # Use the nonplanar item's images for drawing
                        img0_np = batch['image0'][idx_np]
                        img1_np = batch['image1'][idx_np]
                        num_channels_np = img0_np.shape[0]
                        if num_channels_np == 1:
                            ch_np = [0, 0, 0]
                        elif num_channels_np == 2:
                            ch_np = [0, 0, 1]
                        elif num_channels_np == 3:
                            # True RGB: this path applies BGR2RGB before logging,
                            # so select as BGR ([2,1,0]) to end up correct RGB.
                            ch_np = [2, 1, 0]
                        else:
                            mid_np = num_channels_np // 2
                            ch_np = [0, mid_np, -1]
                        img0_epi = (img0_np[ch_np,:,:] * 255).to(torch.uint8).cpu().permute(1, 2, 0).contiguous().numpy().copy()
                        img1_epi = (img1_np[ch_np,:,:] * 255).to(torch.uint8).cpu().permute(1, 2, 0).contiguous().numpy().copy()
                        h, w = img0_epi.shape[:2]
                        # Masking params
                        border_px = 10
                        # Build keep list by checking epipolar line intersection and border margin in both images
                        kept_indices = []
                        for k in range(min(mk0.shape[0], mk1.shape[0])):
                            p0 = mk0[k].detach().cpu().numpy().astype(np.float32)
                            p1 = mk1[k].detach().cpu().numpy().astype(np.float32)
                            # Border checks (both images)
                            inside0 = (p0[0] >= border_px) and (p0[0] <= (w - 1 - border_px)) and (p0[1] >= border_px) and (p0[1] <= (h - 1 - border_px))
                            inside1 = (p1[0] >= border_px) and (p1[0] <= (w - 1 - border_px)) and (p1[1] >= border_px) and (p1[1] <= (h - 1 - border_px))
                            if not (inside0 and inside1):
                                continue
                            # Epipolar lines
                            x0h = np.array([p0[0], p0[1], 1.0], dtype=np.float32)
                            x1h = np.array([p1[0], p1[1], 1.0], dtype=np.float32)
                            l1 = (F_main @ x0h.reshape(3, 1)).reshape(3)   # in image1
                            l0 = (F_main.T @ x1h.reshape(3, 1)).reshape(3) # in image0
                            a1, b1, c1 = float(l1[0]), float(l1[1]), float(l1[2])
                            a0, b0, c0 = float(l0[0]), float(l0[1]), float(l0[2])
                            # Intersection with image bounds via corner sign test
                            vals1 = [c1,
                                     a1 * (w - 1) + c1,
                                     b1 * (h - 1) + c1,
                                     a1 * (w - 1) + b1 * (h - 1) + c1]
                            intersects1 = (min(vals1) <= 0.0) and (max(vals1) >= -0.0)
                            vals0 = [c0,
                                     a0 * (w - 1) + c0,
                                     b0 * (h - 1) + c0,
                                     a0 * (w - 1) + b0 * (h - 1) + c0]
                            intersects0 = (min(vals0) <= 0.0) and (max(vals0) >= -0.0)
                            if intersects0 and intersects1:
                                kept_indices.append(k)

                        if len(kept_indices) == 0:
                            # Nothing valid to draw
                            pass
                        else:
                            # Color palette (BGR) for up to 5 matches
                            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
                            m = min(5, len(kept_indices))
                            for t in range(m):
                                k = kept_indices[t]
                                color = colors[t % len(colors)]
                                p0 = mk0[k].detach().cpu().numpy().astype(np.float32)
                                p1 = mk1[k].detach().cpu().numpy().astype(np.float32)
                                # draw p0 in img0 and p1 in img1 with matching colors
                                cv2.circle(img0_epi, (int(round(p0[0])), int(round(p0[1]))), 4, color, thickness=-1)
                                cv2.circle(img1_epi, (int(round(p1[0])), int(round(p1[1]))), 4, color, thickness=-1)
                            # principal points (red)
                            cx0, cy0 = K0n[0, 2], K0n[1, 2]
                            cx1, cy1 = K1n[0, 2], K1n[1, 2]
                            cv2.circle(img0_epi, (int(round(cx0)), int(round(cy0))), 4, (0, 0, 255), thickness=-1)
                            cv2.circle(img1_epi, (int(round(cx1)), int(round(cy1))), 4, (0, 0, 255), thickness=-1)
                            # Draw epipolar lines and annotate distances with matching color
                            for t in range(m):
                                k = kept_indices[t]
                                color = colors[t % len(colors)]
                                p0 = mk0[k].detach().cpu().numpy().astype(np.float32)
                                p1 = mk1[k].detach().cpu().numpy().astype(np.float32)
                                x0h = np.array([p0[0], p0[1], 1.0], dtype=np.float32)
                                l1_main = (F_main @ x0h.reshape(3, 1)).reshape(3)
                                seg = line_points_from_l(l1_main, w, h)
                                if seg is not None:
                                    cv2.line(img1_epi, seg[0], seg[1], color, thickness=2)
                                try:
                                    l0_main = (F_main.T @ np.array([p1[0],p1[1],1.0],dtype=np.float32).reshape(3,1)).reshape(3)
                                    seg0 = line_points_from_l(l0_main, w, h)
                                    if seg0 is not None:
                                        cv2.line(img0_epi, seg0[0], seg0[1], color, thickness=2)
                                except Exception:
                                    pass
                                a, b, c = float(l1_main[0]), float(l1_main[1]), float(l1_main[2])
                                denom = max((a*a + b*b) ** 0.5, 1e-8)
                                dist1 = abs(a * float(p1[0]) + b * float(p1[1]) + c) / denom
                                cv2.putText(img1_epi, f"{dist1:.1f}px", (int(round(p1[0]))+5, int(round(p1[1]))-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

                        # log
                        concat = cv2.hconcat([cv2.cvtColor(img0_epi, cv2.COLOR_BGR2RGB), cv2.cvtColor(img1_epi, cv2.COLOR_BGR2RGB)])
                        log_dict[f'{suffix}epi/{idx_np}_img'] = wandb.Image(concat)
                        # no extra scalar logs
        except Exception as e:
            # only print real errors; skip the common no-nonplanar case quietly
            if str(e) != 'no_nonplanar':
                print(f"Warning: Epipolar viz failed: {e}")

        # If nothing was added (e.g., due to upstream failures), log a minimal fallback concat
        if len(log_dict) == 0:
            try:
                img0_np = image0.cpu().numpy() if torch.is_tensor(image0) else image0
                img1_np = image1.cpu().numpy() if torch.is_tensor(image1) else image1
                fallback = cv2.hconcat([cv2.cvtColor(img0_np, cv2.COLOR_RGB2BGR), cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)])
                log_dict[f'{suffix}image_raw/{0}'] = wandb.Image(fallback[:, :, ::-1])
            except Exception as e:
                print(f"Warning: Could not build fallback image log: {e}")
        # Log all images at once with a single call
        try:
            logger.experiment.log(log_dict)
        except Exception as e:
            print(f"Warning: WandB log failed: {e}")
    
    except Exception as e:
        print(f"Warning: Image logging failed: {e}")
        return 


def log_triplet_viz(logger, batch, pred, viz_pack, suffix, top_k=200):
    if logger is None or not hasattr(logger, 'experiment') or logger.experiment is None:
        return
    try:
        correspondences = viz_pack.get('correspondences', None)
        extra_pred = viz_pack.get('extra_pred', None)
        if correspondences is None or len(correspondences) == 0:
            return
        b, c, h, w = batch['image0'].shape
        log_dict = {}

        # Nonplanar triplet: show view-0 (source), view-1 (anchors), view-2 (candidates + reprojection)
        for idx, ci in enumerate(correspondences):
            # Only log the first triplet images
            if idx > 0:
                continue
            if ci.get('pair_type') == 'nonplanar' and ci.get('np3info', None) is not None:
                info = ci['np3info']
                cand2_idx = info.get('cand2_idx', None)
                if cand2_idx is None or cand2_idx.numel() == 0:
                    continue
                # Prepare base images for view-0, view-1 and view-2
                img0_t = (batch['image0'][idx].detach().cpu() * 255).to(torch.uint8)
                img1_t = (batch['image1'][idx].detach().cpu() * 255).to(torch.uint8)
                # view-2 actual image if present (hsi_np2/rgb_np2)
                img2_t = None
                if 'hsi_np2' in batch and batch['hsi_np2'].dim() == 4 and batch['hsi_np2'].shape[0] > idx:
                    img2_t = (batch['hsi_np2'][idx].detach().cpu() * 255).to(torch.uint8)
                elif 'rgb_np2' in batch and batch['rgb_np2'].dim() == 4 and batch['rgb_np2'].shape[0] > idx:
                    img2_t = (batch['rgb_np2'][idx].detach().cpu() * 255).to(torch.uint8)
                # choose RGB-like slice
                def to_rgb(x):
                    ch = vis_rgb_channels(x.shape[0])
                    return x[ch,:,:].permute(1,2,0).numpy()
                img0 = to_rgb(img0_t)
                img1 = to_rgb(img1_t)
                img2 = to_rgb(img2_t) if img2_t is not None else img1.copy()

                # Anchors in view-1 (p1_px), matched in view-0 (p0_px), and candidate keypoints in view-2 (kpts2_px)
                p1_px = ci.get('p1_px', None)
                p0_px = ci.get('p0_px', None)
                k2px = ci.get('kpts2_px', None)
                if p1_px is None or k2px is None or p1_px.numel() == 0:
                    continue
                # Draw anchors
                draw0 = img0.copy()
                if p0_px is not None and p0_px.numel() > 0:
                    for p in p0_px.detach().cpu().numpy().astype(int):
                        cv2.circle(draw0, (int(p[0]), int(p[1])), 3, (0,255,255), -1)  # cyan: matched points in view-0
                draw1 = img1.copy()
                for p in p1_px.detach().cpu().numpy().astype(int):
                    cv2.circle(draw1, (int(p[0]), int(p[1])), 3, (0,255,0), -1)  # green: anchors in view-1
                # Draw top-k candidates for each anchor in view-2
                draw2 = img2.copy()
                M, C = cand2_idx.shape
                cand_flat = k2px[cand2_idx.view(-1)].view(M, C, 2).detach().cpu().numpy()
                for m in range(min(M, 50)):
                    # positive in column 0 in our mining
                    for cidx in range(min(C, 10)):
                        q = cand_flat[m, cidx]
                        color = (0,255,0) if cidx == 0 else (255,0,0)
                        cv2.circle(draw2, (int(q[0]), int(q[1])), 2, color, -1)

                # If triangulation was used, overlay predicted reprojections
                tri = ci.get('tri', None)
                if tri is not None and 'u2_pred' in tri and tri['u2_pred'].numel() > 0:
                    for q in tri['u2_pred'].detach().cpu().numpy().astype(int):
                        cv2.circle(draw2, (int(q[0]), int(q[1])), 3, (0,255,255), 1)

                # If epipolar cycle used, draw epipolar lines of view-2 via view-0 and view-1
                if ci.get('epi_cycle', False):
                    try:
                        # Rebuild F02/F12 as in loss for preview
                        K0 = ci.get('K0'); K1 = ci.get('K1'); K2 = ci.get('K2')
                        R01 = ci.get('R01'); t01 = ci.get('t01')
                        R02 = ci.get('R02'); t02 = ci.get('t02')
                        if all(x is not None for x in [K0,K1,K2,R01,t01,R02,t02]) and p1_px.numel()>0:
                            # torchify
                            dev = p1_px.device
                            dt = p1_px.dtype
                            def to_t(x):
                                return x if torch.is_tensor(x) else torch.as_tensor(x, device=dev, dtype=dt)
                            K0 = to_t(K0); K1 = to_t(K1); K2 = to_t(K2)
                            R01 = to_t(R01); t01 = to_t(t01)
                            R02 = to_t(R02); t02 = to_t(t02)
                            def E_from_Rt(R,t):
                                tx,ty,tz = t.view(-1)
                                T = torch.tensor([[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]], device=dev, dtype=dt)
                                return T @ R
                            E02 = E_from_Rt(R02, t02)
                            R12 = R02 @ R01.t()
                            t12 = (t02.view(3,1) - R12 @ t01.view(3,1)).view(3)
                            E12 = E_from_Rt(R12, t12)
                            K2i = torch.inverse(K2); K0i = torch.inverse(K0); K1i = torch.inverse(K1)
                            F02 = K2i.t() @ E02 @ K0i
                            F12 = K2i.t() @ E12 @ K1i
                            ones = torch.ones((p1_px.shape[0],1), device=dev, dtype=dt)
                            # pick subset for clarity
                            sel = torch.linspace(0, p1_px.shape[0]-1, steps=min(20,p1_px.shape[0]), device=dev).long()
                            x0h = torch.cat([ci['p0_px'][sel], ones[:len(sel)]], dim=1)
                            x1h = torch.cat([ci['p1_px'][sel], ones[:len(sel)]], dim=1)
                            l2_via0 = (F02 @ x0h.t()).t().detach().cpu().numpy()
                            l2_via1 = (F12 @ x1h.t()).t().detach().cpu().numpy()
                            h2, w2 = draw2.shape[:2]
                            def draw_line(img, l, color):
                                a,b,c = l
                                # intersections
                                pts=[]
                                if abs(a) > 1e-8:
                                    x0 = -c/a
                                    x1 = -(c + b*(h2-1))/a
                                    if 0<=x0<w2 and 0<=x1<w2:
                                        cv2.line(img, (int(round(x0)),0), (int(round(x1)),h2-1), color, 1)
                                if abs(b) > 1e-8:
                                    y0 = -c/b
                                    y1 = -(c + a*(w2-1))/b
                                    if 0<=y0<h2 and 0<=y1<h2:
                                        cv2.line(img, (0,int(round(y0))), (w2-1,int(round(y1))), color, 1)
                            for l in l2_via0:
                                draw_line(draw2, l, (0,255,0))
                            for l in l2_via1:
                                draw_line(draw2, l, (255,0,0))
                    except Exception:
                        pass

                # Concatenate view-0 | view-1 | view-2
                try:
                    log_dict[f'{suffix}np_{idx}'] = wandb.Image(cv2.hconcat([draw0, draw1, draw2]))
                except Exception:
                    pass

            # Planar triplet: show view-0 (source), view-1 (paired), view-2 (cycle consistency)
            if ci.get('pair_type') == 'planar' and ci.get('h_cycle', None) is not None:
                hc = ci['h_cycle']
                H01, H12, H02 = hc['H01'], hc['H12'], hc['H02']
                kpts0 = hc.get('kpts0', None)
                if kpts0 is None or kpts0.numel()==0:
                    continue
                # Build triplet images: view-0, view-1, synthesized view-2 via H02
                img0_t = (batch['image0'][idx].detach().cpu() * 255).to(torch.uint8)
                img1_t = (batch['image1'][idx].detach().cpu() * 255).to(torch.uint8)
                def to_rgb(x):
                    ch = vis_rgb_channels(x.shape[0])
                    return x[ch,:,:].permute(1,2,0).numpy()
                img0_rgb = to_rgb(img0_t)
                img1_rgb = to_rgb(img1_t)
                h0, w0 = img0_rgb.shape[:2]
                H02_np = H02.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(H02) else np.asarray(H02, dtype=np.float32)
                img2 = cv2.warpPerspective(img0_rgb, H02_np, (w0, h0))
                # Convert kpts0 to pixels in view-0 geometry
                if kpts0.abs().max() <= 1.0:
                    x0 = ((kpts0 + 1) / 2) * kpts0.new_tensor([w0-1, h0-1])
                else:
                    x0 = kpts0
                ones = torch.ones((x0.shape[0],1), device=x0.device, dtype=x0.dtype)
                x0h = torch.cat([x0, ones], 1)
                def warpH(H, x):
                    y = (H @ x.t()).t(); y = y[:,:2] / (y[:,2:3] + 1e-8); return y
                u2a = warpH(H12 @ H01, x0h)
                u2b = warpH(H02, x0h)
                # Draw markers
                draw0 = img0_rgb.copy()
                for p in x0.detach().cpu().numpy().astype(int):
                    cv2.circle(draw0, (int(p[0]), int(p[1])), 2, (255,255,255), -1)
                draw1 = img1_rgb.copy()
                draw = img2.copy()
                for p in u2a.detach().cpu().numpy().astype(int):
                    cv2.circle(draw, (int(p[0]), int(p[1])), 2, (0,255,0), -1)
                for p in u2b.detach().cpu().numpy().astype(int):
                    cv2.circle(draw, (int(p[0]), int(p[1])), 2, (0,0,255), -1)
                # Concatenate view-0 | view-1 | view-2
                try:
                    log_dict[f'{suffix}pl_{idx}'] = wandb.Image(cv2.hconcat([draw0, draw1, draw]))
                except Exception:
                    pass

        if len(log_dict) > 0:
            try:
                logger.experiment.log(log_dict)
            except Exception as e:
                print(f"Warning: WandB triplet log failed: {e}")
    except Exception as e:
        print(f"Warning: Triplet viz failed: {e}")