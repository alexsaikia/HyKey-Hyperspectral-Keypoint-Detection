import torch
import torch.nn.functional as F

from models.utils.utils import warp, compute_keypoints_distance, mutual_argmin, EmptyTensorError
from models.utils.matches import mnn_with_gates


class CorrespondenceComputer:
    """Encapsulates correspondence computation logic used by HyKey models.

    This class forwards unknown attribute access to the bound model so that
    existing logic can refer to `self.<attr>` unchanged.
    """

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        # Delegate attribute access to the underlying model (dkd, thresholds, etc.)
        try:
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(name)

    def mask_keep_indices(self, k, mask, H, W):
        """Vectorized equivalent of the per-keypoint train_mask filter.

        Returns a LongTensor of indices (ascending) into ``k`` for keypoints that
        are finite, land inside the image, and fall on a True/non-zero mask pixel.
        This matches the original Python loop exactly but avoids one GPU->CPU sync
        per keypoint (the dominant cost when ~400-800 keypoints are processed).
        """
        if k.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=k.device)
        kpx = (k + 1) / 2 * k.new_tensor([W - 1, H - 1])
        finite = torch.isfinite(kpx).all(dim=1)
        # int(.round()) in the original truncates toward zero after rounding; round()
        # to nearest then cast is equivalent for the non-negative in-bounds case.
        xr = kpx[:, 0].round().long()
        yr = kpx[:, 1].round().long()
        in_bounds = finite & (xr >= 0) & (xr < W) & (yr >= 0) & (yr < H)
        xr_c = xr.clamp(0, W - 1)
        yr_c = yr.clamp(0, H - 1)
        mask_vals = mask[0, yr_c, xr_c].to(torch.bool)
        keep_bool = in_bounds & mask_vals
        return torch.nonzero(keep_bool, as_tuple=False)[:, 0]

    def select_pair_type_and_rand(self, batch, i, pair_type, rand):
        ptype_i = 'planar'
        if 'pair_type' in batch:
            if isinstance(batch['pair_type'], (list, tuple)):
                ptype_i = batch['pair_type'][i]
            elif isinstance(batch['pair_type'], str):
                ptype_i = batch['pair_type']
        if pair_type is not None:
            ptype_i = pair_type
        use_rand = (rand and (ptype_i != 'nonplanar'))
        return ptype_i, use_rand

    def clamp01(self, k):
        if k.numel()==0: return k
        eps = 1e-6
        return k.clamp(min=-1.0 + eps, max=1.0 - eps)

    def sample_desc(self, dmap, kxy):
        if kxy.numel()==0:
            return torch.empty(0, dmap.shape[1], device=dmap.device, dtype=dmap.dtype)
        kxy = self.clamp01(kxy)
        d = F.grid_sample(dmap.unsqueeze(0), kxy.view(1,1,-1,2), mode='bilinear', align_corners=True)[0,:,0,:].t()
        d = F.normalize(d, p=2, dim=1)
        # Ensure descriptors participate in backpropagation
        if not d.requires_grad:
            d.requires_grad_(True)
        return d

    def sample_scores_and_desc(self, pred, i, k):
        s = F.grid_sample(pred['scores_map'][i:i+1], k.view(1,1,-1,2), mode='bilinear', align_corners=True).squeeze()
        d = self.sample_desc(pred['descriptor_map'][i], k)
        return s, d

    def nms_local(self, k, s, H, W):
        if k.numel()==0: return k, s, torch.arange(0,0, device=k.device)
        kpx_ = (k/2+0.5) * k.new_tensor([W-1, H-1])
        dist = compute_keypoints_distance(kpx_.detach(), kpx_.detach())
        local = dist < self.radius
        # Vectorized equivalent of the per-anchor greedy suppression.
        # Original loop semantics (order-independent, since `keep` is only ever set
        # False and each decision is computed from the fixed `s`/`local`):
        #   p is suppressed iff there exists an anchor ii with >1 neighbour such that
        #   p is in N(ii) but p is not the (first-)argmax of s over N(ii).
        # winner[ii] = first index achieving max score within N(ii); torch.argmax and
        # the original s[nbr].argmax() both return the lowest such index, so ties match.
        N = k.shape[0]
        deg = local.sum(dim=1)
        masked = torch.where(local, s.unsqueeze(0).expand(N, N),
                             s.new_full((), float('-inf')))
        winner = masked.argmax(dim=1)                       # (N,) winner index per anchor
        ar = torch.arange(N, device=k.device)
        suppress = (local & (deg > 1).unsqueeze(1)
                    & (winner.unsqueeze(1) != ar.unsqueeze(0))).any(dim=0)
        keep = ~suppress
        sel = torch.where(keep)[0]
        return k[sel], s[sel], sel

    def compute(self, pred0, pred1, batch, rand=None,
                extra_pred=None, extra_batch=None, pair_type=None):
        """Moved from HyKey.compute_correspondence (behavior preserved)."""
        # NOTE: The body below mirrors the original implementation to avoid behavior changes.
        if rand is None:
            rand = self.training

        B, _, H, W = pred0['scores_map'].shape
        wh = pred0['scores_map'][0].new_tensor([[W - 1, H - 1]])

        # Detect keypoints from score maps
        kp0_list, disp0_list, kpscore0_list = self.dkd.detect_keypoints(pred0['scores_map'])
        kp1_list, disp1_list, kpscore1_list = self.dkd.detect_keypoints(pred1['scores_map'])

        # Optional 3rd-view detection
        if extra_pred is not None and ('scores_map' in extra_pred):
            kp2_list, disp2_list, kpscore2_list = self.dkd.detect_keypoints(extra_pred['scores_map'])
        else:
            kp2_list, disp2_list, kpscore2_list = [], [], []

        # Build return preds (with the exact kpts/descs sampled here)
        p0_ret = pred0.copy(); p1_ret = pred1.copy()
        p0_ret['keypoints'] = []; p1_ret['keypoints'] = []
        p0_ret['scores']    = []; p1_ret['scores']    = []
        p0_ret['descriptors'] = []; p1_ret['descriptors'] = []
        p0_ret['score_dispersity'] = disp0_list
        p1_ret['score_dispersity'] = disp1_list

        correspondences = []

        for i in range(B):
            k0 = kp0_list[i]; k1 = kp1_list[i]
            if k0.numel():
                finite = torch.isfinite(k0).all(dim=1)
                k0 = k0[finite]
            if k1.numel():
                finite = torch.isfinite(k1).all(dim=1)
                k1 = k1[finite]

            # Respect train_mask to avoid NaN descriptors.
            # Vectorized gather (no per-keypoint Python loop / GPU sync): produces the
            # exact same kept indices as the original per-point check, but in one shot.
            if 'train_mask' in pred0 and k0.numel():
                mask = pred0['train_mask'][i]
                keep = self.mask_keep_indices(k0, mask, H, W)
                if keep.numel():
                    k0 = k0[keep]
                    disp0_list[i] = disp0_list[i][keep]

            if 'train_mask' in pred1 and k1.numel():
                mask = pred1['train_mask'][i]
                keep = self.mask_keep_indices(k1, mask, H, W)
                if keep.numel():
                    k1 = k1[keep]
                    disp1_list[i] = disp1_list[i][keep]

            # Early exit when any side has no kpts
            if (k0.numel()==0) or (k1.numel()==0):
                p0_ret['keypoints'].append(k0); p1_ret['keypoints'].append(k1)
                dev = pred0['scores_map'].device
                p0_ret['scores'].append(torch.empty(0, device=dev)); p1_ret['scores'].append(torch.empty(0, device=dev))
                D = pred0['descriptor_map'].shape[1]
                p0_ret['descriptors'].append(torch.empty(0, D, device=dev)); p1_ret['descriptors'].append(torch.empty(0, D, device=dev))
                correspondences.append({'pair_type': pair_type or batch.get('pair_type','planar'),
                                        'correspondence0': None, 'correspondence1': None,
                                        'dist': pred0['scores_map'][0].new_tensor(0)})
                continue

            # Disable random points for nonplanar pairs (your request)
            ptype_i, use_rand = self.select_pair_type_and_rand(batch, i, pair_type, rand)

            if use_rand:
                # add a handful of random points per view (same count as existing to keep balance)
                r0 = torch.rand(len(k0), 2, device=k0.device) * 2 - 1
                r1 = torch.rand(len(k1), 2, device=k1.device) * 2 - 1
                k0 = torch.cat([k0, r0], dim=0)
                k1 = torch.cat([k1, r1], dim=0)

            # Clamp and sample maps
            k0 = self.clamp01(k0); k1 = self.clamp01(k1)
            s0, d0 = self.sample_scores_and_desc(pred0, i, k0)
            s1, d1 = self.sample_scores_and_desc(pred1, i, k1)

            # Local NMS by radius on each view (keep strongest per neighborhood)
            k0, s0, sel0 = self.nms_local(k0, s0, H, W)
            k1, s1, sel1 = self.nms_local(k1, s1, H, W)
            d0 = d0[sel0]; d1 = d1[sel1]

            # Store into return preds
            p0_ret['keypoints'].append(k0); p1_ret['keypoints'].append(k1)
            p0_ret['scores'].append(s0);    p1_ret['scores'].append(s1)
            p0_ret['descriptors'].append(d0); p1_ret['descriptors'].append(d1)

            # Common ids
            ids0 = torch.arange(k0.shape[0], device=k0.device, dtype=torch.long)
            ids1 = torch.arange(k1.shape[0], device=k1.device, dtype=torch.long)

            # Pixel coordinates for later geometry
            k0px = ((k0 + 1)/2) * k0.new_tensor([W-1, H-1])
            k1px = ((k1 + 1)/2) * k1.new_tensor([W-1, H-1])

            if ptype_i == 'nonplanar':
                # Helper to select the i-th row from possibly full-batch arrays using orig_idx mapping
                def sel_cam_row(arr):
                    if arr is None:
                        return None
                    if torch.is_tensor(arr) and arr.dim() >= 2:
                        if arr.dim() == 3:
                            # Prefer matching the current subset size; else fall back to original index map
                            if arr.shape[0] == B:
                                return arr[i]
                            elif ('orig_idx' in batch) and torch.is_tensor(batch['orig_idx']) and batch['orig_idx'].numel() >= (i+1):
                                oi = int(batch['orig_idx'][i].item())
                                # Guard if arr includes original-batch rows
                                if oi < arr.shape[0]:
                                    return arr[oi]
                                else:
                                    return arr[min(i, arr.shape[0]-1)]
                            else:
                                return arr[min(i, arr.shape[0]-1)]
                        elif arr.dim() == 2:
                            # Handle batched vectors/matrices with shape (B, N)
                            if arr.shape[0] == B:
                                return arr[i]
                            elif ('orig_idx' in batch) and torch.is_tensor(batch['orig_idx']) and batch['orig_idx'].numel() >= (i+1):
                                oi = int(batch['orig_idx'][i].item())
                                if oi < arr.shape[0]:
                                    return arr[oi]
                                else:
                                    return arr[min(i, arr.shape[0]-1)]
                            else:
                                return arr
                        else:
                            return arr
                    # Non-tensor fallback
                    try:
                        return arr[i]
                    except Exception:
                        return arr
                # Mutual NN on descriptors
                # MNN with model-configured gates
                ids0_d, ids1_d, _ = mnn_with_gates(d0, d1, getattr(self, 'np_min_sim', None), getattr(self, 'np_ratio', None))

                # Fallback: if no matches survive, synthesize pseudo-correspondences from best matches
                if ids0_d.numel() == 0 and (d0.numel() > 0 and d1.numel() > 0):
                    try:
                        M = 16
                        # select top-M keypoints from view-1 by score
                        m = min(M, s1.shape[0])
                        if m > 0:
                            vals, sel1 = torch.topk(s1, k=m)
                            ids1_d = sel1
                            # map to view-0 using nearest by descriptor similarity
                            if d0.numel() > 0:
                                best0 = (d1[ids1_d] @ d0.t()).argmax(dim=1)
                                ids0_d = best0
                    except Exception:
                        pass

                p0_px = k0px[ids0_d] if ids0_d.numel() else k0px.new_zeros((0,2))
                p1_px = k1px[ids1_d] if ids1_d.numel() else k1px.new_zeros((0,2))
                conf = (s0[ids0_d] * s1[ids1_d]) if ids0_d.numel() else s0.new_zeros(0)

                # Intrinsics/poses → E_gt selection (keep all matches; no Sampson filtering)
                K0 = batch.get('K0', None); K1 = batch.get('K1', None)
                R_01 = batch.get('R_01', None); t_01 = batch.get('t_01', None)
                E_gt = None; K0_i = None; K1_i = None; R_i = None; t_i = None
                if (K0 is not None) and (K1 is not None) and (R_01 is not None) and (t_01 is not None) and (p0_px.numel()>0):
                    K0_i = sel_cam_row(K0) if torch.is_tensor(K0) else torch.as_tensor(K0, device=p0_px.device, dtype=p0_px.dtype)
                    K1_i = sel_cam_row(K1) if torch.is_tensor(K1) else torch.as_tensor(K1, device=p0_px.device, dtype=p0_px.dtype)
                    R_i  = sel_cam_row(R_01) if torch.is_tensor(R_01) else torch.as_tensor(R_01, device=p0_px.device, dtype=p0_px.dtype)
                    ti_  = sel_cam_row(t_01) if torch.is_tensor(t_01) else torch.as_tensor(t_01, device=p0_px.device, dtype=p0_px.dtype)
                    t_i  = ti_.view(3)

                    def skew(v):
                        tx,ty,tz = v.view(-1)
                        M = v.new_zeros(3,3); M[0,1],M[0,2] = -tz,  ty; M[1,0],M[1,2] =  tz,-tx; M[2,0],M[2,1] = -ty, tx
                        return M
                    def _E(R,t):
                        return skew(t.view(-1)) @ R
                    def median_sampson(E, p0, p1, K0, K1):
                        x0 = torch.cat([p0, p0.new_ones(p0.shape[0],1)], 1)
                        x1 = torch.cat([p1, p1.new_ones(p1.shape[0],1)], 1)
                        x0n = (torch.inverse(K0) @ x0.t()).t()
                        x1n = (torch.inverse(K1) @ x1.t()).t()
                        Ex0  = (E  @ x0n.t()).t()
                        Etx1 = (E.t() @ x1n.t()).t()
                        x1tEx0 = (x1n * Ex0).sum(1)
                        denom = Ex0[:,0].pow(2)+Ex0[:,1].pow(2)+Etx1[:,0].pow(2)+Etx1[:,1].pow(2) + 1e-12
                        d = x1tEx0.pow(2) / denom
                        return d.median()

                    E01     = _E(R_i, t_i)
                    R10     = R_i.t(); t10 = (-R_i.t() @ t_i.view(3,1)).view(3)
                    E01_alt = _E(R10.t(), (-R10 @ t10.view(3,1)).view(3))
                    m1 = median_sampson(E01, p0_px, p1_px, K0_i, K1_i)
                    m2 = median_sampson(E01_alt, p0_px, p1_px, K0_i, K1_i)
                    E_gt = E01 if (m1 <= m2) else E01_alt

                # Optional geometric consistency gating (Sampson error in pixel domain)
                try:
                    if (E_gt is not None) and (p0_px.numel() > 0) and (p1_px.numel() > 0):
                        K0inv = torch.inverse(K0_i); K1inv = torch.inverse(K1_i)
                        Fgt = K1inv.t() @ E_gt @ K0inv
                        # Build homogeneous pixel coords
                        x0h = torch.cat([p0_px, p0_px.new_ones(p0_px.shape[0], 1)], dim=1)
                        x1h = torch.cat([p1_px, p1_px.new_ones(p1_px.shape[0], 1)], dim=1)
                        Ex0  = (Fgt  @ x0h.t()).t()
                        Etx1 = (Fgt.t() @ x1h.t()).t()
                        x1tEx0 = (x1h * Ex0).sum(dim=1)
                        denom = Ex0[:,0].pow(2) + Ex0[:,1].pow(2) + Etx1[:,0].pow(2) + Etx1[:,1].pow(2) + 1e-12
                        d_sampson = x1tEx0.pow(2) / denom  # in pixel^2 units
                        th2 = float(self.np_sampson_px) ** 2
                        gmask = d_sampson <= th2
                        if gmask.numel() > 0 and gmask.sum() < gmask.numel():
                            # Filter matched ids by geometry
                            ids0_d = ids0_d[gmask]
                            ids1_d = ids1_d[gmask]
                            p0_px = p0_px[gmask]
                            p1_px = p1_px[gmask]
                            if conf.numel() > 0:
                                conf = conf[gmask]
                except Exception:
                    pass

                ci = {
                    'pair_type': 'nonplanar',
                    'ids0': ids0, 'ids1': ids1,
                    'ids0_d': ids0_d, 'ids1_d': ids1_d,
                    'scores0': s0, 'scores1': s1,
                    'S01': (d0 @ d1.t()), 'S10': (d1 @ d0.t()),
                    'p0_px': p0_px, 'p1_px': p1_px, 'conf': conf,
                    'K0': K0_i, 'K1': K1_i, 'E_gt': E_gt,
                    'R01': R_i, 't01': t_i,
                    'H1': H, 'W1': W, 'H0': H, 'W0': W,
                    'batch_index': i,
                }

                correspondences.append(ci)

            elif (ptype_i == 'planar') and ('H_mat' in batch):
                # Classical homography supervision path
                H01 = batch['H_mat'][i]
                try:
                    warp01 = {'mode':'homo', 'homography_matrix': H01, 'width': W, 'height': H}
                    warp10 = {'mode':'homo', 'homography_matrix': torch.inverse(H01), 'width': W, 'height': H}
                    k0px_ = ((k0/2+0.5)*wh).to(k0)
                    k1px_ = ((k1/2+0.5)*wh).to(k1)
                    k0px, k01px, ids0_full, ids0_out = warp(k0px_, warp01)
                    k1px, k10px, ids1_full, ids1_out = warp(k1px_, warp10)
                    if (k0px.numel()==0) or (k1px.numel()==0):
                        raise EmptyTensorError

                    d01 = compute_keypoints_distance(k0px, k10px)
                    d10 = compute_keypoints_distance(k1px, k01px)
                    dL2 = (d01 + d10.t()) / 2.
                    midx = mutual_argmin(dL2)
                    dmm  = dL2[midx]
                    ok   = (dmm.detach() < self.train_gt_th)
                    ids0_d = midx[0][ok]
                    ids1_d = midx[1][ok]
                    corr0  = ids0_full[ids0_d]
                    corr1  = ids1_full[ids1_d]

                    # Sparse descriptor tensors for logging & dense losses
                    d0v = d0[ids0_full]
                    d1v = d1[ids1_full]
                    S01 = d0v @ d1v.t()

                    # nn indices (for dense reliability losses)
                    if (k01px.numel()>0) and (k1px.numel()>0):
                        nn01 = compute_keypoints_distance(k01px.detach(), k1px.detach()).argmin(dim=1)
                    else:
                        nn01 = torch.empty(0, dtype=torch.long, device=d0.device)
                    if (k10px.numel()>0) and (k0px.numel()>0):
                        nn10 = compute_keypoints_distance(k10px.detach(), k0px.detach()).argmin(dim=1)
                    else:
                        nn10 = torch.empty(0, dtype=torch.long, device=d0.device)

                    ci = {
                        'pair_type': 'planar',
                        'correspondence0': corr0, 'correspondence1': corr1,
                        'scores0': s0[ids0_full], 'scores1': s1[ids1_full],
                        'kpts01': 2 * k01px.detach()/wh - 1,
                        'kpts10': 2 * k10px.detach()/wh - 1,
                        'ids0': ids0, 'ids1': ids1,
                        'ids0_out': ids0_out, 'ids1_out': ids1_out,
                        'ids0_d': ids0_d, 'ids1_d': ids1_d,
                        'dist': dL2,
                        'dist_l1': (compute_keypoints_distance(k0px, k10px, p=1) + compute_keypoints_distance(k1px, k01px, p=1).t())/2.,
                        'S01': S01, 'S10': S01.t(),
                        'nn01': nn01, 'nn10': nn10,
                        'H0': H, 'W0': W,
                    }

                    correspondences.append(ci)

                except EmptyTensorError:
                    correspondences.append({'pair_type': 'planar', 'correspondence0': None, 'correspondence1': None,
                                            'dist': pred0['scores_map'][0].new_tensor(0)})
            else:
                # Unknown type; push a safe empty pack
                correspondences.append({'pair_type': ptype_i,
                                        'correspondence0': None, 'correspondence1': None,
                                        'dist': pred0['scores_map'][0].new_tensor(0)})

        return correspondences, p0_ret, p1_ret


