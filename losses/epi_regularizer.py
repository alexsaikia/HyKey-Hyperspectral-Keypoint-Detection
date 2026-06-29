import torch


def epi_soft_regularizer(
    d0_list, d1_list,         # lists of (Ni,D) descriptor tensors (require_grad=True)
    k0_list, k1_list,         # lists of (Ni,2) keypoints in **pixels** (no grad OK)
    K0, K1, R, t,             # per-batch intrinsics/pose; each shape (B, 3,3)/(B,3) etc.
    tau_match: float = 0.07,  # softmax temperature on descriptor sims
    huber_px: float = 3.0,    # Huber delta in **pixels**
    max_kpts: int = 512,      # cap per-view keypoints for compute
    symmetric: bool = True,  # enable img1->img0 term too (slower)
    clip_percentile: float = None,  # clip residuals at this quantile per-sample (e.g., 0.99)
    max_px: float = None,     # hard cap on residuals in pixels after sqrt
    trim_ratio: float = 0.0,  # drop top fraction of per-anchor losses before averaging (e.g., 0.1)
):
    """
    Differentiable epipolar regularizer:
      For each anchor i in image-0, compute P_ij = softmax( (d0_i · d1_j) / tau_match ).
      Loss_i = sum_j P_ij * Huber( Sampson_px(x0_i, x1_j) ).

    Returns a scalar tensor (mean over batch).
    """

    device = d0_list[0].device if len(d0_list) else (K0.device if torch.is_tensor(K0) else "cpu")
    total = torch.tensor(0.0, device=device)
    count = 0

    eps = 1e-12

    def skew(v):
        x, y, z = v.view(-1)
        M = v.new_zeros(3, 3)
        M[0,1], M[0,2] = -z,  y
        M[1,0], M[1,2] =  z, -x
        M[2,0], M[2,1] = -y,  x
        return M

    def build_F(K0i, K1i, Ri, ti):
        E = skew(ti.view(-1)) @ Ri
        return torch.linalg.inv(K1i).T @ E @ torch.linalg.inv(K0i)

    def sampson_matrix(Fgt, x0_px, x1_px):
        N0 = x0_px.shape[0]; N1 = x1_px.shape[0]
        if N0 == 0 or N1 == 0:
            return x0_px.new_zeros((N0, N1))
        x0h = torch.cat([x0_px, torch.ones(N0, 1, device=x0_px.device, dtype=x0_px.dtype)], dim=1)
        x1h = torch.cat([x1_px, torch.ones(N1, 1, device=x1_px.device, dtype=x1_px.dtype)], dim=1)
        Ex0  = (Fgt @ x0h.T).T
        Etx1 = (Fgt.T @ x1h.T).T
        num = x1h @ Ex0.T
        den = (Ex0[:, :2].pow(2).sum(dim=1)).unsqueeze(0) + (Etx1[:, :2].pow(2).sum(dim=1)).unsqueeze(1) + eps
        d2  = (num * num) / den
        return d2.T

    def huber_on_sqrt(d2, delta, cap_px=None, clip_q=None):
        s = torch.sqrt(torch.clamp_min(d2, 0.0) + eps)
        if cap_px is not None and cap_px > 0:
            s = torch.clamp_max(s, float(cap_px))
        if clip_q is not None and 0.5 < float(clip_q) < 1.0 and s.numel() >= 4:
            q = torch.quantile(s.detach(), float(clip_q))
            s = torch.clamp_max(s, q)
        m = s <= delta
        return torch.where(m, 0.5 * s * s, delta * (s - 0.5 * delta))

    B = len(d0_list)
    for i in range(B):
        d0 = d0_list[i]; d1 = d1_list[i]
        x0 = k0_list[i]; x1 = k1_list[i]
        if d0.numel() == 0 or d1.numel() == 0 or x0.numel() == 0 or x1.numel() == 0:
            continue
        if d0.shape[0] > max_kpts:
            d0 = d0[:max_kpts]; x0 = x0[:max_kpts]
        if d1.shape[0] > max_kpts:
            d1 = d1[:max_kpts]; x1 = x1[:max_kpts]
        K0i = K0[i] if torch.is_tensor(K0) and K0.dim() >= 3 else torch.as_tensor(K0, device=device, dtype=d0.dtype)
        K1i = K1[i] if torch.is_tensor(K1) and K1.dim() >= 3 else torch.as_tensor(K1, device=device, dtype=d0.dtype)
        Ri  = R[i]  if torch.is_tensor(R)  and R.dim()  >= 3 else torch.as_tensor(R,  device=device, dtype=d0.dtype)
        ti  = t[i]  if torch.is_tensor(t)  and t.dim()  >= 2 else torch.as_tensor(t,  device=device, dtype=d0.dtype)
        Fgt = build_F(K0i, K1i, Ri, ti)
        S01 = (d0 @ d1.T) / max(tau_match, 1e-6)
        P01 = torch.softmax(S01, dim=1)
        d2_01 = sampson_matrix(Fgt, x0, x1)
        hv_01 = huber_on_sqrt(d2_01, huber_px, cap_px=max_px, clip_q=clip_percentile)
        per_anchor_01 = (P01 * hv_01).sum(dim=1)
        if per_anchor_01.numel() > 1 and trim_ratio is not None and float(trim_ratio) > 0.0:
            keep = max(1, int(round(per_anchor_01.numel() * (1.0 - float(trim_ratio)))))
            loss_01 = torch.topk(-per_anchor_01, k=keep).values.neg().mean()
        else:
            loss_01 = per_anchor_01.mean()
        if symmetric:
            R10 = Ri.T
            t10 = (-Ri.T @ ti.view(3, 1)).view(3)
            F10 = build_F(K1i, K0i, R10, t10)
            S10 = (d1 @ d0.T) / max(tau_match, 1e-6)
            P10 = torch.softmax(S10, dim=1)
            d2_10 = sampson_matrix(F10, x1, x0)
            hv_10 = huber_on_sqrt(d2_10, huber_px, cap_px=max_px, clip_q=clip_percentile)
            per_anchor_10 = (P10 * hv_10).sum(dim=1)
            if per_anchor_10.numel() > 1 and trim_ratio is not None and float(trim_ratio) > 0.0:
                keep = max(1, int(round(per_anchor_10.numel() * (1.0 - float(trim_ratio)))))
                loss_10 = torch.topk(-per_anchor_10, k=keep).values.neg().mean()
            else:
                loss_10 = per_anchor_10.mean()
            total = total + 0.5 * (loss_01 + loss_10)
        else:
            total = total + loss_01
        count += 1

    return total / max(count, 1)
