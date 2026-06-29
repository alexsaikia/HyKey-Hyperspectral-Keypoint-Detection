import torch
from typing import Optional, Tuple


def mnn_with_gates(desc0: torch.Tensor, desc1: torch.Tensor,
                   min_sim_threshold: Optional[float] = None,
                   lowe_ratio_threshold: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mutual nearest neighbor matching (cosine sim), optional min-sim and Lowe ratio gates.
    Returns (idx0, idx1, sims_top1).
    """
    if desc0.numel() == 0 or desc1.numel() == 0:
        dev = desc0.device if desc0.numel() else (desc1.device if desc1.numel() else 'cpu')
        return torch.empty(0, dtype=torch.long, device=dev), torch.empty(0, dtype=torch.long, device=dev), torch.empty(0, device=dev)
    sim = desc0 @ desc1.t()
    mi0 = torch.argmax(sim, dim=1)
    mi1 = torch.argmax(sim, dim=0)
    mask_mnn = torch.arange(mi0.numel(), device=sim.device) == mi1[mi0]
    idx0 = torch.where(mask_mnn)[0]
    idx1 = mi0[idx0]
    sims_top1 = sim[idx0, idx1] if idx0.numel() > 0 else sim.new_zeros(0)
    if min_sim_threshold is not None:
        keep = sims_top1 >= float(min_sim_threshold)
        idx0 = idx0[keep]; idx1 = idx1[keep]; sims_top1 = sims_top1[keep]
    if lowe_ratio_threshold is not None and lowe_ratio_threshold > 0.0 and idx0.numel() > 0 and sim.shape[1] >= 2:
        top2 = torch.topk(sim[idx0], k=2, dim=1)
        ratio = top2.values[:, 0] / (top2.values[:, 1] + 1e-12)
        keep = ratio >= float(lowe_ratio_threshold)
        idx0 = idx0[keep]; idx1 = idx1[keep]; sims_top1 = sims_top1[keep]
    return idx0, idx1, sims_top1


