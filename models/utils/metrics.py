import torch
import numpy as np


def auc_success_curve(errs: torch.Tensor, T: float, device=None) -> torch.Tensor:
    try:
        if errs is None:
            return torch.tensor(0.0, device=device or 'cpu')
        if not torch.is_tensor(errs):
            errs = torch.tensor(errs, dtype=torch.float32, device=device or 'cpu')
        if errs.numel() == 0:
            return torch.tensor(0.0, device=errs.device)
        T_safe = float(T) if float(T) > 0 else 1e-6
        taus = torch.linspace(0.0, float(T), max(int(float(T)), 1) + 1, device=errs.device)
        suc = (errs.unsqueeze(0) <= taus.unsqueeze(1)).float().mean(dim=1)
        return torch.trapz(suc, taus) / T_safe
    except Exception:
        return torch.tensor(0.0, device=device or (errs.device if torch.is_tensor(errs) else 'cpu'))


def success_curve(errs: torch.Tensor, T: float, device=None):
    if errs is None:
        dev = device or 'cpu'
        return torch.tensor([], device=dev), torch.tensor([], device=dev)
    if not torch.is_tensor(errs):
        errs = torch.tensor(errs, dtype=torch.float32, device=device or 'cpu')
    if errs.numel() == 0:
        return torch.tensor([], device=errs.device), torch.tensor([], device=errs.device)
    taus = torch.linspace(0.0, float(T), max(int(float(T)), 1) + 1, device=errs.device)
    suc = (errs.unsqueeze(0) <= taus.unsqueeze(1)).float().mean(dim=1)
    return taus, suc


def to_numpy_safe(tensor: torch.Tensor):
    try:
        if tensor is None:
            return None
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)
    except Exception:
        return np.asarray([])


