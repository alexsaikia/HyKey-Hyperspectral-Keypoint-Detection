import torch

def weighted_mean(x, w, eps=1e-8):
    """Compute the weighted mean of a tensor."""
    w_sum = w.sum().clamp(min=eps)
    return (x * w).sum() / w_sum

def huber(x, delta=1.0):
    """Compute Huber loss."""
    return torch.where(torch.abs(x) < delta, 0.5 * x ** 2, delta * (torch.abs(x) - 0.5 * delta))