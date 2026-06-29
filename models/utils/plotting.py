import os
try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

from .metrics import to_numpy_safe


def save_success_curve_plot(taus, suc, title: str, xlabel: str, ylabel: str, out_path: str) -> None:
    if not _HAVE_PLT:
        return
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(to_numpy_safe(taus), to_numpy_safe(suc), label='Success curve', color='#1f77b4', linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim(0.0, 1.0)
        if hasattr(taus, 'numel') and taus.numel() > 0:
            plt.xlim(float(taus.min().item()), float(taus.max().item()))
        else:
            plt.xlim(0.0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception:
        pass


