import numpy as np
import matplotlib.pyplot as plt

def plot_matches(img0: np.ndarray, img1: np.ndarray,
                 mkpts0: np.ndarray, mkpts1: np.ndarray,
                 mask, out_path: str, title: str = ""):
    """Side-by-side match visualisation; green=inlier, red=outlier, blue=mask-less."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:] = img1

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(canvas)
    ax.axis('off')
    if title:
        ax.set_title(title)
    for i, (p0, p1) in enumerate(zip(mkpts0, mkpts1)):
        c = '#3b82f6' if mask is None else ('#22c55e' if mask[i] else '#ef4444')
        ax.plot([p0[0], p1[0] + w0], [p0[1], p1[1]], color=c, linewidth=0.5, alpha=0.8)
        ax.scatter([p0[0], p1[0] + w0], [p0[1], p1[1]], color=c, s=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)