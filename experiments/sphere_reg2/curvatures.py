"""
Calcola le curvature (mean H, gaussian K) della WM surface usando cortech.
Salva i risultati su disco per usarli dopo nello script di valutazione.

""" 

import numpy as np
import cortech
from pathlib import Path

# ============== CONFIG ==============
SUBJECTS = ["sub-0259", "sub-1297"]
HEMI = "lh"
DATASET_NAME = "OASIS3"
SMOOTH_ITER = 10

ORIG_ROOT = Path(f"/projects/brainnet-data/sphere-reg/{DATASET_NAME}")
OUT_DIR = Path(f"/home/s252125/my_project/experiments/sphere_reg2/curvatures/{DATASET_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ====================================


for sub in SUBJECTS:
    print(f"\nProcessing {sub}")
    wm_path = ORIG_ROOT / sub / f"{HEMI}.white"

    if not wm_path.exists():
        print(f"  {wm_path} non esiste, skippato")
        continue

    try:
        s = cortech.Surface.from_file(str(wm_path))
        print(f"  WM surface loaded: {len(s.vertices)} vertices")

        curv = s.compute_curvature(smooth_iter=SMOOTH_ITER)

        # H = mean curvature, K = gaussian curvature
        H = np.asarray(curv.H, dtype=np.float32)
        K = np.asarray(curv.K, dtype=np.float32)

        out_path = OUT_DIR / f"{sub}_{HEMI}_curvature.npz"
        np.savez(out_path, H=H, K=K)

        print(f"  H: shape={H.shape}, min={H.min():.4f}, max={H.max():.4f}, mean={H.mean():.4f}")
        print(f"  K: shape={K.shape}, min={K.min():.4f}, max={K.max():.4f}, mean={K.mean():.4f}")
        print(f"  Saved -> {out_path}")

    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nDone. All curvatures saved in: {OUT_DIR}")