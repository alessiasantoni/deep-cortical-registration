"""
Calcola le curvature (mean H, gaussian K) della WM surface per un dataset completo.
Scopre i soggetti automaticamente, salta quelli gia' fatti, logga gli errori.
"""

import numpy as np
import cortech
from pathlib import Path
import csv

# ============== CONFIG ==============
HEMI = "lh"
DATASET_NAME = "OASIS3"
SMOOTH_ITER = 10

ORIG_ROOT = Path(f"/projects/brainnet-data/sphere-reg/{DATASET_NAME}")
OUT_DIR = Path(f"/home/s252125/my_project/experiments/sphere_reg2/curvatures/{DATASET_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SKIP_EXISTING = True   # se True, non ricalcola soggetti gia' salvati
# ====================================


def discover_subjects(root: Path, hemi: str):
    """Trova tutti i soggetti che hanno il file <hemi>.white."""
    subjects = []
    for sub_dir in sorted(root.iterdir()):
        if not sub_dir.is_dir():
            continue
        if (sub_dir / f"{hemi}.white").exists():
            subjects.append(sub_dir.name)
    return subjects


subjects = discover_subjects(ORIG_ROOT, HEMI)
print(f"Trovati {len(subjects)} soggetti con {HEMI}.white in {ORIG_ROOT}")

log_path = OUT_DIR / f"_curvature_log_{HEMI}.csv"
log_rows = []

n_done, n_skip, n_err = 0, 0, 0

for i, sub in enumerate(subjects, 1):
    out_path = OUT_DIR / f"{sub}_{HEMI}_curvature.npz"

    if SKIP_EXISTING and out_path.exists():
        n_skip += 1
        continue

    wm_path = ORIG_ROOT / sub / f"{HEMI}.white"
    print(f"[{i}/{len(subjects)}] {sub}", end="  ")

    try:
        s = cortech.Surface.from_file(str(wm_path))
        curv = s.compute_curvature(smooth_iter=SMOOTH_ITER)

        H = np.asarray(curv.H, dtype=np.float32)
        K = np.asarray(curv.K, dtype=np.float32)

        np.savez(out_path, H=H, K=K)

        print(f"OK  n_vert={H.shape[0]}  H[{H.min():.3f},{H.max():.3f}]  "
              f"K[{K.min():.3f},{K.max():.3f}]")
        log_rows.append({
            "subject": sub, "status": "ok", "n_vertices": H.shape[0],
            "H_mean": float(H.mean()), "H_std": float(H.std()),
            "K_mean": float(K.mean()), "K_std": float(K.std()),
        })
        n_done += 1

    except Exception as e:
        print(f"ERROR: {e}")
        log_rows.append({"subject": sub, "status": f"error: {e}",
                         "n_vertices": -1, "H_mean": np.nan, "H_std": np.nan,
                         "K_mean": np.nan, "K_std": np.nan})
        n_err += 1

# Salva il log
if log_rows:
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

print(f"\nDone. Calcolati: {n_done}, skippati: {n_skip}, errori: {n_err}")
print(f"Salvati in: {OUT_DIR}")
print(f"Log: {log_path}")