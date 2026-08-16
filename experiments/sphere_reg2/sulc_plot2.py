import csv
from pathlib import Path
import matplotlib.pyplot as plt

RUNS_BASE = Path("experiments/sphere_reg2/runs")

RUNS = [    
    {"label": "ico=3", "dir": "20260610_081353_train_ico3", "w_sulc": 1.0, "color": "C3"},
    {"label": "ico=4", "dir": "20260609_133153_finetune_ico4", "w_sulc": 1.0, "color": "C0"},
    {"label": "ico=5", "dir": "20260609_160419_finetune_ico5", "w_sulc": 1.0, "color": "C1"},
    {"label": "ico=6", "dir": "20260609_181249_finetune_ico6", "w_sulc": 1.0, "color": "C2"},
]


def load_train_sulc_raw(run_dir, w_sulc):
    """TRAIN sulc da loss_history.csv (1 punto/epoca). De-pesata."""
    ep, sulc = [], []
    with open(run_dir / "loss_history.csv", newline="") as f:
        for row in csv.DictReader(f):
            ep.append(int(row["epoch"]))
            sulc.append(float(row["train_sulc"]) / w_sulc)
    return ep, sulc


def load_val_sulc_raw(run_dir):
    """VAL sulc dai file per-subject (gia' raw, niente W_SULC). Piu' rada."""
    ep, sulc = [], []
    for f in run_dir.glob("val_subject_losses_epoch_*.csv"):
        e = int(f.stem.split("_")[-1])
        vals = [float(r["sulc"]) for r in csv.DictReader(open(f, newline=""))]
        if vals:
            ep.append(e)
            sulc.append(sum(vals) / len(vals))
    order = sorted(range(len(ep)), key=lambda i: ep[i])
    return [ep[i] for i in order], [sulc[i] for i in order]


plt.figure(figsize=(12, 6))

offset = 0
boundaries = []

for info in RUNS:
    run_dir = RUNS_BASE / info["dir"]
    if not (run_dir / "loss_history.csv").exists():
        print(f"  [skip] {info['label']}")
        continue

    # TRAIN (linea piena, fitta)
    tr_ep, tr_sulc = load_train_sulc_raw(run_dir, info["w_sulc"])
    tr_x = [(e - tr_ep[0]) + offset + 1 for e in tr_ep]
    plt.plot(tr_x, tr_sulc, color=info["color"], label=f"{info['label']} train")

    # VAL (tratteggiata + marker, piu' rada). Stesso shift, ancorato all'inizio del train.
    va_ep, va_sulc = load_val_sulc_raw(run_dir)
    if va_ep:
        va_x = [(e - tr_ep[0]) + offset + 1 for e in va_ep]   # stessa origine del train
        plt.plot(va_x, va_sulc, color=info["color"], linestyle="--",
                 marker="o", markersize=4, label=f"{info['label']} val")

    offset = tr_x[-1]
    boundaries.append(offset)
    print(f"  {info['label']}: train {tr_sulc[-1]:.4f}, val {va_sulc[-1] if va_ep else float('nan'):.4f}")

for b in boundaries[:-1]:
    plt.axvline(x=b, color="gray", linestyle=":", alpha=0.6)

plt.xlabel("Epoch (cumulative)")
plt.ylabel("Sulc loss (raw)")
plt.title("Sulc loss train vs val, coarse-to-fine: 3 → 4 → 5 → 6")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sulc_train_val_concat.pdf")
plt.savefig("sulc_train_val_concat.png", dpi=300)
plt.close()
print("\nSalvato: sulc_train_val_concat.{pdf,png}")