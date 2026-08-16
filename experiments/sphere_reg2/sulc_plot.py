
import csv
from pathlib import Path
import matplotlib.pyplot as plt

RUNS_BASE = Path("experiments/sphere_reg2/runs")

# ORDINE conta: vengono concatenate in questa sequenza.
RUNS = [
    {"label": "ico=3", "dir": "20260610_081353_train_ico3", "color": "C3"},
    {"label": "ico=4", "dir": "20260609_133153_finetune_ico4", "color": "C0"},
    {"label": "ico=5", "dir": "20260609_160419_finetune_ico5", "color": "C1"},
    {"label": "ico=6", "dir": "20260609_181249_finetune_ico6", "color": "C2"},
]


def load_sulc_from_subject_files(run_dir: Path):
    """Media la colonna 'sulc' (gia' RAW, non pesata) sui soggetti, per ogni epoca."""
    epochs, sulc_means = [], []
    for csv_file in run_dir.glob("val_subject_losses_epoch_*.csv"):
        ep = int(csv_file.stem.split("_")[-1])
        vals = [float(row["sulc"]) for row in csv.DictReader(open(csv_file, newline=""))]
        if vals:
            epochs.append(ep)
            sulc_means.append(sum(vals) / len(vals))
    order = sorted(range(len(epochs)), key=lambda i: epochs[i])
    return [epochs[i] for i in order], [sulc_means[i] for i in order]


plt.figure(figsize=(12, 6))

offset = 0                    # da dove parte (sull'asse x) il livello corrente
boundaries = []               # dove finisce ogni fase, per le linee divisorie
last_y = None                 # ultimo valore sulc, per collegare le fasi

for info in RUNS:
    run_dir = RUNS_BASE / info["dir"]
    ep, sulc = load_sulc_from_subject_files(run_dir)
    if not ep:
        print(f"  [skip] {info['label']}: nessun file val_subject in {run_dir}")
        continue

    # Shift dell'asse x: questo livello parte dopo la fine del precedente.
    # Rinormalizzo le epoche a partire da 1, poi aggiungo l'offset.
    ep_shifted = [(e - ep[0]) + offset + 1 for e in ep]

    # Collega la prima x di questa fase all'ultima dell'altra (continuita' visiva).
    xs, ys = ep_shifted, sulc
    if last_y is not None:
        xs = [offset] + ep_shifted        # punto di giunzione
        ys = [last_y] + sulc

    plt.plot(xs, ys, color=info["color"], marker="o", markersize=4, label=info["label"])
    print(f"  {info['label']}: sulc raw {sulc[0]:.4f} -> {sulc[-1]:.4f} "
          f"(asse x: {ep_shifted[0]}..{ep_shifted[-1]})")

    offset = ep_shifted[-1]               # il prossimo parte da qui
    boundaries.append(offset)
    last_y = sulc[-1]

# Linee verticali tratteggiate ai confini tra fasi
for b in boundaries[:-1]:
    plt.axvline(x=b, color="gray", linestyle=":", alpha=0.6)

plt.xlabel("Epoch (cumulative)")
plt.ylabel("Sulc loss (raw)")
plt.title("Sulc loss validation coarse-to-fine: 3 → 4 → 5 → 6")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sulc_concatenated.png", dpi=600)
plt.close()
print("\nSalvato: sulc_concatenated.png")