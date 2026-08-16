""" To compute the DICE for a specific region or more"""

from pathlib import Path
import pandas as pd
import numpy as np

# ================= CONFIG =================

OUTPUT_DIR = Path(
    "experiments/sphere_reg2/eval_parcellation"
)

MINE_CSV = OUTPUT_DIR / "dice_mine_vs_fs.csv"
SUP_CSV = OUTPUT_DIR / "dice_supervisor_vs_fs.csv"

# Inserisci i nomi esattamente come compaiono nelle colonne dei CSV
SELECTED_REGIONS = [
    "postcentral"
]

# ==========================================

mine_df = pd.read_csv(MINE_CSV)
sup_df = pd.read_csv(SUP_CSV)

# Controllo regioni disponibili
missing = [
    region
    for region in SELECTED_REGIONS
    if region not in mine_df.columns or region not in sup_df.columns
]

if missing:
    raise ValueError(
        f"Regioni non trovate: {missing}\n"
        f"Regioni disponibili:\n"
        + "\n".join(mine_df.columns[2:-1])
    )

# Dice medio considerando solamente le regioni selezionate
mine_df["selected_mean_dice"] = mine_df[SELECTED_REGIONS].mean(
    axis=1,
    skipna=True
)

sup_df["selected_mean_dice"] = sup_df[SELECTED_REGIONS].mean(
    axis=1,
    skipna=True
)

# Tabella con risultati per soggetto
results = mine_df[["dataset", "subject"]].copy()

results["mine_selected_mean"] = mine_df["selected_mean_dice"]
results["supervisor_selected_mean"] = sup_df["selected_mean_dice"]

results["difference"] = (
    results["mine_selected_mean"]
    - results["supervisor_selected_mean"]
)

print("\n=== DICE MEDIO SULLE REGIONI SELEZIONATE ===\n")

for _, row in results.iterrows():
    print(
        f"{row['dataset']}/{row['subject']} | "
        f"Mia: {row['mine_selected_mean']:.4f} | "
        f"Supervisor: {row['supervisor_selected_mean']:.4f} | "
        f"Diff: {row['difference']:+.4f}"
    )

# Media complessiva su tutti i soggetti
mine_global = results["mine_selected_mean"].mean()
sup_global = results["supervisor_selected_mean"].mean()

print("\n=== RISULTATO GLOBALE ===")
print(f"Mia        : {mine_global:.4f}")
print(f"Supervisor : {sup_global:.4f}")
print(f"Differenza : {mine_global - sup_global:+.4f}")

# Salvataggio
results.to_csv(
    OUTPUT_DIR / "dice_selected_regions.csv",
    index=False
)

print(
    f"\nSalvato in: "
    f"{OUTPUT_DIR / 'dice_selected_regions.csv'}"
)