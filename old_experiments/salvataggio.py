import numpy as np

sulc_file = "/projects/brainnet-data/sphere-reg/Buckner40/sub-001/lh.sulc"

# Carica il file come array di numeri binari
sulc_data = np.fromfile(sulc_file, dtype=np.float32)

print(f"Shape dei dati di lh.sulc: {sulc_data.shape}")