"""
Dataset per spherical registration. 
Modifica: carica anche la WM surface del soggetto (lh.white) per usarla come 
riferimento nella metric distortion loss (suggerimento del prof).
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SphereRegDataset(Dataset):
    """Dataset per registrazione sferica.
    
    Per ogni soggetto restituisce:
      - sulc_subj: (1, N) sulcal depth sui vertici della mesh
      - sphere_reg: (3, N) sfera già registrata da FreeSurfer (lh.sphere.reg)
      - wm_verts: (3, N) vertici della white matter surface
      - subject, dataset: nomi (per identificazione)
    """

    def __init__(self, root_dir: str, datasets: list[str], hemi: str = "lh"):
        self.root = Path(root_dir)
        self.datasets = datasets
        self.hemi = hemi

        self.subjects = []
        for ds in datasets:
            ds_dir = self.root / ds
            if not ds_dir.exists():
                continue
            for sub_dir in sorted(ds_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue
                sulc_path = sub_dir / f"{hemi}.sulc.pt"
                sphere_path = sub_dir / f"{hemi}.sphere.reg"
                wm_path = sub_dir / f"{hemi}.white"
                if sulc_path.exists() and sphere_path.exists() and wm_path.exists():
                    self.subjects.append((ds, sub_dir.name))

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        dataset, subject = self.subjects[idx]
        sub_dir = self.root / dataset / subject

        # Sulc del soggetto (caricata come tensor.pt)
        sulc = torch.load(sub_dir / f"{self.hemi}.sulc.pt", weights_only=True).float()
        sulc = sulc.unsqueeze(0)  # (1, N)

        # Sfera registrata da FreeSurfer
        sphere_verts, _ = nib.freesurfer.read_geometry(
            str(sub_dir / f"{self.hemi}.sphere.reg")
        )
        sphere_verts = torch.from_numpy(sphere_verts).float().T  # (3, N)

        # WM surface del soggetto (per metric loss)
        wm_verts, _ = nib.freesurfer.read_geometry(
            str(sub_dir / f"{self.hemi}.white")
        )
        wm_verts = torch.from_numpy(wm_verts).float().T  # (3, N)

        return {
            "sulc_subj": sulc,
            "sphere_reg": sphere_verts,
            "wm_verts": wm_verts,
            "subject": subject,
            "dataset": dataset,
        }

    def __repr__(self):
        return (
            f"SphereRegDataset(n={len(self.subjects)}, "
            f"datasets={self.datasets}, hemi={self.hemi})"
        )


# ============== Helper: caricare la griglia template ==============

def load_template_grid(grid_dir: str, hemi: str) -> torch.Tensor:
    """Carica la griglia 2D del template (sulc) come tensor (1, H, W)."""
    grid_path = Path(grid_dir) / f"{hemi}.white.features.npz"
    data = np.load(grid_path)
    sulc = torch.tensor(data["sulc"], dtype=torch.float32).unsqueeze(0)  # (1, H, W)
    return sulc


# ============== Conversione cartesiane -> coordinate griglia ==============

def cartesian_to_grid_coords(vertices: torch.Tensor) -> torch.Tensor:
    """Converte vertici 3D in coordinate (x, y) per F.grid_sample, in [-1, 1]."""
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)

    v_unit = F.normalize(vertices.float(), dim=1)
    x, y, z = v_unit[:, 0], v_unit[:, 1], v_unit[:, 2]

    # Clamp stretto per evitare derivate infinite di arccos ai poli
    theta = torch.arccos(z.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    phi = torch.arctan2(y, x)

    grid_x = phi / torch.pi
    grid_y = (theta / torch.pi) * 2.0 - 1.0

    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)


def sample_grid_at_vertices(grid: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Campiona la griglia 2D alle posizioni dei vertici.
    
    grid: (B, C, H, W) o (1, C, H, W)
    vertices: (B, 3, N)
    
    Returns: (B, C, N)
    """
    B = vertices.shape[0]
    if grid.ndim == 3:
        grid = grid.unsqueeze(0)
    if grid.shape[0] == 1 and B > 1:
        grid = grid.expand(B, -1, -1, -1)

    coords = cartesian_to_grid_coords(vertices)  # (B, N, 1, 2)

    sampled = F.grid_sample(
        grid, coords,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (B, C, N, 1)

    return sampled.squeeze(-1)