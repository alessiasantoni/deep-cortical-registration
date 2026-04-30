"""
Dataset semplificato per la spherical registration.

Carica:
  - sulc_subj  (1, N) sulla mesh nativa del soggetto
  - sphere_reg (3, N) posizioni 3D della sphere.reg del soggetto
  - template_grid (1, H, W) griglia 2D della sulc del template (caricata una volta)
"""



from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_template_grid(grid_dir: Path | str, hemi: str = "lh") -> torch.Tensor:
    """Carica la griglia 2D del template (sulc) come tensore (1, H, W)."""
    grid_dir = Path(grid_dir)
    npz = np.load(grid_dir / f"{hemi}.white.features.npz")
    return torch.tensor(npz["sulc"], dtype=torch.float32).unsqueeze(0)  # (1, H, W)


def cartesian_to_grid_coords(vertices: torch.Tensor) -> torch.Tensor:
    """Converte vertici 3D in coordinate (x, y) per F.grid_sample, in [-1, 1].

    Parameters
    ----------
    vertices : (B, 3, N) o (3, N)

    Returns
    -------
    grid : (B, N, 1, 2) pronto per F.grid_sample
    """
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)  # (1, 3, N)

    v_unit = F.normalize(vertices.float(), dim=1)
    x, y, z = v_unit[:, 0], v_unit[:, 1], v_unit[:, 2]

    theta = torch.arccos(z.clamp(-1.0, 1.0))   # [0, pi]
    phi = torch.arctan2(y, x)                  # [-pi, pi]

    grid_x = phi / torch.pi                    # [-1, 1]
    grid_y = (theta / torch.pi) * 2.0 - 1.0    # [-1, 1]

    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, N, 1, 2)


def sample_grid_at_vertices(
    grid: torch.Tensor,        # (B, C, H, W) oppure (C, H, W)
    vertices: torch.Tensor,    # (B, 3, N)
    align_corners: bool = True,
) -> torch.Tensor:
    """Campiona la griglia 2D alle posizioni 3D dei vertici.

    Returns
    -------
    sampled : (B, C, N)
    """
    if grid.ndim == 3:
        grid = grid.unsqueeze(0)
    if grid.shape[0] == 1 and vertices.shape[0] > 1:
        grid = grid.expand(vertices.shape[0], -1, -1, -1)

    coords = cartesian_to_grid_coords(vertices)  # (B, N, 1, 2)
    sampled = F.grid_sample(
        grid, coords,
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners,
    )
    return sampled.squeeze(-1)  # (B, C, N)


class SphereRegDataset(Dataset):
    """Dataset minimale per la registrazione sferica.

    Per ogni soggetto restituisce:
        sulc_subj   : (1, N) valori sulc sulla mesh nativa
        sphere_reg  : (3, N) posizioni 3D dei vertici sulla sfera del soggetto
        subject     : nome del soggetto
        dataset     : nome del dataset
    """

    def __init__(
        self,
        root_dir: str | Path,
        datasets: list[str],
        hemi: str = "lh",
    ):
        self.root_dir = Path(root_dir)
        self.hemi = hemi
        self.datasets = datasets

        self.subjects: list[tuple[str, str]] = []
        for ds in self.datasets:
            ds_dir = self.root_dir / ds
            if not ds_dir.is_dir():
                raise FileNotFoundError(f"Missing dataset dir: {ds_dir}")
            for subj_dir in sorted(ds_dir.iterdir()):
                if not subj_dir.is_dir():
                    continue
                ok = (
                    (subj_dir / f"{hemi}.sulc.pt").exists()
                    and (subj_dir / f"{hemi}.sphere.reg").exists()
                )
                if ok:
                    self.subjects.append((ds, subj_dir.name))

        if not self.subjects:
            raise RuntimeError(f"No valid subjects in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.subjects)

    def __repr__(self) -> str:
        return (
            f"SphereRegDataset(n={len(self)}, datasets={self.datasets}, hemi={self.hemi})"
        )

    def __getitem__(self, idx: int) -> dict:
        ds, sub = self.subjects[idx]
        subj_dir = self.root_dir / ds / sub

        sulc = torch.load(
            subj_dir / f"{self.hemi}.sulc.pt", weights_only=True
        ).float().unsqueeze(0)  # (1, N)

        verts, _ = nib.freesurfer.read_geometry(
            #str(subj_dir / f"{self.hemi}.sphere.reg")
            str(subj_dir / f"{self.hemi}.sphere")
        )
        sphere_reg = torch.tensor(verts.astype(np.float32)).T  # (3, N)

        return dict(
            sulc_subj=sulc,
            sphere_reg=sphere_reg,
            subject=sub,
            dataset=ds,
        )