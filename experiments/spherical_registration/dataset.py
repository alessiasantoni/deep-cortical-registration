"""
Dataset and template-grid helpers.

Each item has the subject sulc, the FreeSurfer sphere and the WM surface
(lh.white), which is the reference for the metric term of the loss.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SphereRegDataset(Dataset):
    def __init__(self, root_dir, datasets, hemi="lh",
                 split_dir=None, split="train", bad_surfaces_file=None):
        self.root = Path(root_dir)
        self.datasets = datasets
        self.hemi = hemi

        # subjects with broken surfaces, {dataset: [sub-xxxx, ...]}
        bad = {}
        if bad_surfaces_file is not None:
            with open(bad_surfaces_file) as f:
                bad = json.load(f)

        self.subjects = []
        for ds in datasets:
            split_file = Path(split_dir) / f"{ds}.{split}.txt"
            if not split_file.exists():
                print(f"  WARNING: no split file {split_file}, skipping {ds}")
                continue

            with open(split_file) as f:
                subject_ids = [line.strip() for line in f if line.strip()]

            bad_set = set(bad.get(ds, []))
            n_before = len(subject_ids)
            subject_ids = [s for s in subject_ids if s not in bad_set]
            n_excluded = n_before - len(subject_ids)

            n_kept = 0
            n_missing = 0
            for sub in subject_ids:
                sub_dir = self.root / ds / sub
                sulc_path   = sub_dir / "derivatives" / f"{hemi}.sulc.pt"
                sphere_path = sub_dir / f"{hemi}.sphere.reg"
                wm_path     = sub_dir / f"{hemi}.white"
                if sulc_path.exists() and sphere_path.exists() and wm_path.exists():
                    self.subjects.append((ds, sub))
                    n_kept += 1
                else:
                    n_missing += 1

            print(f"  {ds}.{split}: {n_kept} kept, {n_excluded} bad, {n_missing} missing")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        dataset, subject = self.subjects[idx]
        sub_dir = self.root / dataset / subject

        sulc = torch.load(sub_dir / "derivatives" / f"{self.hemi}.sulc.pt", weights_only=True).float()
        sulc = sulc.unsqueeze(0)  # (1, N)

        sphere_verts, _ = nib.freesurfer.read_geometry(
            str(sub_dir / f"{self.hemi}.sphere.reg")
        )
        sphere_verts = torch.from_numpy(sphere_verts).float().T  # (3, N)

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


def load_template_grid(grid_dir: str, hemi: str) -> torch.Tensor:
    """Template sulc on the lat-long grid, (1, H, W)."""
    grid_path = Path(grid_dir) / f"{hemi}.white.features.npz"
    data = np.load(grid_path)
    sulc = torch.tensor(data["sulc"], dtype=torch.float32).unsqueeze(0)
    return sulc


def cartesian_to_grid_coords(vertices: torch.Tensor) -> torch.Tensor:
    """Vertices (B, 3, N) -> (B, N, 1, 2) coordinates in [-1, 1] for F.grid_sample
    (x = longitude, y = colatitude)."""
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)

    v_unit = F.normalize(vertices.float(), dim=1)
    x, y, z = v_unit[:, 0], v_unit[:, 1], v_unit[:, 2]

    # keep away from the poles, the derivative of arccos blows up at +-1
    theta = torch.arccos(z.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    phi = torch.arctan2(y, x)

    grid_x = phi / torch.pi
    grid_y = (theta / torch.pi) * 2.0 - 1.0

    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)


def sample_grid_at_vertices(grid: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Bilinear sampling of grid (B or 1, C, H, W) at vertices (B, 3, N) -> (B, C, N)."""
    B = vertices.shape[0]
    if grid.ndim == 3:
        grid = grid.unsqueeze(0)
    if grid.shape[0] == 1 and B > 1:
        grid = grid.expand(B, -1, -1, -1)

    coords = cartesian_to_grid_coords(vertices)

    sampled = F.grid_sample(
        grid, coords,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return sampled.squeeze(-1)
