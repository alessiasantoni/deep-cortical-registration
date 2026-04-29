
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

import brainsynth.resources
from brainnet.modules.graph import GenericSurfaceModule

# Internal cache

_TEMPLATE_SPHERE_CACHE: dict = {}  # (hemi, radius) -> Tensor (N_native, 3)
_ICO_VERTS_CACHE: dict = {}        # (order, hemi, radius) -> Tensor (N_ico, 3)


# 3D (x,y,z) → sferiche (theta, phi)
def cartesian_to_spherical(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert xyz sphere coordinates to (theta, phi)."""
    if vertices.ndim >= 2 and vertices.shape[-1] == 3:
        v = F.normalize(vertices.float(), dim=-1)
        x, y, z = v.unbind(dim=-1)
    elif vertices.ndim == 3 and vertices.shape[1] == 3:
        v = F.normalize(vertices.float(), dim=1)
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
    else:
        raise ValueError(f"Unsupported vertex shape: {tuple(vertices.shape)}")

    theta = torch.arccos(z.clamp(-1.0, 1.0))
    phi = torch.atan2(y, x)
    return theta, phi

# Sferiche (theta, phi) → coordinate griglia [-1,1]
def spherical_to_grid(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Map spherical coordinates to grid_sample coordinates in [-1, 1]."""
    grid_x = phi / torch.pi
    grid_y = 2.0 * (theta / torch.pi) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)

# Vertici 3D → griglia per grid_sample
def vertices_to_grid(vertices: torch.Tensor) -> torch.Tensor:
    """Convert sphere vertices to a grid_sample grid of shape (B, N, 1, 2)."""
    if vertices.ndim == 2 and vertices.shape[-1] == 3:
        vertices = vertices.unsqueeze(0)
    elif vertices.ndim == 2 and vertices.shape[0] == 3:
        vertices = vertices.T.unsqueeze(0)
    elif vertices.ndim == 3 and vertices.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unsupported vertex shape: {tuple(vertices.shape)}")

    theta, phi = cartesian_to_spherical(vertices)
    return spherical_to_grid(theta, phi).unsqueeze(2)

# Campiona la griglia 2D alle posizioni 3D dei vertici
def sample_spherical_grid(
    image: torch.Tensor,
    vertices: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
    #align_corners: bool = False,
) -> torch.Tensor:
    """Sample a spherical feature grid at 3D sphere locations."""
    if image.ndim != 4:
        raise ValueError(f"image must have shape (B, C, H, W), got {tuple(image.shape)}")

    grid = vertices_to_grid(vertices)
    batch = grid.shape[0]

    if image.shape[0] == 1 and batch > 1:
        image = image.expand(batch, -1, -1, -1)
    elif image.shape[0] != batch:
        raise ValueError(
            f"Batch mismatch between image {tuple(image.shape)} and vertices {tuple(grid.shape)}"
        )

    sampled = F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return sampled.squeeze(-1)


def _template_sphere(hemi: str, sphere_radius: float = 100.0) -> torch.Tensor:
    """Load and cache fsaverage template sphere.reg positions: (N_native, 3)."""
    key = (hemi, sphere_radius)
    if key not in _TEMPLATE_SPHERE_CACHE:
        tmpl = brainsynth.resources.Template()
        v = tmpl.load_surface(hemi, "sphere.reg")["vertices"]  # (N_native, 3)
        v = F.normalize(v.float(), dim=-1) * sphere_radius
        _TEMPLATE_SPHERE_CACHE[key] = v
    return _TEMPLATE_SPHERE_CACHE[key]


def _ico_verts(order: int, hemi: str, sphere_radius: float = 100.0) -> torch.Tensor:
    """Return and cache ico vertex positions (N_ico, 3) at the given order.

    The first N_k rows of the template sphere.reg give the ico mesh positions at
    order k because FsAverageTopology appends new vertices during subdivision.
    """
    key = (order, hemi, sphere_radius)
    if key not in _ICO_VERTS_CACHE:
        tmpl = _template_sphere(hemi, sphere_radius)
        #gsm = GenericSurfaceModule(0, order, order, topology="FsAverageTopology")
        gsm = GenericSurfaceModule(0, order, order, topology="DeepSurferTopology")
        n_k = gsm.topologies[order].n_vertices
        _ICO_VERTS_CACHE[key] = tmpl[:n_k]
    return _ICO_VERTS_CACHE[key]


# Resampling helpers

# Ricampiona dati dalla mesh nativa alla mesh ico
def resample_to_ico(
    data: torch.Tensor,           # (N_native,)
    sphere_reg: torch.Tensor,     # (N_native, 3)
    ico_positions: torch.Tensor,  # (N_ico, 3)
) -> torch.Tensor:
    """Resample a scalar field from native mesh to ico mesh via nearest neighbor.

    For each ico vertex at template position p_i, find the nearest native vertex
    in subject sphere.reg space and return data[v*].

    Returns
    -------
    Tensor of shape (N_ico,)
    """
    sr_np = F.normalize(sphere_reg.float(), dim=-1).cpu().numpy()
    ip_np = F.normalize(ico_positions.float(), dim=-1).cpu().numpy()

    tree = cKDTree(sr_np)
    _, idx = tree.query(ip_np, k=1, workers=-1)
    idx = torch.from_numpy(idx.astype(np.int64))

    return data[idx]


# Ricampiona la sphere.reg alla mesh ico
def resample_sphere_to_ico(
    sphere_reg: torch.Tensor,     # (N_native, 3)
    ico_positions: torch.Tensor,  # (N_ico, 3)
) -> torch.Tensor:
    """Resample subject sphere.reg positions onto the ico mesh.

    Returns
    -------
    Tensor of shape (3, N_ico)
    """
    sr_np = F.normalize(sphere_reg.float(), dim=-1).cpu().numpy()
    ip_np = F.normalize(ico_positions.float(), dim=-1).cpu().numpy()

    tree = cKDTree(sr_np)
    _, idx = tree.query(ip_np, k=1, workers=-1)
    idx = torch.from_numpy(idx.astype(np.int64))

    return sphere_reg[idx].T  # (3, N_ico)


# Dataset

class SphereRegDataset(Dataset):
    """Dataset for sulcal-depth spherical registration.

    Parameters
    ----------
    root_dir : str | Path
        Root directory, e.g. /projects/brainnet-data/sphere-reg
    datasets : list[str] | None
        Dataset names to include. If None, include all subdirectories.
    hemispheres : tuple[str, ...]
        Hemispheres to load, e.g. ("lh",) or ("lh", "rh")
    ico_order : int | None
        If int, resample to that ico order.
        If None, load native resolution directly (245762 vertices).
    sphere_radius : float
        Sphere radius used in the project.
    template_sulc : dict | None
        Optional precomputed template sulcal depth.
    """

    def __init__(
        self,
        root_dir: str | Path,
        datasets: list[str] | None = None,
        hemispheres: tuple[str, ...] = ("lh", "rh"),
        ico_order: int | None = 5,
        sphere_radius: float = 100.0,
        template_sulc: dict | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.hemispheres = list(hemispheres)
        self.ico_order = ico_order
        self.sphere_radius = sphere_radius
        self.template_sulc = template_sulc
        self._subject_grid_cache: dict[tuple[str, str, str], torch.Tensor] = {}

        if datasets is None:
            datasets = sorted(p.name for p in self.root_dir.iterdir() if p.is_dir())
        self.datasets = datasets

        self.subjects: list[tuple[str, str]] = []
        for ds in self.datasets:
            ds_dir = self.root_dir / ds
            if not ds_dir.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

            for subj_dir in sorted(ds_dir.iterdir()):
                if not subj_dir.is_dir():
                    continue

                ok = all(
                    (subj_dir / f"{h}.sulc.pt").exists()
                    and (subj_dir / f"{h}.sphere.reg").exists()
                    for h in self.hemispheres
                )
                if ok:
                    self.subjects.append((ds, subj_dir.name))

        if len(self.subjects) == 0:
            raise RuntimeError(
                f"No valid subjects found in {self.root_dir} for datasets {self.datasets}"
            )

        # Shared vertex positions
        if self.ico_order is None:
            # Native mode: full template sphere positions (245762, 3)
            self._ico_verts: dict[str, torch.Tensor] = {
                h: _template_sphere(h, sphere_radius)
                for h in self.hemispheres
            }
        else:
            self._ico_verts = {
                h: _ico_verts(self.ico_order, h, sphere_radius)
                for h in self.hemispheres
            }

    def __len__(self) -> int:
        return len(self.subjects)

    def __repr__(self) -> str:
        return (
            f"SphereRegDataset("
            f"n_subjects={len(self)}, "
            f"datasets={self.datasets}, "
            f"ico_order={self.ico_order}, "
            f"hemispheres={self.hemispheres})"
        )

    def _load_hemi(self, subj_dir: Path, hemi: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load subject sulc and subject sphere.reg for one hemisphere.

        Returns
        -------
        sulc       : (N_native,)
        sphere_reg : (N_native, 3)
        """
        sulc = torch.load(subj_dir / f"{hemi}.sulc.pt", weights_only=True).float()

        verts, _ = nib.freesurfer.read_geometry(str(subj_dir / f"{hemi}.sphere.reg"))
        sphere_reg = torch.tensor(verts.astype(np.float32))

        return sulc, sphere_reg

    def __getitem__(self, idx: int) -> dict:
        """Return one sample.

        Keys
        ----
        sulc_subj     : dict[hemi -> Tensor (1, N)]
        sphere_reg    : dict[hemi -> Tensor (3, N)]   <-- subject sphere support
        ico_verts     : dict[hemi -> Tensor (3, N)]   <-- template positions
        template_sulc : dict[hemi -> Tensor (1, N)] or None
        dataset       : str
        subject       : str

        N = N_ico if ico_order is set, else 245762 in native mode.
        """
        ds, subject = self.subjects[idx]
        subj_dir = self.root_dir / ds / subject

        sulc_subj: dict[str, torch.Tensor] = {}
        sulc_native: dict[str, torch.Tensor] = {}
        #sulc_grid: dict[str, torch.Tensor] = {}
        sphere_reg_out: dict[str, torch.Tensor] = {}

        for hemi in self.hemispheres:
            sulc_nat, sreg_nat = self._load_hemi(subj_dir, hemi)
            sulc_native[hemi] = sulc_nat.unsqueeze(0)

            if self.ico_order is None:
                # Native mode: do not resample anything
                sulc_subj[hemi] = sulc_nat.unsqueeze(0)   # (1, 245762)
                sphere_reg_out[hemi] = sreg_nat.T         # (3, 245762)
            else:
                ico_pos = self._ico_verts[hemi]           # (N_ico, 3)
                sulc_subj[hemi] = resample_to_ico(
                    sulc_nat, sreg_nat, ico_pos
                ).unsqueeze(0)                            # (1, N_ico)
                sphere_reg_out[hemi] = resample_sphere_to_ico(
                    sreg_nat, ico_pos
                )                                         # (3, N_ico)

            #cache_key = (ds, subject, hemi)
            #if cache_key not in self._subject_grid_cache:
             #   self._subject_grid_cache[cache_key] = rasterize_native_feature_to_grid(
              #      sulc_nat,
               #     _template_sphere(hemi, self.sphere_radius),
               # )
            #sulc_grid[hemi] = self._subject_grid_cache[cache_key]

        ico_verts = {h: self._ico_verts[h].T for h in self.hemispheres}  # (3, N)

        sample = dict(
            sulc_subj=sulc_subj,
            sulc_native=sulc_native,
            #sulc_grid=sulc_grid,
            sphere_reg=sphere_reg_out,
            ico_verts=ico_verts,
            dataset=ds,
            subject=subject,
        )
        if self.template_sulc is not None:
            sample["template_sulc"] = self.template_sulc
        return sample


# Carica template come griglia 2D (1, H, W)
def load_feature_grid_images(
    grid_dir: str | Path,
    hemispheres: tuple[str, ...] = ("lh", "rh"),
    feature: str = "sulc",
) -> dict[str, torch.Tensor]:
    """Load fsaverage spherical feature grids as (1, H, W) images."""
    grid_dir = Path(grid_dir)
    result: dict[str, torch.Tensor] = {}

    for hemi in hemispheres:
        npz = np.load(grid_dir / f"{hemi}.white.features.npz")
        result[hemi] = torch.tensor(npz[feature], dtype=torch.float32).unsqueeze(0)

    return result

