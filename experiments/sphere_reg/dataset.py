"""Dataset for sulcal-depth-based spherical registration.

Data layout (HPC)
-----------------
/projects/brainnet-data/sphere-reg/{dataset}/{subject}/
    lh.sulc.pt          float32 tensor  (N_native,)         sulcal depth
    lh.sphere.reg       FreeSurfer geometry (N_native, 3)   registered sphere
    rh.sulc.pt
    rh.sphere.reg

Each subject's lh.sphere.reg maps native-mesh vertices to their positions on the
fsaverage template sphere (245 762 vertices, same for all subjects).  The first
N_k vertices of the template sphere.reg correspond to the FsAverageTopology
icosahedral mesh at order k (e.g. N_5 = 10 242, N_7 = 163 842).  We exploit
this structure to resample both sulcal depth and sphere.reg positions from the
native mesh onto the ico mesh via nearest-neighbour lookup.

What the dataset returns
------------------------
sulc_subj     : dict[hemi -> Tensor (1, N_ico)]   subject sulcal depth on ico mesh
sphere_reg    : dict[hemi -> Tensor (3, N_ico)]   subject sphere.reg on ico mesh
ico_verts     : dict[hemi -> Tensor (3, N_ico)]   template ico vertex positions
                                                   (same for every subject)
template_sulc : dict[hemi -> Tensor (1, N_ico)]   template (group-mean) sulcal depth
"""

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


# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------

_TEMPLATE_SPHERE_CACHE: dict = {}  # (hemi, radius) -> Tensor (N_native, 3)
_ICO_VERTS_CACHE: dict = {}        # (order, hemi, radius) -> Tensor (N_ico, 3)


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


def spherical_to_grid(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Map spherical coordinates to grid_sample coordinates in [-1, 1]."""
    grid_x = phi / torch.pi
    grid_y = 2.0 * (theta / torch.pi) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


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


def sample_spherical_grid(
    image: torch.Tensor,
    vertices: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
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


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Template sulcal depth from spherical grid
# ---------------------------------------------------------------------------

def load_template_sulc_from_grid(
    grid_dir: str | Path,
    hemispheres: tuple[str, ...] = ("lh", "rh"),
    ico_order: int | None = 5,
    sphere_radius: float = 100.0,
) -> dict[str, torch.Tensor]:
    """Load template sulcal depth from a precomputed spherical grid.

    Expects files:
        {hemi}.white.features.npz
    with array:
        sulc : (512, 1024)
    """
    grid_dir = Path(grid_dir)
    result: dict[str, torch.Tensor] = {}

    for hemi in hemispheres:
        npz = np.load(grid_dir / f"{hemi}.white.features.npz")
        sulc_grid = torch.tensor(npz["sulc"], dtype=torch.float32)  # (512, 1024)

        if ico_order is None:
            verts = _template_sphere(hemi, sphere_radius)
        else:
            verts = _ico_verts(ico_order, hemi, sphere_radius)

        v_unit = F.normalize(verts.float(), dim=-1)
        theta = torch.arccos(v_unit[:, 2].clamp(-1, 1))
        phi = torch.arctan2(v_unit[:, 1], v_unit[:, 0])

        grid_x = phi / torch.pi
        grid_y = (theta / torch.pi) * 2.0 - 1.0

        sampling_grid = (
            torch.stack([grid_x, grid_y], dim=-1)
            .unsqueeze(0)
            .unsqueeze(2)
        )  # (1, N, 1, 2)

        sampled = F.grid_sample(
            sulc_grid.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (1, 1, N, 1)

        result[hemi] = sampled.squeeze(0).squeeze(-1)  # (1, N)

    return result


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


def rasterize_native_feature_to_grid(
    values: torch.Tensor,
    template_vertices: torch.Tensor,
    *,
    height: int = 512,
    width: int = 1024,
) -> torch.Tensor:
    """Interpolate native per-vertex values onto a dense regular spherical grid."""
    if values.ndim != 1:
        raise ValueError(f"values must have shape (N,), got {tuple(values.shape)}")
    if template_vertices.ndim != 2 or template_vertices.shape[1] != 3:
        raise ValueError(
            f"template_vertices must have shape (N, 3), got {tuple(template_vertices.shape)}"
        )
    if values.shape[0] != template_vertices.shape[0]:
        raise ValueError("values/template_vertices length mismatch")

    theta, phi = cartesian_to_spherical(template_vertices)
    theta_np = theta.cpu().numpy()
    phi_np = phi.cpu().numpy()
    values_np = values.float().cpu().numpy()

    src = np.stack([phi_np, theta_np], axis=1)
    src_wrap = np.concatenate(
        [
            src,
            np.stack([phi_np - 2.0 * np.pi, theta_np], axis=1),
            np.stack([phi_np + 2.0 * np.pi, theta_np], axis=1),
        ],
        axis=0,
    )
    values_wrap = np.tile(values_np, 3)

    phi_lin = np.linspace(-np.pi, np.pi, width, dtype=np.float32)
    theta_lin = np.linspace(0.0, np.pi, height, dtype=np.float32)
    grid_phi, grid_theta = np.meshgrid(phi_lin, theta_lin)
    target = np.stack([grid_phi.reshape(-1), grid_theta.reshape(-1)], axis=1)

    interp = griddata(src_wrap, values_wrap, target, method="linear", fill_value=np.nan)
    if np.isnan(interp).any():
        nearest = griddata(src_wrap, values_wrap, target, method="nearest")
        interp = np.where(np.isnan(interp), nearest, interp)

    return torch.from_numpy(interp.reshape(height, width).astype(np.float32)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Template sulcal depth as group mean
# ---------------------------------------------------------------------------

def compute_template_sulc(
    root_dir: str | Path,
    datasets: list[str],
    hemispheres: tuple[str, ...] = ("lh", "rh"),
    ico_order: int | None = 5,
    sphere_radius: float = 100.0,
    max_subjects: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute group-mean sulcal depth as template proxy.

    If ico_order is None, compute directly at native resolution.
    """
    root_dir = Path(root_dir)

    if ico_order is not None:
        ico_verts_map = {
            h: _ico_verts(ico_order, h, sphere_radius)
            for h in hemispheres
        }
    else:
        ico_verts_map = None

    accum: dict[str, list[torch.Tensor]] = {h: [] for h in hemispheres}
    count = 0

    for ds in datasets:
        ds_dir = root_dir / ds
        if not ds_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

        for subj_dir in sorted(ds_dir.iterdir()):
            if max_subjects is not None and count >= max_subjects:
                break
            if not subj_dir.is_dir():
                continue

            ok = all(
                (subj_dir / f"{h}.sulc.pt").exists()
                and (subj_dir / f"{h}.sphere.reg").exists()
                for h in hemispheres
            )
            if not ok:
                continue

            for hemi in hemispheres:
                sulc_nat = torch.load(
                    subj_dir / f"{hemi}.sulc.pt",
                    weights_only=True,
                ).float()

                if ico_order is None:
                    sulc = sulc_nat
                else:
                    verts, _ = nib.freesurfer.read_geometry(str(subj_dir / f"{hemi}.sphere.reg"))
                    sreg_nat = torch.tensor(verts.astype(np.float32))
                    sulc = resample_to_ico(sulc_nat, sreg_nat, ico_verts_map[hemi])

                accum[hemi].append(sulc)

            count += 1

    if count == 0:
        raise RuntimeError("No subjects were loaded. Check root_dir and dataset names.")

    print(f"Template sulcal depth computed from {count} subjects.")
    return {
        hemi: torch.stack(accum[hemi], dim=0).mean(dim=0).unsqueeze(0)
        for hemi in hemispheres
    }
