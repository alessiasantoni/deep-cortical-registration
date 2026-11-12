"""
aparc (Desikan-Killiany) names -> the two manual protocols of Mindboggle-101
(Klein & Tourville, 2012).

DKT31: aparc without bankssts, frontalpole and temporalpole, whose cortex went to
       the neighbouring regions. The manual DKT31 annots use the aparc names.
DKT25: DKT31 with cingulate, middle frontal and inferior frontal merged (see
       DKT31_TO_DKT25). Built from DKT31 here: the color table of the DKT25
       manual annots does not match their names, 6 regions end up with the same value.

Vertices outside the protocol (medial wall, corpus callosum, unlabeled) get "".
"""

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

REMOVED_IN_DKT = ("bankssts", "frontalpole", "temporalpole")

DKT31_REGIONS = (
    "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal",
    "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate",
    "lateraloccipital", "lateralorbitofrontal", "lingual", "medialorbitofrontal",
    "middletemporal", "parahippocampal", "paracentral", "parsopercularis",
    "parsorbitalis", "parstriangularis", "pericalcarine", "postcentral",
    "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate",
    "rostralmiddlefrontal", "superiorfrontal", "superiorparietal",
    "superiortemporal", "supramarginal", "transversetemporal", "insula",
)

DKT31_TO_DKT25 = {
    "caudalanteriorcingulate": "cingulate", "rostralanteriorcingulate": "cingulate",
    "posteriorcingulate": "cingulate", "isthmuscingulate": "cingulate",
    "caudalmiddlefrontal": "middlefrontal", "rostralmiddlefrontal": "middlefrontal",
    "parsopercularis": "inferiorfrontal", "parsorbitalis": "inferiorfrontal",
    "parstriangularis": "inferiorfrontal",
}

DKT25_REGIONS = tuple(sorted({DKT31_TO_DKT25.get(r, r) for r in DKT31_REGIONS}))
assert len(DKT31_REGIONS) == 31 and len(DKT25_REGIONS) == 25


def read_annot_names(path):
    """Region name of every vertex of an annot, "" where there is no label."""
    labels, _, names = nib.freesurfer.read_annot(str(path))
    names = np.array([n.decode() for n in names] + [""])
    return names[np.where(labels >= 0, labels, len(names) - 1)]


def to_dkt31(names):
    """Keeps the 31 DKT31 regions, everything else becomes ""."""
    return np.where(np.isin(names, DKT31_REGIONS), names, "")


def to_dkt25(names31):
    """DKT31 names -> DKT25 names."""
    out = names31.copy()
    for src, dst in DKT31_TO_DKT25.items():
        out[names31 == src] = dst
    return out


def aparc_atlas_to_dkt31(sphere, names):
    """aparc atlas -> DKT31. The vertices of bankssts, frontalpole and temporalpole
    take the region of the closest kept vertex, like the manual protocol does with
    the neighbouring regions. Medial wall and corpus callosum stay ""."""
    names = names.copy()
    removed = np.isin(names, REMOVED_IN_DKT)
    kept = np.isin(names, DKT31_REGIONS)
    unit = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    _, nn = cKDTree(unit[kept]).query(unit[removed])
    names[removed] = names[kept][nn]
    return to_dkt31(names)
