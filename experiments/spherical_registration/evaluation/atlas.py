"""
The fsaverage aparc atlas used by the evaluation scripts: FreeSurfer's own annotation
on its own sphere (surf/lh.sphere.reg, 163842 vertices). The scripts move it onto a
registered sphere by nearest neighbour.

fsaverage is taken from $FREESURFER_HOME/subjects/fsaverage unless --fsaverage_dir is given.
"""

import os
from pathlib import Path

import nibabel as nib


def add_fsaverage_argument(parser):
    default = Path(os.environ.get("FREESURFER_HOME", "")) / "subjects" / "fsaverage"
    parser.add_argument("--fsaverage_dir", default=str(default),
                        help="fsaverage subject folder (default: $FREESURFER_HOME/subjects/fsaverage)")


def fsaverage_files(fsaverage_dir, hemi="lh"):
    """Paths of the fsaverage sphere and aparc annotation."""
    fsaverage_dir = Path(fsaverage_dir)
    return fsaverage_dir / "surf" / f"{hemi}.sphere.reg", fsaverage_dir / "label" / f"{hemi}.aparc.annot"


def load_fsaverage_atlas(fsaverage_dir, hemi="lh"):
    """Returns sphere vertices, one label per vertex and the region names."""
    sphere_path, annot_path = fsaverage_files(fsaverage_dir, hemi)
    if not sphere_path.exists() or not annot_path.exists():
        raise FileNotFoundError(f"fsaverage atlas not found in {fsaverage_dir}: "
                                "set $FREESURFER_HOME or pass --fsaverage_dir")
    sphere, _ = nib.freesurfer.read_geometry(str(sphere_path))
    labels, _, names = nib.freesurfer.read_annot(str(annot_path))
    assert sphere.shape[0] == len(labels), "fsaverage sphere and annotation do not match"
    return sphere, labels, names
