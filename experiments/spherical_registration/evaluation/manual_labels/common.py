"""
Things shared by the Mindboggle-101 scripts in this folder.

--data_root must contain one folder per subject:
  <sub-id>/lh.white
  <sub-id>/lh.labels.DKT31.manual.annot
  <sub-id>/derivatives/lh.sulc
  <sub-id>/derivatives/lh.sphere.reg.freesurfer
  <sub-id>/derivatives/lh.sphere.reg.josa

--id_map has one line per subject, "<Mindboggle name> <sub-id>"
(e.g. "NKI-TRT-20-8 sub-080"); the Mindboggle name is only needed to drop EXCLUDE.
"""

from experiments.spherical_registration.evaluation.atlas import add_fsaverage_argument, fsaverage_files  # noqa: F401

HEMI = "lh"
# not real subjects (an average template and Colin27), left out
EXCLUDE = ("Afterthought", "Colin27")


def add_data_arguments(parser):
    parser.add_argument("--data_root", required=True, help="folder with one sub-* folder per Mindboggle-101 subject")
    parser.add_argument("--id_map", required=True, help='text file with lines "<Mindboggle name> <sub-id>"')


def load_subjects(id_map):
    """sub-* ids of the subjects to evaluate."""
    subjects = []
    with open(id_map) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and not parts[0].startswith(EXCLUDE):
                subjects.append(parts[1])
    return subjects
