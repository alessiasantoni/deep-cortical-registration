from pathlib import Path
import nibabel as nib
import numpy as np

#INPUT_DIR = Path("/home/s252125/my_project/experiments/sphere_reg2/runs/spheres/epoch_030")
INPUT_DIR = Path("/projects/brainnet-data/sphere-reg/ABIDE/sub-0043/lh.sphere")
OUTPUT_DIR = Path("/home/s252125/my_project/experiments/sphere_reg2/runs/try")
OUTPUT_DIR.mkdir(exist_ok=True)

SUBJECT = "sub-0043"

# path al sulc del soggetto, adattalo al tuo dataset
SULC_PATH = Path("/projects/brainnet-data/sphere-reg/ABIDE/sub-0043/lh.sulc")


def write_vtk_with_sulc(path_out, verts, faces, sulc):
    sulc = np.asarray(sulc).reshape(-1)

    assert len(verts) == len(sulc), f"verts={len(verts)}, sulc={len(sulc)}"

    with open(path_out, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("registered sphere with sulc\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write(f"POINTS {len(verts)} float\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        f.write(f"POLYGONS {len(faces)} {len(faces) * 4}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        f.write(f"\nPOINT_DATA {len(verts)}\n")
        f.write("SCALARS sulc float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for s in sulc:
            f.write(f"{float(s)}\n")


sulc = nib.freesurfer.read_morph_data(str(SULC_PATH))

for sphere_file in INPUT_DIR.glob(f"*{SUBJECT}*sphere.reg"):
    verts, faces = nib.freesurfer.read_geometry(str(sphere_file))

    out_path = OUTPUT_DIR / f"{sphere_file.name}.vtk"
    write_vtk_with_sulc(out_path, verts, faces, sulc)
    print("Saved:", out_path)

"""
scp s252125@health-pcalc2:/home/s252125/my_project/experiments/sphere_reg2/runs/spheres/epoch_010/vtk/*sub-0043* "C:/spheres_sulc_v2/"
scp s252125@health-pcalc2:/home/s252125/my_project/experiments/sphere_reg2/runs/spheres/epoch_020/vtk/*sub-0043* "C:/spheres_sulc_v2/"
scp s252125@health-pcalc2:/home/s252125/my_project/experiments/sphere_reg2/runs/spheres/epoch_030/vtk/*sub-0043* "C:/spheres_sulc_v2/"
"""