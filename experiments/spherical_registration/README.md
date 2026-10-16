# Spherical registration

Coarse-to-fine deep-learning registration of cortical surfaces onto a template sphere (fsaverage), driven by sulcal depth (sulc). For each subject the network deforms an initial sphere so that the subject's sulc matches the template sulc, while keeping the mesh well-behaved (no folded triangles, metric close to the white-matter surface). Results are compared against FreeSurfer's `sphere.reg`.

## How it works

- **Input per subject:** `sulc` (`derivatives/{hemi}.sulc.pt`), the white-matter surface (`{hemi}.white`, used only in the loss), and the template sulc rasterized on a 2D grid (`fsaverage_grid/`).
- **Model** ([network.py](network.py), `SphereRegNet`): one U-Net (edge convolutions on icosahedral meshes) per resolution level, from `ico_in` to `ico_out`. At each level the vertices are moved for a fixed number of iterations (`n_iter`): the template sulc is sampled at the current positions, concatenated with the subject sulc, and the U-Net predicts a displacement. The displacement is applied as a rotation (Euler step), so points stay on the sphere without a separate projection.
- **Loss** ([loss.py](loss.py), `SulcalAlignmentLoss`): sulc alignment with the template + metric preservation against the WM surface + folded-area penalty + a differentiable folded-triangle count (`sulc`, `metric`, `area`, `count`, each with its own weight).
- **Training is progressive:** ico3 is trained from scratch, then ico4, ico5 and ico6 are added one level per run, loading the previous checkpoint and freezing the lower levels.

## Layout

```
spherical_registration/
├── dataset.py            SphereRegDataset, template grid loading and sampling
├── network.py            SphereRegNet
├── loss.py               SulcalAlignmentLoss and mesh-quality terms
├── train.py              ico3 training from scratch
├── finetune_train.py     one fine level per run (--level 4 | 5 | 6)
├── fsaverage_grid/       template sulc grid (.npz)
├── fsaverage_aparc/      fsaverage aparc annotation on the brainsynth template mesh
└── evaluation/
    ├── inference.py            predicted spheres for one split (.sphere.reg)
    ├── statistics.py           mine vs FreeSurfer vs JOSA: MSE, correlation, folding
    ├── parcel_dice.py          Dice of the aparc parcellation (mine and JOSA), FreeSurfer as reference
    ├── hausdorff_boundaries.py boundary distances (HD90, HD100, MBD) of the parcellations vs FreeSurfer
    ├── parcel_dice_pairwise.py pairwise Dice / MAE / correlation on a few datasets
    └── outlier.py              VTK export of outlier subjects for ParaView
```

## Requirements

- The `brainnet`, `brainsynth` and `cortech` packages from this repository, plus `torch`, `nibabel`, `numpy`, `scipy`, `matplotlib`.
- The dataset under `/projects/brainnet-data/sphere-reg`, and the subject splits and `bad_surfaces.json` under `/projects/brainnet-data/subject_splits`.
- For parcel Dice, `fsaverage_aparc/{hemi}.aparc.annot` is the fsaverage aparc resampled onto the brainsynth template mesh: its labels follow the vertex order of the template `sphere.reg` (loaded through `brainsynth.resources.Template`), so always pair it with that sphere.

## Usage

Run everything from the project root so that the `experiments.spherical_registration` imports resolve.

```bash
# 1. ico3 from scratch -> runs/ico3_train
python -m experiments.spherical_registration.train

# 2. fine levels, in order; each starts from the previous level's best checkpoint
python -m experiments.spherical_registration.finetune_train --level 4
python -m experiments.spherical_registration.finetune_train --level 5
python -m experiments.spherical_registration.finetune_train --level 6

# 3. evaluation of one run
python -m experiments.spherical_registration.evaluation.inference --run_dir runs/ico6_train
python -m experiments.spherical_registration.evaluation.statistics --run_dir runs/ico6_train
python -m experiments.spherical_registration.evaluation.parcel_dice --run_dir runs/ico6_train
python -m experiments.spherical_registration.evaluation.hausdorff_boundaries --run_dir runs/ico6_train
python -m experiments.spherical_registration.evaluation.outlier --run_dir runs/ico6_train
python -m experiments.spherical_registration.evaluation.parcel_dice_pairwise --run_dir runs/ico6_train
```

Outputs of a run all live under its folder:

```
runs/ico6_train/
├── best_val_total.pt
├── test_predictions/        inference.py
├── statistics/              statistics.py
├── parcel_dice/             parcel_dice.py (+ hausdorff_*.csv from hausdorff_boundaries.py)
│   └── outlier_vtk/         outlier.py
└── dice_pairwise/           parcel_dice_pairwise.py
```

Training writes, for each level: `best_val_total.pt`, `loss_per_step.csv`, `loss_history.csv`, loss plots and sample spheres per epoch. The per-step CSVs share one schema (with an `ico_level` column), so the four levels can be concatenated to plot each loss component across levels.

## Configuration

Hyperparameters and loss weights are constants in the `CONFIG` block of each script. Output locations are set from the command line: `--out_dir` for `train.py` and `finetune_train.py` (default `runs/`, results in `<out_dir>/ico<level>_train`), `--run_dir` for the evaluation scripts, which read `<run_dir>/test_predictions` unless `--pred_dir` points elsewhere (e.g. predictions on all subjects of the unseen datasets). `finetune_train.py --level N` starts from `<out_dir>/ico<N-1>_train/best_val_total.pt`, so use the same `--out_dir` for all levels.
