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
├── fsaverage_aparc/      fsaverage sphere and aparc / DKT annotations
└── evaluation/
    ├── inference.py            predicted spheres for the test set (.sphere.reg)
    ├── statistics.py           mine vs FreeSurfer: MSE, correlation, folding
    ├── parcel_dice.py          Dice of the aparc parcellation, FreeSurfer as reference
    ├── parcel_dice_pairwise.py pairwise Dice / MAE / correlation on a few datasets
    ├── dice_spec.py            Dice of selected regions (precentral, postcentral) for outliers
    └── outlier.py              VTK export of outlier subjects for ParaView
```

## Requirements

- The `brainnet`, `brainsynth` and `cortech` packages from this repository, plus `torch`, `nibabel`, `numpy`, `scipy`, `matplotlib`.
- The dataset under `/projects/brainnet-data/sphere-reg`, and the subject splits and `bad_surfaces.json` under `/projects/brainnet-data/subject_splits`.
- For parcel Dice, the FreeSurfer 8.2.0 fsaverage subject (`parcel_dice_pairwise.py` reads it from `/software/freesurfer/8.2.0/subjects/fsaverage`).

## Usage

Run everything from the project root so that the `experiments.spherical_registration` imports resolve.

```bash
# 1. ico3 from scratch
python -m experiments.spherical_registration.train

# 2. fine levels, in order; each starts from the previous level's best checkpoint
python -m experiments.spherical_registration.finetune_train --level 4
python -m experiments.spherical_registration.finetune_train --level 5
python -m experiments.spherical_registration.finetune_train --level 6

# 3. evaluation (scripts are configured through the constants at the top of each file)
python -m experiments.spherical_registration.evaluation.inference
python -m experiments.spherical_registration.evaluation.statistics
python -m experiments.spherical_registration.evaluation.parcel_dice
```

Training writes, for each level: `best_val_total.pt`, `loss_per_step.csv`, `loss_history.csv`, loss plots and sample spheres per epoch. The per-step CSVs share one schema (with an `ico_level` column), so the four levels can be concatenated to plot each loss component across levels.

## Configuration

There are no command-line configs apart from `--level`: hyperparameters, loss weights and paths are constants in the `CONFIG` block of each script.

Some of these paths still point to `experiments/sphere_reg2/` (`runs/`, `new_network/`) for checkpoints and outputs. Update them if you keep results elsewhere. `finetune_train.py` reads the ico3 checkpoint from `SAVE_ROOT / "ico3_train"`, which is where `train.py` writes it, so keep `SAVE_DIR` in `train.py` and `SAVE_ROOT` in `finetune_train.py` consistent.
