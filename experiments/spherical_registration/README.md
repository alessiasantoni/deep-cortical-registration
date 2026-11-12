# Spherical registration

Coarse-to-fine deep-learning registration of the cortical surface (left hemisphere, white-matter surface) onto the fsaverage template sphere, driven by sulcal depth (sulc). For each subject the network deforms an initial sphere so that the subject's sulc matches the template sulc, while keeping the mesh well-behaved (no folded triangles, edge lengths close to those of the white-matter surface). No registration is used as training target. Results are compared with FreeSurfer and JOSA, and with the manual labels of Mindboggle-101.

## How it works

- **Input per subject:** `sulc` (`derivatives/{hemi}.sulc.pt`), the white-matter surface (`{hemi}.white`, used only in the loss), and the template sulc rasterized on a 2D latitude-longitude grid (`fsaverage_grid/`). All surfaces share the topology of the template mesh, so every subject starts from the same undeformed template sphere.
- **Model** ([network.py](network.py), `SphereRegNet`): one graph U-Net (edge convolutions on recursively subdivided spherical meshes) and one deformation head per resolution level, from ico3 to ico6. At each level the vertices are moved for a number of iterations: the template sulc is sampled at the current positions, concatenated with the subject sulc, and the network predicts a vector that is applied as a rotation on the sphere (forward Euler step), so points stay on the sphere.
- **Iterations per level:** at ico3 the number of iterations is drawn at random in [15, 40] at every training step and fixed to 25 for validation and inference; then 7, 5 and 3 iterations at ico4, ico5 and ico6.
- **Loss** ([loss.py](loss.py), `SulcalAlignmentLoss`): sulc alignment with the template + edge-length preservation against the WM surface + folded-area penalty + a differentiable folded-triangle count (`sulc`, `metric`, `area`, `count`, each with its own weight per level).
- **Training is progressive:** ico3 is trained from scratch, then ico4, ico5 and ico6 are added one level per run, loading the previous checkpoint and freezing the lower levels.

## Layout

```
spherical_registration/
├── dataset.py            SphereRegDataset, template grid loading and sampling
├── network.py            SphereRegNet
├── loss.py               SulcalAlignmentLoss and mesh-quality terms
├── train.py              ico3 training from scratch
├── finetune_train.py     one fine level per run (--level 4 | 5 | 6); also holds the data paths
├── fsaverage_grid/       template sulc grid (.npz)
└── evaluation/
    ├── atlas.py                fsaverage aparc atlas shared by all the evaluation scripts
    ├── inference.py            predicted spheres for the test set (.sphere.reg)
    ├── statistics.py           proposed vs FreeSurfer vs JOSA: sulc MSE and correlation with fsaverage, folding
    ├── parcel_dice.py          Dice of the aparc parcellation (proposed, JOSA, no registration), FreeSurfer as reference
    ├── hausdorff_boundaries.py boundary distances (HD90, HD100, MBD) of the parcellations vs FreeSurfer
    └── manual_labels/          evaluation against the manual labels of Mindboggle-101
        ├── common.py               data arguments and subject list
        ├── dkt_labels.py           aparc -> DKT31 / DKT25 manual protocols
        ├── inference_mindboggle.py predicted spheres of the Mindboggle-101 subjects
        ├── dice_vs_manual.py       Dice vs manual labels
        └── hausdorff_vs_manual.py  boundary distances vs manual labels
```

## Requirements

- The `brainnet`, `brainsynth` and `cortech` packages (https://github.com/simnibs/brainnet, https://github.com/simnibs/brainsynth, https://github.com/simnibs/cortech), plus `torch`, `nibabel`, `numpy`, `scipy`, `matplotlib`.
- The data folder, the subject splits and `bad_surfaces.json`, whose paths are set at the top of [finetune_train.py](finetune_train.py) (`DATA_ROOT`, `SPLIT_DIR`, `BAD_SURFACES`) and used by all the other scripts. Each subject folder contains `{hemi}.white` and `derivatives/` with `{hemi}.sulc`, `{hemi}.sulc.pt`, `{hemi}.sphere.reg.freesurfer` and `{hemi}.sphere.reg.josa`.
- FreeSurfer's `fsaverage` subject, used as atlas for the parcellation: its original `label/{hemi}.aparc.annot` on `surf/{hemi}.sphere.reg`. It is read from `$FREESURFER_HOME/subjects/fsaverage`, or from `--fsaverage_dir`.
- For `manual_labels/`: the Mindboggle-101 subjects in the same layout, with `{hemi}.labels.DKT31.manual.annot`, and a text file with one line per subject, `<Mindboggle name> <sub-id>` (see [manual_labels/common.py](evaluation/manual_labels/common.py)).

## Usage

Run everything from the repository root so that the `experiments.spherical_registration` imports resolve.

```bash
# 1. ico3 from scratch -> runs/ico3_train
python -m experiments.spherical_registration.train

# 2. fine levels, in order; each starts from the previous level's best checkpoint
python -m experiments.spherical_registration.finetune_train --level 4
python -m experiments.spherical_registration.finetune_train --level 5
python -m experiments.spherical_registration.finetune_train --level 6

# 3. evaluation against FreeSurfer and JOSA
export FREESURFER_HOME=/path/to/freesurfer
python -m experiments.spherical_registration.evaluation.inference --run_dir runs/ico6_train
python -m experiments.spherical_registration.evaluation.statistics --run_dir runs/ico6_train
python -m experiments.spherical_registration.evaluation.parcel_dice --run_dir runs/ico6_train
python -m experiments.spherical_registration.evaluation.hausdorff_boundaries --run_dir runs/ico6_train

# 4. evaluation against the manual labels of Mindboggle-101
M=experiments.spherical_registration.evaluation.manual_labels
python -m $M.inference_mindboggle --data_root <Mindboggle101> --id_map <id_map.txt> --run_dir runs/ico6_train
python -m $M.dice_vs_manual       --data_root <Mindboggle101> --id_map <id_map.txt> --run_dir runs/ico6_train
python -m $M.hausdorff_vs_manual  --data_root <Mindboggle101> --id_map <id_map.txt> --run_dir runs/ico6_train
```

Outputs of a run all live under its folder:

```
runs/ico6_train/
├── best_val_total.pt
├── test_predictions/        inference.py
├── statistics/              statistics.py
├── parcel_dice/             parcel_dice.py (+ hausdorff_*.csv from hausdorff_boundaries.py)
├── mindboggle_predictions/  inference_mindboggle.py
├── dice_vs_manual/          dice_vs_manual.py
└── hausdorff_vs_manual/     hausdorff_vs_manual.py
```

Training writes, for each level: `best_val_total.pt`, `loss_per_step.csv`, `loss_history.csv`, loss plots and sample spheres per epoch. The per-step CSVs share one schema (with an `ico_level` column), so the four levels can be concatenated to plot each loss component across levels.

## Configuration

Hyperparameters and loss weights are constants in the `CONFIG` block of `train.py` and `finetune_train.py`. Output locations are set from the command line: `--out_dir` for `train.py` and `finetune_train.py` (default `runs/`, results in `<out_dir>/ico<level>_train`), `--run_dir` for the evaluation scripts, which read `<run_dir>/test_predictions` unless `--pred_dir` points elsewhere. `finetune_train.py --level N` starts from `<out_dir>/ico<N-1>_train/best_val_total.pt`, so use the same `--out_dir` for all levels.

## Reproducibility

The graph convolutions use `torch.index_reduce`, which is not deterministic on the GPU: running the same checkpoint twice gives spheres that differ by about 0.1 (sphere radius 100), which changes the transferred label of a few vertices per subject (about 0.03%). The evaluation scripts are deterministic for given spheres.
