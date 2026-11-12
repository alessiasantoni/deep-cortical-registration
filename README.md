# Deep cortical registration

### Ongoing master thesis project carried out at the Technical University of Denmark (DTU) and Politecnico di Milano.

Deep-learning registration of the cortical surface onto the fsaverage template sphere, driven by sulcal depth. A coarse-to-fine graph network deforms the sphere of each subject so that its sulcal depth matches the template, while keeping the mesh regular (no folded triangles, edge lengths close to the white-matter surface). The results are compared with FreeSurfer and JOSA, and with the manual labels of Mindboggle-101.

## Repository

```
experiments/spherical_registration/   the registration network: training and evaluation
brainnet/  brainsynth/  cortech/       libraries the code depends on
```

Model, training, evaluation and how to run them are described in [experiments/spherical_registration/README.md](experiments/spherical_registration/README.md).
