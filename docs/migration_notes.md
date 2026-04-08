
### Relevant research assets
- `anaconda_projects/db/mfg_posterior_project/mfg_post.ipynb`
- `anaconda_projects/db/mfg_posterior_project/clear_mfg_post.ipynb`
- `anaconda_projects/db/Sampling_project/Sampling.ipynb`
- ~14 generated PDF figures in `mfg_posterior_project/`


## Notebook roles
### `mfg_post.ipynb`
Large development notebook with:
- geometry primitives
- SDF / stiffness / wrench model
- analytic and numerical Jacobians
- Alg 1 / Alg 2
- diagnostics
- measurement-design ablations


### `Sampling.ipynb`
Sampling and visualization notebook for Matrix-Fisher behavior, pi-ball plots, tangent-space views, and family comparisons.

## GitHub 
Use **one repository** unless you want the sampling work to become a standalone reusable package.
A good first public version is:
- one repo
- two notebook subfolders
- one shared `src/mfg_pose_estimation/` package

## First extraction
1. `geometry.py`
2. `distributions.py`
3. `sdf.py`
4. `wrench_model.py`
5. `jacobians.py`
6. `algorithms.py`

After that, refactor the experimental blocks and visualization helpers.

## Notebook recommendation
If you want just one polished notebook in the first GitHub release, prefer:
- `clear_mfg_post.ipynb` for posterior / algorithm work
- `Sampling.ipynb` for distribution visualization
