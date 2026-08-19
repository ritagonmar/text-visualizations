<!--
# Research project
This is a template for the repository of a reseach project containing research code. It provides the folder structure and the pre-commit hooks, and it assumes you are using uv as your packaging manager.

1. After creating a new repo off this one, initialize a uv environment by running `uv init --python 3.12`. This creates the environment and adds all uv-related files to the repo.
2. To set up the pre-commit hooks, activate your uv environment with `source .venv/bin/activate`, and then run `make install_hooks`.
3. Run `make install_jupyter` to get jupyter working.
4. Run `make install_python_basics` to install some python basic files.
5. For the installable package, a name has to be choosen, the `src/` folder renamed, and the `mypyproject.toml` file and notebook imports edited accordingly. Afterwards, in the uv environment, run `uv pip install -e .`.
-->

# Text visualizations

Learning 2D text visualizations end-to-end: a projection head on top of an SBERT backbone is trained with a contrastive, t-SNE-like InfoNCE loss, so the resulting 2D embeddings are simultaneously a good visualization (t-SNE-like layout) and a good sentence embedding (high KNN/MTEB accuracy). Experiments are run on a dataset of ICLR paper abstracts (`data/iclr25v2.parquet`), comparing loss variants (Cosine/Gaussian/t/Cauchy InfoNCE) and text augmentation strategies (overlapping sentence pairs, cropping, masking).

## Pipeline

1. `scripts/03-rgm-baseline-sbert-tsne.py`: baseline: SBERT embeddings + classic (non-parametric) t-SNE on the ICLR abstracts, with KNN accuracy computed in both high- and low-dimensional space.
2. `scripts/train_model.py` / `scripts/04-rgm-train-tsne-and-augm.py`: trains the SBERT-based projector end-to-end with a contrastive InfoNCE loss on augmented sentence pairs, driven by a YAML experiment config from `configs/`.
3. `scripts/train_hyper_sweep.py` / `scripts/06-rgm-sweep-tsne-and-augm.py`: runs hyperparameter sweeps (learning rate, batch size, scale, ...) over the sweep configs in `configs/temp_sweep/`.
4. `scripts/01-rgm-analysis-exps.ipynb`, `02-rgm-analysis-hyper-sweeps-cauchy.ipynb`, `02-rgm-analysis-hyper-sweeps-t.ipynb`, `05-rgm-analysis-tsne-and-augm.ipynb`, `07-rgm-analysis-augmentations.ipynb`: notebooks analyzing individual experiments, hyperparameter sweeps, and augmentation strategies.

Each experiment is defined by a YAML config (`configs/base_config.yaml` plus per-experiment overrides) specifying the backbone model, data augmentation, and training hyperparameters.

Core code lives in `src/text_visualizations/`:
- `models.py`: model wrappers, including the SBERT + projection head (`ModelProjector`).
- `infonce.py`: InfoNCE loss variants (Cosine, Gaussian, t, Cauchy).
- `data_stuff.py`: dataset/augmentation classes (overlapping sentence pairs, cropping, masking).
- `train_stuff.py`: training loop and pooling functions.
- `eval_functions.py`: KNN, linear-probe, and MTEB evaluation.
- `dim_red.py`, `embeddings.py`, `metrics.py`: t-SNE, embedding generation, and metric helpers.
- `config_helpers.py`, `logger.py`, `sweep_analysis_helpers.py`, `plotting.py`, `scalebars.py`: config parsing, experiment logging, sweep analysis, and plotting utilities.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (Python 3.12).

```bash
uv sync
make install_hooks  # installs pre-commit hooks
```
