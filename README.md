# Bridging Representational and Functional Similarity

This repo implements a pipeline to compare neural networks via:
- **Representational similarity** between intermediate activations (CKA, SVCCA/CCA, RSA, and alignment-based scores).
- **Functional similarity** via **model stitching**: train lightweight “stitchers” that map activations from one model into another and report **accuracy-ratio matrices** across layer pairs.

## Installation

From the repo root (e.g. `/mnt/cephfs/home/voz/almudevar/similarity`):

```bash
pip install -r requirements.txt
```

Notes:
- `requirements.txt` pins `torch==1.12.1+cu113` / `torchvision==0.13.1+cu113`. Adjust if you need a different CUDA/CPU build.
- The scripts in `bin/` hardcode a `PROJECT_ROOT` (`/home/voz/almudevar/similarity`). If you move the repo, update `PROJECT_ROOT` in those scripts (or ensure the path matches via symlink).

## Entry points

- Train CNNs (from scratch): `python bin/train_cnn.py ...`
- Train linear baselines (from scratch): `python bin/train_linear.py ...`
- Run stitching + representation metrics: `python bin/run_stitching_extended.py ...`

## Quickstart

1) Train two models (example: a CNN and a linear baseline on CIFAR-10):

```bash
python bin/train_cnn.py --model resnet20 --dataset cifar10 --epochs 100 --device cuda
python bin/train_linear.py --model linear_medium --dataset cifar10 --epochs 200 --device cuda
```

2) Compute similarity + stitching matrices (single experiment):

```bash
python bin/run_stitching_extended.py \\
  --mode single \\
  --model1 resnet20 \\
  --model2 linear_medium \\
  --dataset cifar10 \\
  --epochs 10 \\
  --stitcher-type all \\
  --device cuda
```

Outputs are written under `experiments/<model1>_<model2>_<dataset>/` (configurable with `--output-dir`).

## Models

### CNN models (`src/cnn_models.py`)

Examples:
- Custom: `tiny_cnn`, `narrow_cnn`, `narrow_cnn_wide`, `simple_cnn`
- CIFAR ResNets: `resnet20`, `resnet32`, `resnet44`, `resnet56`, `resnet110`
- CIFAR MobileNetV3: `mobilenetv3`, `mobilenetv3_small`, `mobilenetv3_large`
- CIFAR ShuffleNetV2: `shufflenetv2`, `shufflenetv2_small`, `shufflenetv2_large`
- CIFAR DenseNets: `densenet40`, `densenet100`, `densenet_bc100`, `densenet_bc190`, `densenet_bc250`

### Linear models (`src/linear_models.py`)

Available: `linear_small`, `linear_medium`, `linear_large`, `linear_deep`, `linear_wide`.

## Stitching and metrics

Stitching is implemented in `src/improved_stitching.py` (`ImprovedModelStitcher`) and supports multiple stitcher families:
- `affine`: unconstrained linear/1×1-conv mapping
- `orthogonal`: orthogonal mapping
- `orthogonal_scaled`: scaled-orthogonal mapping

Representation metrics are implemented in `src/similarity_metrics.py` and include:
- `cka`
- `svcca` / `cca`
- `rsa`
- alignment-based scores: `l2`, `procrustes`, `orthogonal_scaled`, `invertible_affine`

You can control how convolutional features are aggregated for representation metrics via:
- `--similarity-aggregation gap|flatten|spatial_samples`

## Outputs

`bin/run_stitching_extended.py` creates an experiment folder:

- `experiments/<model1>_<model2>_<dataset>/metadata.json`
- `experiments/<...>/*accuracy_ratio_matrix.npy` (and additional `*.npy` matrices per stitcher type)
- `experiments/<...>/cka_before_matrix.npy` (and other `*_before_matrix.npy` similarity matrices)

Trained checkpoints are saved under `trained_models/` as `*_best.pth` / `*_final.pth` along with `*_metadata.json`.

## Reproducing the paper

- Use `bin/train_cnn.py` / `bin/train_linear.py` to produce checkpoints and then run `bin/run_stitching_extended.py` to generate the similarity and stitching matrices used in the analysis.
