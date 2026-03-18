# Ev-GCN Pruning

Hardware-aware Graph Neural Network for event-based vision. Combines structured pruning and quantization to optimize GNN inference on resource-constrained hardware (FPGAs).

## Overview

Event cameras produce sparse, asynchronous streams of pixel-level brightness changes. This project processes those events as spatial-temporal graphs and classifies them using a PointNet-style GNN. The pipeline supports:

- **Structured pruning** — reduces channel counts per layer via L_n-norm filter removal
- **Post-training quantization** — 6 or 8-bit integer-only inference with observer-based calibration
- **Design space exploration** — exhaustive or tree-search over pruning × quantization configurations, scored against FPGA BRAM and multiplier budgets

### Supported Datasets

| Dataset | Classes | Description |
|---|---|---|
| MNIST-DVS | 10 | DVS recordings of moving MNIST digits |
| CIFAR10-DVS | 10 | DVS recordings of CIFAR-10 images |
| N-CARS | 2 | Cars vs. background |
| N-Caltech101 | 101 | Event-based Caltech101 |

## Architecture

```
Input Events (x, y, t, p)
  └─► Graph Construction (C++ / Neighbour Matrix in 3D)
        └─► Conv1 (PointNet) → Pool1
              └─► Conv2 → Conv3 → Pool2
                    └─► Conv4 → Conv5 → PoolOut
                          └─► Linear1 → ReLU → Linear2 → LogSoftmax
```

- **PointNet convolutions**: Message passing over spatial-temporal neighbourhoods; supports float, calibration, and quantized modes.
- **Graph pooling**: Spatial clustering to progressively reduce the graph.
- **Quantization**: FakeQuantize + Observer pattern; BN parameters are fused into preceding weights before export.

## Setup

```bash
conda create -n dvs_prun python=3.9
conda activate dvs_prun
conda install -c conda-forge libstdcxx-ng

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install omegaconf opencv-python matplotlib psutil wandb lightning numba pybind11 tqdm pandas

python setup.py build_ext --inplace   # compile C++ graph-construction extension
```

## Usage

### 1. Train a baseline model

```bash
python train.py
```

Configure dataset, architecture, and hyperparameters in `configs/`. Checkpoints are saved to `checkpoints/`.

### 2. Explore the pruning × quantization design space

```bash
# Exhaustive Cartesian-product search (CIFAR)
python explore.py

# Efficient tree search — finds Pareto-optimal accuracy / hardware tradeoffs
python explore_futher_by_tree.py          # MNIST-DVS
python explore_futher_by_tree_ncaltech.py # N-Caltech101
```

Results are written to `results_<dataset>.csv` with per-layer pruning, bit-widths, BRAM estimates, multiplier counts, and test accuracy.

### 3. Fine-tune a pruned / quantized model

```bash
python finetune.py
```

Runs calibration, quantization, and a short fine-tuning pass at a reduced learning rate. Saves weights to `weights_<dataset>/`.

### 4. Export weights for hardware deployment

```bash
python generate_weights_outputs.py
```

Exports quantized weights, biases, scales, and zero-points as binary/text files for use in an FPGA or embedded implementation. Also saves intermediate layer activations for debugging.

## Project Structure

```
configs/                    YAML configs per dataset
data/
  base/event_ds.py          Core event dataset class
  base/augmentation.py      Augmentation utilities
  utils/matrix_neighbour.cpp C++ KNN graph builder (OpenMP)
  cifar.py / mnist.py / ncaltech.py / ncars.py   Per-dataset loaders
models/
  layers/my_pointnet.py     PointNet graph convolution
  layers/my_linear.py       Quantized linear layer
  layers/my_max_pool.py     Graph pooling
  model.py                  Full model (MyModel)
  model_tiny.py             Tiny model for debugging
  recognition.py            PyTorch Lightning wrapper
  quantisation/observer.py  Quantization observers
utils/
  structured_pruning.py     Filter pruning helpers
  precompute_space.py       Design space enumeration
  generate_outputs.py       Debug output generation
  visualisation.py          Visualization utilities
train.py                    Training entry point
test.py                     Evaluation with pruning / quantization
finetune.py                 Fine-tuning script
explore.py                  Exhaustive design space search
explore_futher_by_tree.py   Tree-based Pareto search
generate_weights_outputs.py Weight export for deployment
setup.py                    Build C++ extension
```

## Configuration

Edit the YAML files in `configs/` to change:

- `data_name` — dataset selector (`mnist-dvs`, `cifar10-dvs`, `ncaltech101`, `ncars`)
- Layer widths, quantization bit-widths, graph radii
- Training hyperparameters (learning rate, epochs, batch size)
- Data augmentation (rotation, horizontal flip)

If you find the resources usefull, please cite the paper:

```
@INPROCEEDINGS{11215154,
  author={Wzorek, Piotr and Jeziorek, Kamil and Kryjak, Tomasz},
  booktitle={2025 Signal Processing: Algorithms, Architectures, Arrangements, and Applications (SPA)}, 
  title={Hardware-aware Graph Neural Networks prunning for embedded event-based vision}, 
  year={2025},
  volume={},
  number={},
  pages={182-187},
  keywords={Adaptation models;Accuracy;Quantization (signal);Event detection;Search methods;Robot vision systems;Signal processing algorithms;Cameras;System-on-chip;Field programmable gate arrays;SoC FPGA;Graph Convolutional Nerual Networks;Event Cameras;Prunning;Quantization},
  doi={10.23919/SPA65537.2025.11215154}}
```