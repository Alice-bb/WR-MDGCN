# WR-MDGCN: Wavelet Residual and Multi-Scale Dynamic Graph Convolution Network

This repository contains the official PyTorch implementation of the paper:

**"WR-MDGCN: Wavelet Residual and Multi-Scale Dynamic Graph Convolution Network for Traffic Flow Prediction"** (Under Review)

## Paper Status

-  Under Review 

---

## Introduction

WR-MDGCN is a novel spatiotemporal graph neural network designed to forecast traffic flow more accurately by modeling:

1. **Wavelet Residual Decomposition** to extract periodic and anomalous traffic signals.
2. **Multi-Scale Dynamic Graph Construction** for capturing evolving spatial dependencies.
3. **PSPM (Periodic Signal Processing Module)** with **PGCRU + GMM** for long-term periodic dynamics.
4. **ASPM (Anomalous Signal Processing Module)** with **HDGCRU + DGP** for anomaly-aware hierarchical diffusion.
5. **SSM (Signal Separation Module)** for decoupling input signals and multi-path fusion.

---

## Table of Contents

- configs: training Configs and model configs for each dataset
- lib: contains self-defined modules for our work, such as data loading, data pre-process, normalization, and evaluate metrics.
- model: implementation of our model
- pre-trained: the trained model files (.pth) for result verification and reproduction

---

## Requirements

- Python ≥ 3.7.16
- PyTorch ≥ 1.13
- NumPy, Pandas, SciPy, Matplotlib
- PyWavelets ≥ 1.3.0

---

## Model Training

```bash
python run.py --dataset {DATASET_NAME} --mode {MODE_NAME}
```

Replace `{DATASET_NAME}` with one of `PEMSD3`, `PEMSD4`, `PEMSD7`, `PEMSD8`, `PEMS-BAY`, `xian_taxi`

such as `python run.py --datasets PEMSD4`

There are two options for `{MODE_NAME}` : `train` and `test`

Selecting `train` will retrain the model and save the trained model parameters and records in the `experiment` folder.

With `test` selected, run.py will import the trained model parameters from `{DATASET_NAME}.pth` in the `pre-trained` folder. 

---
## Tips for Dataset Compatibility

Different datasets contain different feature types. Please **manually modify `get_dataloader()` in `dataloader.py`** and the corresponding model inputs in the following files according to the dataset you use:

### Supported Features per Dataset

| Dataset        | flow | speed | occupy |
| -------------- | ---- | ----- | ------ |
| **PEMSD3**     | ✔️    | ❌     | ❌      |
| **PEMSD4**     | ✔️    | ✔️     | ✔️      |
| **PEMSD7**     | ✔️    | ❌     | ❌      |
| **PEMSD8**     | ✔️    | ✔️     | ✔️      |
| **PEMS-BAY**   | ❌    | ✔️     | ❌      |
| **Xi’an Taxi** | ✔️    | ✔️     | ❌      |

###  Files to Update

If you use a dataset that does **not contain certain features**, please comment out the corresponding parts in:

- `dataloader.py`: `get_dataloader()`
   e.g., comment out 

  ```
  speed = load_st_speed(...)
  speed = np.array(speed)
  feature_list.append(speed)
  x_speed, y_speed =  Add_Window_Horizon(speed, args.lag, args.horizon, single)
  ```

   if dataset has no speed

- `WRMDGC.py`, `DDGCRNCell.py`, `GWCCell.py`, `DCNN.py`, `DDGCN.py`
   Make sure to  adjust the relevant function parameters in model files such as `WRMDGC.py`, `DDGCRNCell.py`, `GWCCell.py`, etc., to match the selected input features.

