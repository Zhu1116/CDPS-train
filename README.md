# [TIP 2025] CDPS-train

This is the official pytorch code for "**Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral and Multispectral Images Fusion**", which has been accepted by TIP2025.

## Author

**Yang Xu; Jian Zhu; Danfeng Hong; Zhihui Wei; Zebin Wu**

## Requirements 

1. Environment setup

```shell
conda create -n cdps python=3.9
conda activate cdps
```

2. Requirements installation

```shell
pip install -r requirements.txt
```

## Quick Start (using the Cave dataset as an example)

#### Train spatial networks

```bash
python spatial_train.py
```

#### Train spectral networks

```bash
python spectral_train.py
```

## Train on Your Own Data

#### Train the Spatial Network

1. Place the `x.mat` file into the `datasets` folder. This file should contain the keys: `LRHSI`, `HRMSI`, and optionally `HRHSI`.

2. Modify lines 66 and 67 in `spatial_train.py` accordingly. For example:

   ```python
   data_idx = 3
   type_list = ['cave', 'pavia', 'wdc', 'x']
   ```

3. Run:

   ```bash
   python spatial_train.py
   ```

#### Train the Spectral Network

1. Ensure that `x.mat` is placed in the `datasets` folder.

2. Modify lines 66, 67, and 68 in `spectral_train.py` as follows:

   ```python
   data_idx = 3
   type_list = ['cave', 'pavia', 'wdc', 'x']
   step_list = [300000, 500000, 800000, step]  # e.g., if this is the 'ksc' dataset with 176 spectral bands, here can be step=800000
   ```

3. In `script_util.py`, after line 163, set the corresponding spectral network architecture. For example:

   ```python
   elif data_type == 'ksc':
       model = FCN(176, 176, [400, 800, 400], 100, num_embeddings=diffusion_steps)
   ```

   There are no strict requirements here — approximate configurations are fine. Feel free to experiment.

4. Run:

   ```bash
   python spectral_train.py
   ```

## Acknowledge

Some of the codes are built upon [guided-diffusion](https://github.com/openai/guided-diffusion).