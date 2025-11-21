# **FastAIT**

**FastAIT** is an open-source project for **accelerated post-processing in active infrared thermography (AIT)** using modern **GPU computing**.  
It provides optimized implementations of several widely used thermographic analysis methods, enabling fast post-processing in nondestructive testing (NDT) applications.

## Overview

The project includes functionalities for:
- Loading and managing thermographic image sequences  
- Normalizing and preprocessing thermal data  
- Extracting heating and cooling
- Applying post-processing techniques:
  - **Skewness**
  - **Kurtosis**
  - **Pulse Phase Thermography (PPT)**
  - **Principal Component Thermography (PCT)**
  - **Thermographic Signal Reconstruction (TSR)**  

## Installation

### Local installation (development mode)

```bash
git clone https://github.com/yourusername/fastait.git
cd fastait
pip install -r requirements.txt
```

### Install directly from GitHub (recommended)

```bash
pip install git+https://github.com/rusamentiaga/fastait.git
```

## Usage

### Post-processing methods

```python
from fastait.statistics import skewness, kurtosis
from fastait.ppt import ppt
from fastait.pct import pct
from fastait.tsr import tsr

# data (torch.Tensor): Tensor of shape (N, H, W)
data_skewness = skewness(data)
data_kurtosis = kurtosis(data)
data_phase = ppt(data)
data_pct = pct(data, 50)
data_pct = pct(data, n_components=50)
tsr_res = tsr(data, degree=5)
data_tsr = tsr_res.reconstruction()
data_der1 = tsr_res.first_derivative()
data_der2 = tsr_res.second_derivative()
```

See `demo.py` for additional examples.

Use `./download_test_data.sh` to download example datasets.

### View the sequences

To interactively browse through a thermographic image sequence, the `show` function can be used.

```python
%matplotlib widget

show(images)
```

Notes:

* `images` should be a 2D or 3D torch.Tensor of shape (H, W) for a single image or (N, H, W) for a sequence.

* A slider will appear below the image if multiple frames are provided.

* Make sure ipympl is installed for interactive plotting:

### Benchmarking

```bash
python benchmark.py
```

## Benchmark results

Benchmark results obtained on an **NVIDIA RTX 5090 GPU** with an **AMD Ryzen 9 9950X3D CPU**  
for **2000 frames of size 512 × 512 pixels** (≈524 megapixels total):

| Method   | Time (ms) |
|-----------|-----------:|
| Skewness | 11.98 |
| Kurtosis | 11.98 |
| PPT      | 21.35 |
| TSR      | 26.79 |
| PCT      | 31.26 |

## License

This project is released under the **MIT License**.

