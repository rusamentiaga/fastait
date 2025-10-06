# fastait

A project for fast post-processing of active infrared thermography using the GPU.

## Installation

```bash
git clone https://github.com/yourusername/fastait.git
cd fastait
pip install -r requirements.txt
```

## Usage

### Benchmarking

```bash
python benchmark.py
```

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

See demo.py for more examples (run `./download_test_data.sh` to download the data first).

## Results

Results on RTX 5090 with AMD Ryzen 9 9950X3D for 2000 frames of size 512 × 512 pixels (524.3 megapixels)

| Method    | Time (ms) |
|------------|------------|
| Skewness  | 11.98      |
| Kurtosis  | 11.98      |
| PPT       | 21.35      |
| TSR       | 26.79      |
| PCT       | 31.26      |

