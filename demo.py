#%%
# This script demonstrates the usage of various post-processing techniques on a sequence of images.
# It includes loading image data, normalizing, extracting cooling effects, and applying post-processing
# methods such as Skewness, Kurtosis, Pulse Phase Thermography (PPT),
# Principal Component Thermography (PCT), Thermographic Signal Reconstruction (TSR)
# The results are visualized using matplotlib and seaborn.

# Each cell can be run independently in a Jupyter notebook or similar environment.

%matplotlib widget
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import fastait.io
import fastait.data
from fastait.statistics import skewness, kurtosis
from fastait.ppt import ppt
from fastait.tsr import tsr
from fastait.pct import pct
from fastait.plot import show, plot_image_grid
from fastait.profiler import TimeProfiler

sns.set_context("paper")
sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})

from matplotlib.colors import ListedColormap
colors = np.loadtxt('misc/rain.csv', delimiter=',') / 255.0 
cmap = ListedColormap(colors)

#%% Load Data
# Run ./download_test_data.sh to download the data first
# images is a tensor of shape (num_images, height, width)
images = fastait.io.load_csv_folder("data/CFRP-006_facq-145Hz_s-Front_Img-2000", 
                                    dtype=torch.float32, verbose=True)
print(f"Loaded images tensor shape: {images.shape}, dtype: {images.dtype}")
roi_margin_top, roi_margin_left, roi_margin_bottom, roi_margin_right = 20, 20, 40, 20
images = images[:, roi_margin_top:-roi_margin_bottom, roi_margin_left:-roi_margin_right].contiguous()
print(f"Cropped images to remove margin: new shape {images.shape}")
images_normalized = fastait.data.normalize_percentile(images, 3.0, 100.0)
cooling = fastait.data.extract_cooling(images)
device = "cuda" if torch.cuda.is_available() else "cpu"
cooling = cooling.to(device)

#%% Visualize raw images
show(images, cmap=cmap)

#%% Visualize a single image
show(images[100], cmap=cmap)

#%% Visualize a grid of cooling images
plot_image_grid(cooling, cmap="gray", start_index=0, num_images=12, images_per_row=4, 
                fig_width=8, row_height=2, step=30)

#%% Plot mean pixel value per image
mean_per_image = images.mean(dim=(1, 2)).cpu().numpy()
t = torch.arange(len(mean_per_image)).numpy()

fig, ax = plt.subplots()
ax.plot(t, mean_per_image, ".-")
ax.set_xlabel("Frame")
ax.set_ylabel("Mean pixel value")
ax.set_title("Mean Pixel Value per Image")
# log x
ax.set_xscale('log')
ax.grid(True)
fig.tight_layout()
plt.show()

#%% Apply post-processing techniques: Skewness
with TimeProfiler("Skewness") as tp:
    data_skewness = skewness(cooling)

show(data_skewness, cmap=cmap)

#%% Kurtosis
with TimeProfiler("Kurtosis") as tp:
    data_kurtosis = kurtosis(cooling)

show(data_kurtosis, cmap=cmap)

#%% PPT
with TimeProfiler("PPT") as tp:
    data_phase = ppt(cooling)

show(data_phase, cmap=cmap)

#%% PCT
with TimeProfiler("PCT") as tp:
    data_pct = pct(cooling, n_components=50)

plot_image_grid(data_pct.cpu(), cmap="gray", start_index=0, num_images=8, images_per_row=4, 
                fig_width=8, row_height=2, step=1)

show(data_pct, cmap=cmap)

#%% TSR
with TimeProfiler("TSR") as tp:
    tsr_res = tsr(cooling, degree=7)
    coefficients = tsr_res.coefficients()
    data_tsr = tsr_res.reconstruction()
    data_der1 = tsr_res.first_derivative()
    data_der2 = tsr_res.second_derivative()

plot_image_grid(data_der2.cpu(), cmap="gray", start_index=0, num_images=8, images_per_row=4, 
                fig_width=8, row_height=2, step=10)

show(data_der2, cmap=cmap)

#%% TSR fit visualization
data_tsr = tsr_res.reconstruction()
i = cooling.shape[1] // 2
j = cooling.shape[2] // 2
fig, ax = plt.subplots()
plt.plot(cooling[:, i, j].cpu().numpy(), 'k.', label='Noisy')
plt.plot(data_tsr[:, i, j].cpu().numpy(), 'r-', label='Fitted')
plt.title(f'Pixel ({i},{j})')
plt.legend()
plt.tight_layout()
plt.show()
