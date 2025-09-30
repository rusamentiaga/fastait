#%%
from itertools import product
import torch.utils.benchmark as benchmark
import torch
from fastait.pct import pct
from fastait.tsr import tsr

num_threads = torch.get_num_threads()

H = [256, 512]
N = [1000, 2000]
ndtypes = [torch.float32, torch.float64]
ndevices = [torch.device('cpu')]
ndevices = []
if torch.cuda.is_available():
	ndevices.append(torch.device('cuda'))
    
results = []
for n, h, d, t in product(N, H, ndevices, ndtypes):

    images = torch.randn(n, h, h, dtype=t, device=d)
    sub_label = f'[{n} {h} {h}]'
    print(f'Testing {sub_label} {d} {t}')

    torch.cuda.empty_cache()

	# warmup
    for i in range(10):
        data_pct = pct(images, 50)

    t = benchmark.Timer(
        stmt='pct(images, 50)',
        globals={'images': images, 'pct': pct},
        num_threads=num_threads,
        label=str(d),
        sub_label=sub_label,
        description='pct' + f' [{t}]'
    ).blocked_autorange(min_run_time=1)
    results.append(t)

compare = benchmark.Compare(results)
compare.print()

#%%
import torch
import numpy as np

def generate_latex_table(results):
    """
    Generate a LaTeX table from torch.utils.benchmark Measurement results.
    """
    latex = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Benchmark results for \texttt{pct} on different devices and data types. Times are in milliseconds.}",
        r"\begin{tabular}{cccccc}",
        r"\hline",
        r"Input size & Device & Data type & Mean (ms) & Std (ms) \\",
        r"\hline"
    ]

    for t in results:
        # Extract input size and data type
        sub_label = t.sub_label      # e.g., "[500 256 256]"
        description = t.description  # e.g., "pct [torch.float32]"
        label = t.label              # e.g., "cpu" or "cuda"

        dtype = description.split('[')[-1].strip(']')  # "torch.float32"

        # Compute mean and std from raw times
        times_s = np.array(t.raw_times) / t.number_per_run  # Array of measured times in seconds
        mean_ms = np.mean(times_s) * 1000
        std_ms = times_s.std() * 1000

        latex.append(f"{sub_label} & {label} & {dtype} & {mean_ms:.2f} & {std_ms:.2f} \\\\")

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)

# Usage
latex_table = generate_latex_table(results)
print(latex_table)


