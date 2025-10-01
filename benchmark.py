#%%
from itertools import product
import torch
import torch.utils.benchmark as benchmark
from fastait.pct import pct
from fastait.tsr import tsr
from fastait.statistics import skewness, kurtosis
from fastait.ppt import ppt
from fastait.profiler import TimeProfiler
from tqdm import tqdm
from scipy import stats

def test_benchmark_pct(images, n_components):
    return pct(images, n_components)

def test_benchmark_tsr(images, degree):
    tsr_res = tsr(images, degree=degree)
    return tsr_res.reconstruction(), tsr_res.first_derivative(), tsr_res.second_derivative()

def test_benchmark_skewness(images):
    return skewness(images)

def test_benchmark_kurtosis(images):
    return kurtosis(images)

def test_benchmark_ppt(images):
    return ppt(images)

methods = {
    "Skewness": ("test_benchmark_skewness", ""),
    "Kurtosis": ("test_benchmark_kurtosis", ""),
    "PCT": ("test_benchmark_pct", 50),
    "TSR": ("test_benchmark_tsr", 5),
    "PPT": ("test_benchmark_ppt", ""),
}

H = [256, 512]
N = [500, 1000, 2000]
ndtypes = [torch.float32, torch.float64]
ndevices = [torch.device('cpu')]
if torch.cuda.is_available():
	ndevices.append(torch.device('cuda'))

results_all = []
results_by_method = {m: [] for m in methods}
total_iters = len(ndevices) * len(N) * len(H) * len(ndtypes)
bar = tqdm(product(ndevices, N, H, ndtypes), total=total_iters)
for d, n, h, t in bar:
    images = torch.randn(n, h, h, dtype=t, device=d)
    dtype_name = str(t).replace('torch.', '') 
    sub_label = f'({n}, {h}, {h})'
    bar.set_description(f"Testing {sub_label} {d} ({dtype_name})")

    for label, (func_name, arg) in methods.items():
        torch.cuda.empty_cache()
        tmr = benchmark.Timer(
            stmt=f'{func_name}(images, {arg})',
            globals={'images': images, func_name: globals()[func_name]},
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=f'{d} {dtype_name}'
        ).adaptive_autorange()

        results_all.append(tmr)
        results_by_method[label].append(tmr)

for label, res in results_by_method.items():
    benchmark.Compare(res).print()

#%%
import numpy as np
from scipy import stats

def generate_latex_tables_by_method(results):
    """
    Generate separate LaTeX tables for each method from torch.utils.benchmark Measurement results.
    Returns a dictionary with method names as keys and LaTeX table strings as values.
    """
    # Group results by method
    tables = {}
    for t in results:
        method = t.label  # Extract method name
        if method not in tables:
            tables[method] = []
        tables[method].append(t)

    latex_tables = {}

    for method, method_results in tables.items():
        # Group results by input size and data type
        grouped = {}
        for t in method_results:
            sub_label = t.sub_label
            device = t.description.split()[0]
            dtype = t.description.split()[1]
            key = (sub_label, dtype)

            times_s = np.array(t.raw_times) / t.number_per_run
            n = len(times_s)
            mean_s = np.mean(times_s)
            std_s = np.std(times_s, ddof=1)  # sample standard deviation

            # 95% confidence interval
            ci = stats.t.ppf(0.975, df=n-1) * std_s / np.sqrt(n)

            # Convert to milliseconds
            mean_ms = mean_s * 1000
            ci_ms = ci * 1000

            if key not in grouped:
                grouped[key] = {}
            grouped[key][device] = f"{mean_ms:.3f} ± {ci_ms:.3f}"

        # Build LaTeX table
        latex = [
            r"\begin{table}[h!]",
            r"\centering",
            fr"\caption{{Benchmark results for {method}.}}",
            r"\label{{tab:{method.lower()}_comparison}}",
            r"\begin{tabular}{ccrr}",
            r"\toprule",
            r"Input size & Data type & CPU (ms) & GPU (ms) \\",
            r"\midrule"
        ]

        for (sub_label, dtype), devices in grouped.items():
            cpu_mean = devices.get('cpu')
            cuda_mean = devices.get('cuda')
            latex.append(f"{sub_label} & {dtype} & {cpu_mean} & {cuda_mean} \\\\")

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        latex_tables[method] = "\n".join(latex)

    return latex_tables

# Usage
latex_tables = generate_latex_tables_by_method(results_all)
print(latex_tables["PCT"])
# print(latex_tables["Kurtosis"])
