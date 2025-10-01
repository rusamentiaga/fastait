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
from natsort import natsorted
import numpy as np
import pandas as pd
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

def generate_csv_from_results(results, csv_path="benchmark_results.csv"):
    """
    Generate a single CSV with all benchmark results (all methods).
    Each row = (method, input_size, dtype), with one set of columns per device.
    Sorted by method, then input_size (natural order).
    """
    grouped = {}
    devices_seen = set()

    for t in results:
        method = t.label
        sub_label = t.sub_label
        device = t.description.split()[0]   # e.g. "cpu", "cuda"
        dtype = t.description.split()[1]
        devices_seen.add(device)

        times_s = np.array(t.raw_times) / t.number_per_run
        n = len(times_s)
        mean_s = np.mean(times_s)
        std_s = np.std(times_s, ddof=1)

        # 95% confidence interval
        ci = stats.t.ppf(0.975, df=n-1) * std_s / np.sqrt(n)

        # Convert to ms
        mean_ms = mean_s * 1000
        ci_ms = ci * 1000

        key = (method, sub_label, dtype)
        if key not in grouped:
            grouped[key] = {"method": method, "input_size": sub_label, "dtype": dtype}

        grouped[key][f"{device}_mean_ms"] = round(mean_ms, 3)
        grouped[key][f"{device}_ci95_ms"] = round(ci_ms, 3)

    # Convert to DataFrame
    df = pd.DataFrame(grouped.values())

    # Ensure all device columns exist
    for device in devices_seen:
        for suffix in ["mean_ms", "ci95_ms"]:
            col = f"{device}_{suffix}"
            if col not in df.columns:
                df[col] = np.nan

    # Build ordered column list
    cols = ["method", "input_size", "dtype"]
    for device in sorted(devices_seen):  # keep devices in alphabetic order
        cols.extend([f"{device}_mean_ms", f"{device}_ci95_ms"])
    df = df.reindex(columns=cols)

    # Natural sort by method then input_size
    df = df.iloc[natsorted(df.index, key=lambda i: (df.loc[i, "method"], df.loc[i, "input_size"]))].reset_index(drop=True)

    # Save to CSV
    df.to_csv(csv_path, index=False)
    return df

df_results = generate_csv_from_results(results_all, csv_path="benchmark_results.csv")
print("Benchmark results saved to benchmark_results.csv")


