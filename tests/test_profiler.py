import time
import numpy as np
from fastait.profiler import TimeProfiler

def test_time_profiler():
    print("Testing TimeProfiler...")

    with TimeProfiler() as prof:
        time.sleep(0.1)
    assert np.allclose(prof.elapsed_ms(), 100, atol=20)

