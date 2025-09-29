import time
import torch

class TimeProfiler:
    def __init__(self, name=None, cuda_sync=True):
        self.name = name
        self.cuda_sync = cuda_sync
        self.dt = None
        if not torch.cuda.is_available():
            self.cuda_sync = False

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        end = self.time()
        elapsed = end - self.start
        self.dt = elapsed * 1000
        if self.name is not None:
            print(f"{self.name}: {self}")

    def elapsed_ms(self):
        if self.dt is None:
            raise RuntimeError("Timer has not finished. Use inside a 'with' block or after __exit__.")
        return self.dt

    def __str__(self):
        if self.dt is None:
            return "Timer not finished."
        return f"{self.dt:.2f} ms"

    def time(self):
        if self.cuda_sync:
            torch.cuda.synchronize()
        return time.perf_counter()