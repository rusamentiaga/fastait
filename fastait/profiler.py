import time
import torch

class TimeProfiler:
    """
    Context manager for profiling execution time of code blocks.
    Args:
        name (str, optional): Name to identify the timed block in logs. Defaults to None.
        cuda_sync (bool, optional): Whether to synchronize CUDA before timing. Defaults to True.
        logger (callable, optional): Function to log the timing result. Defaults to
    """
    def __init__(self, name=None, cuda_sync=True, logger=print):
        self.name = name
        self.dt_ms = None
        self.logger = logger
        self.cuda_sync = cuda_sync and torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end = self.time()
        elapsed = end - self.start
        self.dt_ms = elapsed * 1000
        if self.name is not None:
            self.logger(f"{self.name}: {self}")
        return False

    def elapsed_ms(self):
        if self.dt_ms is None:
            raise RuntimeError("Timer has not finished. Use inside a 'with' block or after __exit__.")
        return self.dt_ms

    def __str__(self):
        return f"{self.elapsed_ms():.2f} ms"

    def time(self):
        if self.cuda_sync:
            torch.cuda.synchronize()
        return time.perf_counter()