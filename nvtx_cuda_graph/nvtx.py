import torch
import functools
import nvtx_graph_cpp  # the compiled C++ extension

from contextlib import contextmanager

@contextmanager
def _graph_captured_range(msg: str):
    """
    A context manager that, when a CUDA graph capture is active on
    torch.cuda.current_stream(), pushes an NVTX range at entry and
    pops it on exit.
    """
    nvtx_graph_cpp.nvtx_graph_push_range(msg)
    try:
        yield
    finally:
        nvtx_graph_cpp.nvtx_graph_pop_range()

def graph_captured_nvtx_range(msg: str):
    """
    A decorator factory that wraps any function so that, when that
    function is called, it implicitly pushes/pops an NVTX range around
    its body (assuming we're currently capturing a CUDA graph).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _graph_captured_range(msg):
                return func(*args, **kwargs)
        return wrapper
    return decorator