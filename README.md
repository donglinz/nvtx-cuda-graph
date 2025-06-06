# nvtx-cuda-graph

An enhanced NVTX extension that lets you insert NVTX push/pop ranges not only normally but also into a CUDA Graph capture. The range push/pop will automatically decorate your nsys report during graph replay.

Usage:

```
from nvtx_cuda_graph import graph_captured_nvtx_range

@graph_captured_nvtx_range("dummy_kernel")
def dummy_kernel(x):
    return x * 2.0
```

```graph_captured_nvtx_range``` will insert nvtx range node if under graph capturing or mark nvtx range normally if not under capturing.

*Graph capture awared nvtx will significantly increase your runtime overhead so this repo is for debugging purpose only.*
