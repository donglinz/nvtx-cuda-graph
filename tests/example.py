import torch
from nvtx_cuda_graph import graph_captured_nvtx_range
from torch.cuda import CUDAGraph
from torch.cuda import Stream, graph_pool_handle


@graph_captured_nvtx_range("dummy_kernel")
def dummy_kernel(x):
    return x * 2.0

if __name__ == "__main__":

    device = torch.device("cuda")
    x = torch.randn(1024, device=device)
    y = dummy_kernel(x)

    stream = Stream(device=device)
    handle = graph_pool_handle()
    graph = CUDAGraph()

    with torch.cuda.graph(graph, pool=handle, stream=stream):

        y = dummy_kernel(x)
        y = dummy_kernel(x)
        y = dummy_kernel(x)
        
    
    torch.cuda.profiler.start()

    for _ in range(10):

        with torch.cuda.nvtx.range("graph_replay"):

            graph.replay()

    torch.cuda.profiler.stop()