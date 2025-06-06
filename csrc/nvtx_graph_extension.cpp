#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <cassert>

static void NVTXPushCallback(void* userData) {
    const char* name = static_cast<const char*>(userData);
    nvtxRangePushA(name);
}

static void NVTXPopCallback(void*) {
    nvtxRangePop();
}

static cudaStream_t get_current_cuda_stream() {
    auto cs = at::cuda::getCurrentCUDAStream();
    return cs.stream();
}

void nvtx_graph_push_range(const std::string& message) {
    cudaStream_t stream = get_current_cuda_stream();

    cudaGraph_t capturing_graph;
    cudaStreamCaptureStatus capture_status;
    const cudaGraphNode_t *deps = nullptr;
    size_t num_deps = 0;

    cudaError_t err = cudaStreamGetCaptureInfo_v2(
        stream,
        &capture_status,
        nullptr,
        &capturing_graph,
        &deps,
        &num_deps
    );
    TORCH_CHECK(err == cudaSuccess, "cudaStreamGetCaptureInfo_v2 failed");
    TORCH_CHECK(
        capture_status == cudaStreamCaptureStatusActive || capture_status == cudaStreamCaptureStatusNone,
        "nvtx_graph_push_range: invalid capture status"
    );

    if (capture_status == cudaStreamCaptureStatusNone) {
        nvtxRangePushA(message.c_str());
        return;
    }

    char* name_copy = strdup(message.c_str());

    cudaGraphNode_t push_node;
    cudaHostNodeParams push_params = {};
    push_params.fn = &NVTXPushCallback;
    push_params.userData = static_cast<void*>(name_copy);

    // Add the host node to the capturing graph
    err = cudaGraphAddHostNode(
        &push_node,
        capturing_graph,
        deps,
        num_deps,
        &push_params
    );
    TORCH_CHECK(err == cudaSuccess, "cudaGraphAddHostNode (push) failed");

    // Make subsequent nodes depend on this host node
    err = cudaStreamUpdateCaptureDependencies(
        stream,
        &push_node,
        1,
        cudaStreamAddCaptureDependencies
    );
    TORCH_CHECK(err == cudaSuccess, "cudaStreamUpdateCaptureDependencies (push) failed");
}

void nvtx_graph_pop_range() {
    cudaStream_t stream = get_current_cuda_stream();

    cudaGraph_t capturing_graph;
    cudaStreamCaptureStatus capture_status;
    const cudaGraphNode_t *deps = nullptr;
    size_t num_deps = 0;

    // Query capture status and any dependencies
    cudaError_t err = cudaStreamGetCaptureInfo_v2(
        stream,
        &capture_status,
        nullptr,
        &capturing_graph,
        &deps,
        &num_deps
    );
    TORCH_CHECK(err == cudaSuccess, "cudaStreamGetCaptureInfo_v2 failed");
    TORCH_CHECK(
        capture_status == cudaStreamCaptureStatusActive || capture_status == cudaStreamCaptureStatusNone,
        "nvtx_graph_pop_range: invalid capture status"
    );

    if (capture_status == cudaStreamCaptureStatusNone) {
        nvtxRangePop();
        return;
    }

    cudaGraphNode_t pop_node;
    cudaHostNodeParams pop_params = {};
    pop_params.fn = &NVTXPopCallback;
    pop_params.userData = nullptr;

    err = cudaGraphAddHostNode(
        &pop_node,
        capturing_graph,
        deps,
        num_deps,
        &pop_params
    );
    TORCH_CHECK(err == cudaSuccess, "cudaGraphAddHostNode (pop) failed");

    // Make subsequent nodes depend on this host node
    err = cudaStreamUpdateCaptureDependencies(
        stream,
        &pop_node,
        1,
        cudaStreamAddCaptureDependencies
    );
    TORCH_CHECK(err == cudaSuccess, "cudaStreamUpdateCaptureDependencies (pop) failed");
}

PYBIND11_MODULE(nvtx_graph_cpp, m) {
    m.def(
        "nvtx_graph_push_range",
        &nvtx_graph_push_range,
        "Push an NVTX range into the current CUDA graph"
    );
    m.def(
        "nvtx_graph_pop_range",
        &nvtx_graph_pop_range,
        "Pop an NVTX range in the current CUDA graph"
    );
}