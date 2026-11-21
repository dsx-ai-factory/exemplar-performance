# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash disable=SC2034
# bash-regression:skip-guard

# Pinned software versions for the DeepSeek V3 bare-metal workflow on Vera Rubin, sourced by
# install.sh, bundle.sh, and launch.sh. These pins reflect the original Vera Rubin validation.
# The supported GPU counts come from ../megatron_bridge/metadata.yaml.

METADATA_FILE=${METADATA_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/megatron_bridge/metadata.yaml}
MBRIDGE_REPO=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
VALIDATED_MBRIDGE_COMMIT=4a0b7d318269dadbc83acc4c27ca823a22cca6fe

MCORE_REPO=https://github.com/NVIDIA/Megatron-LM.git
MCORE_COMMIT=9be4a57455a08f617596c7a0683bc811a57c336c
# PyTorch 2.14 creates a generator's CUDA graph capture state lazily, on the stream and memory pool
# outside the capture, which breaks the full-iteration capture of the FP8 recipe on every pipeline
# stage after the first. The patch carries the upstream fixes b4d72b79 (#7348) and d6316eb4
# (#7885), which create those states at the start of the capture, rebased onto this commit.
MCORE_PATCHES=(patches/megatron-lm-full-iteration-rng.patch)

APEX_REPO=https://github.com/NVIDIA/apex.git
APEX_COMMIT=575968bc1f9127ccd61003a681472a83af4ff1a1

DEEPEP_REPO=https://github.com/deepseek-ai/DeepEP.git
DEEPEP_COMMIT=17cfb817bccec3a9c247013360cc550c2bac441e
# PyTorch 2.14 headers require C++20, and this DeepEP commit builds HybridEP with -std=c++17.
DEEPEP_BUILD_FIXES=(f725d29699f5bda9ba789456bb9579af69844685)

PYTHON_VERSION=3.12
MIN_DRIVER_MAJOR=580
TORCH_VERSION=2.14.1
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132

# From Transformer Engine commit d1e9c334 on, the fused grouped MLP of the FP8 recipes, which set
# NVTE_ALLOW_NONDETERMINISTIC_ALGO=0, raises an error, so the stack builds the commit before it. Its
# kernels that use architecture-specific features need sm_107a, which nvcc supports from CUDA 13.4,
# so it is compiled with CUDA 13.4 and runs on CUDA 13.2.
TRANSFORMER_ENGINE_REPO=https://github.com/NVIDIA/TransformerEngine.git
TRANSFORMER_ENGINE_VERSION=2.20.0.dev0
TRANSFORMER_ENGINE_COMMIT=80a89adc272887fba3999a6a4b434347327d2c74
TRANSFORMER_ENGINE_CUDA_VERSION=13.4.3
# Later upstream changes, each as "<commit> <path>...": its NVRTC kernel cache unloads modules from a
# static destructor, which can crash the process at exit; its CUDA graph capture retains warm-up
# outputs and temporaries, which the full BF16 run cannot fit; and its grouped MXFP8 quantization
# updates TMA descriptors without a release-acquire fence, so FP8 runs intermittently fail with an
# illegal memory access. The 7ba2f9f1 change builds on the 5f6105b9 one.
TRANSFORMER_ENGINE_BUILD_FIXES=(
    "dd628d3a11b7b52908ea5b7bba7d38d739df6f55 transformer_engine/common/util/cuda_driver.h transformer_engine/common/util/rtc.cpp"
    "29229b5a030d1672bb79af61a735caf42ce1effc transformer_engine/pytorch/graph.py"
    "5f6105b900778068144f1b24e25a2e84066983cb transformer_engine/common/cast/core/grouped_tma.cuh transformer_engine/common/cast/fp8_blockwise/group_quantize_fp8_blockwise.cuh transformer_engine/common/cast/mxfp8/group_quantize_mxfp8.cuh transformer_engine/common/cast/mxfp8/group_scaled_swiglu_mxfp8.cuh transformer_engine/common/cast/nvfp4/specialized/quantize_transpose_nvfp4_tuned_1D.cuh"
    "7ba2f9f17a9016c8b991c0702870b49a7d0f35f8 transformer_engine/common/cast/core/grouped_tma.cuh transformer_engine/common/cast/mxfp8/group_dequantize_mxfp8.cuh transformer_engine/common/cast/mxfp8/group_quantize_mxfp8.cuh"
)

# cuDNN 9.25 and older build no fused attention plans, Triton 3.8 and older cannot compile for
# sm_107a, and NVRTC before CUDA 13.4 rejects sm_107 when Transformer Engine compiles its normalization
# kernels, so the toolkit carries the CUDA 13.4 NVRTC.
CUBLAS_VERSION=13.8.0.4
NVSHMEM_VERSION=3.8.0
NCCL_VERSION=2.31.2
CUDNN_VERSION=9.26.1.1
TRITON_VERSION=3.9.0
NVRTC_VERSION=13.4.92

# CUDA toolkit unpacked from the redistributable archives under the install root. HybridEP compiles its
# kernels at run time against these headers, including nvml.h.
CUDA_REDIST_VERSION=13.2.2
CUDA_REDIST_COMPONENTS=(
    cuda_cccl cuda_crt cuda_cudart cuda_culibos cuda_cupti cuda_nvcc cuda_nvml_dev cuda_nvrtc cuda_nvtx
    cuda_profiler_api libcublas libcufft libcurand libcusolver libcusparse libnvjitlink libnvvm
)
# Components replaced with the build from another CUDA release, as name:release pairs.
CUDA_REDIST_OVERRIDES=(cuda_nvrtc:13.4.3)

UV_VERSION=0.13.6
# uv-aarch64-unknown-linux-gnu.tar.gz
UV_SHA256=229e194255c7958d86d56e8790d969a6f74736592133f0dfd0076f754c642c1d

HF_MODEL_ID=deepseek-ai/DeepSeek-V3

# Print a Vera Rubin value from metadata.yaml with PYTHON (needs PyYAML):
#   scales DTYPE | proxy-scales DTYPE
read_metadata() {
    local python=$1
    shift
    [[ -f ${METADATA_FILE} ]] \
        || {
            printf 'metadata file %s not found; keep this directory next to ../megatron_bridge/ or set METADATA_FILE\n' "${METADATA_FILE}" >&2
            return 1
        }
    "${python}" - "${METADATA_FILE}" "$@" << 'EOF'
import sys

import yaml

path, query, *args = sys.argv[1:]
with open(path) as f:
    metadata = yaml.safe_load(f)
configs = metadata["run"]["gpu_configs"]["vr"]["model_configs"]
dtype = next((c["dtypes"].get(args[0]) for c in configs if c["model_size"] == "671b"), None) or {}
print(" ".join(str(s) for s in dtype.get("scales" if query == "scales" else "proxy_scales", [])))
EOF
}
