#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

# Install the DeepSeek V3 Megatron-Bridge benchmark on a bare-metal Vera Rubin tray (Ubuntu 24.04 or 26.04).
# Safe to re-run: completed steps are detected and skipped.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROG="$(basename "$0")"
source "${SCRIPT_DIR}/versions.sh"

LLMB_INSTALL=${LLMB_INSTALL:-$HOME/llmb}
INSTALL_DRIVER=false
DRIVER_BRANCH=${DRIVER_BRANCH:-${MIN_DRIVER_MAJOR}}
SKIP_SYSTEM=false
SELF_CONTAINED=false
SKIP_HF_CONFIG=false
MAX_JOBS=${MAX_JOBS:-}

usage() {
    cat << EOF
Usage: ${PROG} [options]

Installs, on a Vera Rubin tray, the stack of the published VR results: system packages, the CUDA
${CUDA_REDIST_VERSION} toolkit under the install root, a Python ${PYTHON_VERSION} virtual environment with PyTorch
${TORCH_VERSION}, Transformer Engine ${TRANSFORMER_ENGINE_VERSION} (commit ${TRANSFORMER_ENGINE_COMMIT:0:8}, built with CUDA ${TRANSFORMER_ENGINE_CUDA_VERSION}),
APEX, DeepEP (HybridEP), Megatron Core, and Megatron-Bridge ${VALIDATED_MBRIDGE_COMMIT:0:8},
then verifies the result with check_env.py.

Options:
  --llmb-install DIR     Install root (default: \$LLMB_INSTALL or \$HOME/llmb)
  --install-driver       Install the NVIDIA datacenter driver and Fabric Manager, then exit (reboot required)
  --driver-branch N      Driver branch for --install-driver (default: ${DRIVER_BRANCH})
  --skip-system          Skip apt packages (no root needed); Python ${PYTHON_VERSION} is installed in user space when missing
  --self-contained       Always install Python under the install root, so that bundle.sh can copy the
                         installation to hosts without internet access
  --skip-hf-config       Do not pre-download the ${HF_MODEL_ID} config.json (online runs only)
  --max-jobs N           Parallel compile jobs for Transformer Engine (default: based on CPU and memory)
  -h, --help             Show this help and exit

Environment variables with the same names (LLMB_INSTALL, DRIVER_BRANCH, MAX_JOBS) are also honored.
HF_TOKEN is used for the config download when set. HYBRID_EP_MULTINODE=0 or 1 overrides the detection of the
RDMA headers that HybridEP needs to dispatch beyond one NVLink domain.
EOF
}

die() {
    printf '%s: error: %s\n' "${PROG}" "$*" >&2
    exit 1
}

log() {
    printf '\n[%s] %s\n' "${PROG}" "$*"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --llmb-install | --driver-branch | --max-jobs)
            [[ $# -ge 2 && -n $2 ]] || die "option $1 requires a value"
            ;;&
        --llmb-install) LLMB_INSTALL=$2 ;;
        --driver-branch) DRIVER_BRANCH=$2 ;;
        --max-jobs) MAX_JOBS=$2 ;;
        --install-driver | --skip-system | --self-contained | --skip-hf-config | -h | --help)
            case $1 in
                --install-driver) INSTALL_DRIVER=true ;;
                --skip-system) SKIP_SYSTEM=true ;;
                --self-contained) SELF_CONTAINED=true ;;
                --skip-hf-config) SKIP_HF_CONFIG=true ;;
                *)
                    usage
                    exit 0
                    ;;
            esac
            shift
            continue
            ;;
        *) die "unknown option '$1' (see --help)" ;;
    esac
    shift 2
done

[[ ${DRIVER_BRANCH} =~ ^[0-9]+$ ]] || die "--driver-branch must be a number (got '${DRIVER_BRANCH}')"
if [[ -n ${MAX_JOBS} && ! ${MAX_JOBS} =~ ^[1-9][0-9]*$ ]]; then
    die "--max-jobs must be a positive integer (got '${MAX_JOBS}')"
fi

if ((EUID == 0)); then
    SUDO=()
else
    SUDO=(sudo)
fi

require_root() {
    ((EUID == 0)) || command -v sudo > /dev/null || die "run as root, install sudo, or pass --skip-system"
}

apt_install() {
    "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}

check_os() {
    [[ -r /etc/os-release ]] || die "/etc/os-release not found"
    local id
    # shellcheck disable=SC2153  # ID and VERSION_ID come from /etc/os-release.
    read -r id OS_VERSION < <(. /etc/os-release && echo "${ID} ${VERSION_ID}")
    [[ ${id} == ubuntu && ${OS_VERSION} =~ ^(24|26)\.04$ ]] \
        || die "Ubuntu 24.04 or 26.04 LTS is required (found ${id} ${OS_VERSION})"
    CPU_ARCH=$(uname -m)
    [[ ${CPU_ARCH} == aarch64 ]] || die "Vera Rubin trays are aarch64; this host is ${CPU_ARCH}"
}

# Download a file and verify its SHA-256 checksum.
fetch_verified() {
    local url=$1 sha256=$2 dest=$3
    curl -fsSL --retry 5 -o "${dest}" "${url}" || die "download failed: ${url}"
    printf '%s  %s\n' "${sha256}" "${dest}" | sha256sum --check --quiet - || die "checksum mismatch: ${url}"
}

check_build_tools() {
    local tool missing=()
    for tool in curl g++ gcc git make sha256sum tar xz; do
        command -v "${tool}" > /dev/null || missing+=("${tool}")
    done
    ((${#missing[@]} == 0)) || die "missing build tools: ${missing[*]} (ask an administrator for build-essential, curl, git, and xz-utils)"
}

install_driver() {
    log "Installing NVIDIA datacenter driver branch ${DRIVER_BRANCH} and Fabric Manager"
    "${SUDO[@]}" apt-get update
    apt_install ubuntu-drivers-common
    "${SUDO[@]}" ubuntu-drivers install --gpgpu "nvidia:${DRIVER_BRANCH}-server"
    apt_install "nvidia-utils-${DRIVER_BRANCH}-server" "nvidia-fabricmanager-${DRIVER_BRANCH}"
    "${SUDO[@]}" systemctl enable nvidia-fabricmanager
    log "Driver installed. Reboot, confirm 'nvidia-smi' works, then re-run ${PROG} without --install-driver."
}

check_driver() {
    command -v nvidia-smi > /dev/null || die "nvidia-smi not found; install the driver first (${PROG} --install-driver)"
    local version
    version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 2> /dev/null) \
        || die "nvidia-smi could not query the GPU; check the driver installation"
    ((${version%%.*} >= MIN_DRIVER_MAJOR)) \
        || die "driver ${version} is too old; version ${MIN_DRIVER_MAJOR} or newer is required"
    log "NVIDIA driver ${version} detected"
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | grep -qvx '10\.7' \
        && die "this recipe supports Vera Rubin GPUs (compute capability 10.7) only"
    return 0
}

install_system_packages() {
    local packages=(
        build-essential ca-certificates cmake curl git gnupg ninja-build numactl libnuma-dev pciutils pkg-config rsync
        xz-utils rdma-core ibverbs-providers ibverbs-utils libibverbs-dev
    )
    # Ubuntu 26.04 does not package Python 3.12; select_python installs it in user space there.
    if [[ ${OS_VERSION} == 24.04 ]]; then
        packages+=("python${PYTHON_VERSION}" "python${PYTHON_VERSION}-dev" "python${PYTHON_VERSION}-venv")
    fi
    log "Installing system packages"
    "${SUDO[@]}" apt-get update
    apt_install "${packages[@]}"
}

# Arguments: destination, CUDA release, and name:release overrides of single components.
install_cuda_redist() {
    local dest=$1 version=$2
    shift 2
    local -A releases=()
    local name override release
    for name in "${CUDA_REDIST_COMPONENTS[@]}"; do
        releases[${name}]=${version}
    done
    for override in "$@"; do
        releases[${override%%:*}]=${override#*:}
    done
    local spec
    spec=$(for name in "${!releases[@]}"; do echo "${name} ${releases[${name}]}"; done | sort)

    local marker="${dest}/.llmb-redist"
    if [[ -f ${marker} && $(< "${marker}") == "${spec}" ]]; then
        log "CUDA ${version} components already installed at ${dest}"
        return
    fi
    log "Installing CUDA ${version} components into ${dest} (no root required)"
    local base=https://developer.download.nvidia.com/compute/cuda/redist
    local staging="${dest}.partial" work
    work=$(mktemp -d "${LLMB_INSTALL}/.cuda-redist.XXXXXX")
    for release in $(printf '%s\n' "${releases[@]}" | sort -u); do
        curl -fsSL --retry 5 -o "${work}/redistrib_${release}.json" "${base}/redistrib_${release}.json" \
            || die "could not download the CUDA ${release} manifest"
    done
    command -v python3 > /dev/null || die "python3 is required to read the CUDA manifest"
    # shellcheck disable=SC2086  # Each line of spec is a component name and its release.
    python3 - "${work}" linux-sbsa ${spec} > "${work}/components" << 'EOF'
import json
import sys

# Later releases publish these components under new names with the same archive layout.
RENAMED = {"cuda_cccl": "cccl"}

work, platform, *pairs = sys.argv[1:]
for name, release in zip(pairs[::2], pairs[1::2]):
    with open(f"{work}/redistrib_{release}.json") as f:
        manifest = json.load(f)
    entry = manifest.get(name, manifest.get(RENAMED.get(name), {})).get(platform)
    if entry is None:
        sys.exit(f"CUDA {release} has no {platform} archive of {name}")
    print(name, entry["relative_path"], entry["sha256"])
EOF

    rm -rf "${staging}"
    mkdir -p "${staging}"
    local path sha256
    while read -r name path sha256; do
        log "Downloading ${name}"
        fetch_verified "${base}/${path}" "${sha256}" "${work}/archive.tar.xz"
        tar -xJf "${work}/archive.tar.xz" -C "${staging}" --strip-components=1
    done < "${work}/components"
    rm -rf "${work}"
    # PyTorch's extension builder looks for libraries in lib64; the archives use lib.
    ln -sfn lib "${staging}/lib64"
    "${staging}/bin/nvcc" --version > /dev/null || die "nvcc from the CUDA ${version} archives does not run"
    printf '%s\n' "${spec}" > "${staging}/.llmb-redist"
    rm -rf "${dest}"
    mv "${staging}" "${dest}"
}

default_max_jobs() {
    local cpus mem_gb
    cpus=$(nproc)
    mem_gb=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
    # Each Transformer Engine compile job can use up to about 4 GB of memory.
    local by_mem=$((mem_gb / 4 > 0 ? mem_gb / 4 : 1))
    echo $((cpus < by_mem ? cpus : by_mem))
}

# nvcc ${CUDA_REDIST_VERSION} has no sm_107 target, so APEX gets the highest older 10.x architecture,
# whose binaries compute capability 10.7 runs.
default_cuda_archs() {
    local supported
    supported=$("${CUDA_HOME}/bin/nvcc" --list-gpu-arch | sed -n 's/^compute_\([0-9]*\)$/\1/p')
    if grep -qx 107 <<< "${supported}"; then
        echo 107
        return
    fi
    awk 'int($1 / 10) == 10 && $1 < 107' <<< "${supported}" | sort -n | tail -n 1 | grep . \
        || die "nvcc in ${CUDA_HOME} cannot build for compute capability 10.x"
}

# Use the system python3.12 when it can build extensions; otherwise install a standalone
# Python ${PYTHON_VERSION} with a checksum-verified uv under the install root.
select_python() {
    PYTHON_BIN=$(command -v "python${PYTHON_VERSION}" || true)
    if [[ ${SELF_CONTAINED} == false && -n ${PYTHON_BIN} ]] && "${PYTHON_BIN}" -c '
import ensurepip, os, sysconfig, venv
raise SystemExit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' 2> /dev/null; then
        log "Using ${PYTHON_BIN}"
        return
    fi

    local tools="${LLMB_INSTALL}/tools"
    local uv="${tools}/bin/uv"
    if [[ ! -x ${uv} ]]; then
        # shellcheck disable=SC2153  # UV_VERSION comes from versions.sh.
        log "Installing uv ${UV_VERSION} into ${tools}/bin"
        local archive="${LLMB_INSTALL}/.uv.tar.gz"
        fetch_verified "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-aarch64-unknown-linux-gnu.tar.gz" \
            "${UV_SHA256}" "${archive}"
        mkdir -p "${tools}/bin"
        tar -xzf "${archive}" -C "${tools}/bin" --strip-components=1
        rm -f "${archive}"
    fi
    export UV_PYTHON_INSTALL_DIR="${tools}/python" UV_CACHE_DIR="${LLMB_INSTALL}/.cache/uv"
    log "Installing a standalone Python ${PYTHON_VERSION} into ${UV_PYTHON_INSTALL_DIR}"
    "${uv}" python install --quiet "${PYTHON_VERSION}"
    PYTHON_BIN=$("${uv}" python find --managed-python "${PYTHON_VERSION}")
    log "Using ${PYTHON_BIN}"
}

clone_at_commit() {
    local repo=$1 commit=$2 dest=$3
    if [[ ! -d "${dest}/.git" ]]; then
        log "Cloning ${repo} into ${dest}"
        mkdir -p "$(dirname "${dest}")"
        git clone --quiet --filter=blob:none "${repo}" "${dest}"
    fi
    if [[ $(git -C "${dest}" rev-parse HEAD) != "${commit}" ]]; then
        git -C "${dest}" fetch --quiet origin "${commit}" 2> /dev/null || git -C "${dest}" fetch --quiet origin
        git -C "${dest}" -c advice.detachedHead=false checkout --quiet "${commit}"
    fi
    log "$(basename "${dest}") at ${commit:0:12}"
}

apply_mcore_patches() {
    local patch
    for patch in "${MCORE_PATCHES[@]}"; do
        if git -C "${MCORE_DIR}" apply --reverse --check "${SCRIPT_DIR}/${patch}" 2> /dev/null; then
            log "Megatron Core already carries ${patch##*/}"
        else
            log "Applying ${patch##*/} to Megatron Core"
            git -C "${MCORE_DIR}" apply "${SCRIPT_DIR}/${patch}"
        fi
    done
}

create_venv() {
    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        log "Creating virtual environment ${VENV_DIR}"
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    # The activate script of an earlier installation sets CUDA_HOME; the build needs this toolkit.
    export CUDA_HOME
    # Megatron-Bridge declares setuptools<80 for its build backend. The cmake and ninja wheels cover
    # hosts where they could not be installed with apt; PyYAML reads metadata.yaml.
    python -m pip install --quiet --upgrade pip "setuptools<80" wheel pybind11 cmake ninja pyyaml==6.0.3
}

# Build Transformer Engine at TRANSFORMER_ENGINE_COMMIT. Transformer Engine adds the architecture-specific
# targets (100a and 107a) to the family head 100 itself. Arguments: compile jobs and the cuDNN and NCCL
# wheel directories.
install_transformer_engine_source() {
    local jobs=$1 cudnn_dir=$2 nccl_dir=$3 version
    local cuda=${LLMB_INSTALL}/cuda-${TRANSFORMER_ENGINE_CUDA_VERSION} archs=100
    # The version names only the commit, so the marker also records the fixes, toolkit, and architectures.
    local marker="${TE_DIR}/.llmb-build" build
    build=$(printf '%s\n' "${TRANSFORMER_ENGINE_COMMIT}" "${TRANSFORMER_ENGINE_BUILD_FIXES[@]}" "${cuda}" "${archs}")
    version=$(python -c 'import importlib.metadata as m; print(m.version("transformer_engine"))' 2> /dev/null || true)
    if [[ ${version} == "${TRANSFORMER_ENGINE_VERSION}+${TRANSFORMER_ENGINE_COMMIT:0:7}"* && -f ${marker} && $(< "${marker}") == "${build}" ]]; then
        log "Transformer Engine ${version} already installed"
        return
    fi
    install_cuda_redist "${cuda}" "${TRANSFORMER_ENGINE_CUDA_VERSION}"
    [[ ! -d "${TE_DIR}/.git" ]] || git -C "${TE_DIR}" checkout --quiet -- .
    clone_at_commit "${TRANSFORMER_ENGINE_REPO}" "${TRANSFORMER_ENGINE_COMMIT}" "${TE_DIR}"
    git -C "${TE_DIR}" submodule update --quiet --init --recursive
    local fix parts
    for fix in "${TRANSFORMER_ENGINE_BUILD_FIXES[@]}"; do
        read -r -a parts <<< "${fix}"
        log "Applying the ${parts[*]:1} changes of Transformer Engine ${parts[0]:0:12}"
        git -C "${TE_DIR}" cat-file -e "${parts[0]}^{commit}" 2> /dev/null || git -C "${TE_DIR}" fetch --quiet origin "${parts[0]}"
        git -C "${TE_DIR}" show "${parts[0]}" -- "${parts[@]:1}" | git -C "${TE_DIR}" apply
    done
    # CMake rejects a build tree configured for another compiler or architecture list.
    rm -rf "${TE_DIR}/build"
    # A release splits into transformer_engine, transformer_engine_cu13, and transformer_engine_torch;
    # a source build is one transformer_engine package, so remove a release before building.
    python -m pip uninstall --quiet --yes transformer_engine transformer_engine_cu12 transformer_engine_cu13 \
        transformer_engine_torch 2> /dev/null || true
    python -m pip install --quiet "nvidia-cudnn-frontend>=1.28.0"
    log "Building Transformer Engine ${TRANSFORMER_ENGINE_VERSION} at ${TRANSFORMER_ENGINE_COMMIT:0:12} with ${cuda}; this takes up to an hour"
    # The NCCL EP library is for Transformer Engine's own expert-parallel dispatcher, which this workload does not use.
    env PATH="${cuda}/bin:${PATH}" CUDA_HOME="${cuda}" CUDA_PATH="${cuda}" CUDACXX="${cuda}/bin/nvcc" \
        MAX_JOBS="${jobs}" NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="${archs}" NVTE_WITH_NCCL_EP=0 \
        TORCH_CUDA_ARCH_LIST=10.0 CUDNN_PATH="${cudnn_dir}" \
        CPATH="${cudnn_dir}/include:${nccl_dir}/include:${cuda}/include/cccl${CPATH:+:${CPATH}}" \
        LIBRARY_PATH="${cudnn_dir}/lib:${nccl_dir}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
        python -m pip install --no-build-isolation "${TE_DIR}"
    printf '%s\n' "${build}" > "${marker}"
}

install_python_stack() {
    log "Installing PyTorch ${TORCH_VERSION} (${TORCH_INDEX_URL##*/})"
    python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"

    BUILD_ARCHS=$(default_cuda_archs)
    local jobs=${MAX_JOBS:-$(default_max_jobs)}
    # CUDA extensions compile against the cuDNN and NCCL headers from the wheels PyTorch installed.
    # The host compiler also needs the CUDA 13 CCCL headers, which nvcc finds on its own.
    local cudnn_dir nccl_dir
    cudnn_dir=$(python -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])')
    nccl_dir=$(python -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0])')
    local build_env=(
        PATH="${CUDA_HOME}/bin:${PATH}"
        MAX_JOBS="${jobs}"
        CUDNN_PATH="${cudnn_dir}"
        CPATH="${cudnn_dir}/include:${nccl_dir}/include:${CUDA_HOME}/include/cccl${CPATH:+:${CPATH}}"
        LIBRARY_PATH="${cudnn_dir}/lib:${nccl_dir}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    )

    install_transformer_engine_source "${jobs}" "${cudnn_dir}" "${nccl_dir}"

    if python -c 'import torch, fused_weight_gradient_mlp_cuda' 2> /dev/null; then
        log "APEX CUDA extensions already installed"
    else
        clone_at_commit "${APEX_REPO}" "${APEX_COMMIT}" "${APEX_DIR}"
        log "Building APEX C++ and CUDA extensions (sm_${BUILD_ARCHS})"
        env "${build_env[@]}" TORCH_CUDA_ARCH_LIST="${BUILD_ARCHS:0:-1}.${BUILD_ARCHS: -1}" APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
            python -m pip install --no-build-isolation "${APEX_DIR}"
    fi

    # Megatron Core builds in an isolated environment because it needs setuptools>=80.
    log "Installing Megatron Core"
    python -m pip install -e "${MCORE_DIR}"

    # Megatron-Bridge declares multimodal, export, and logging extras that pre-training does not use.
    # pip reports them as missing; requirements.txt holds the packages this benchmark needs.
    log "Installing Megatron-Bridge and benchmark requirements"
    python -m pip install --no-build-isolation --no-deps -e "${MBRIDGE_DIR}"
    python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

    # Installing Transformer Engine reverts cuBLAS to PyTorch's pin, so the pinned libraries go last.
    local pins=(
        "nvidia-cublas==${CUBLAS_VERSION}"
        "nvidia-nvshmem-cu13==${NVSHMEM_VERSION}"
        "nvidia-nccl-cu13==${NCCL_VERSION}"
        "nvidia-cudnn-cu13==${CUDNN_VERSION}"
        "triton==${TRITON_VERSION}"
        "nvidia-cuda-nvrtc==${NVRTC_VERSION}"
    )
    log "Installing ${pins[*]}"
    python -m pip install --no-deps "${pins[@]}"

    install_deepep "${build_env[@]}"
}

# Dispatching beyond one NVLink domain uses the GPU-initiated RDMA path, which needs the verbs and mlx5
# development files. DeepSeek V3 runs expert parallelism of at most 72 ranks, inside one NVL72 domain.
hybrid_ep_multinode() {
    if [[ -n ${HYBRID_EP_MULTINODE:-} ]]; then
        echo "${HYBRID_EP_MULTINODE}"
        return
    fi
    local libdir
    libdir=/usr/lib/$(gcc -dumpmachine)
    if [[ -f /usr/include/infiniband/mlx5dv.h && -f /usr/include/infiniband/verbs.h &&
        -e ${libdir}/libmlx5.so && -e ${libdir}/libibverbs.so ]]; then
        echo 1
    else
        echo 0
    fi
}

# Like the NeMo images, DeepEP builds sm_100, which Vera Rubin runs; HybridEP compiles its kernels for the
# same target at run time. Arguments: the build environment.
install_deepep() {
    local multinode marker="${VENV_DIR}/.llmb-deepep-build" fingerprint
    multinode=$(hybrid_ep_multinode)
    fingerprint=$(printf '%s\n' "commit=${DEEPEP_COMMIT}" "hybrid_ep_multinode=${multinode}" "build_fixes=${DEEPEP_BUILD_FIXES[*]}")
    if [[ -f ${marker} && $(< "${marker}") == "${fingerprint}" ]] && python - "${DEEPEP_COMMIT:0:7}" << 'EOF' 2> /dev/null; then
import sys
from importlib.metadata import version

import torch  # noqa: F401  (loads the CUDA libraries the extensions link against)
import deep_ep_cpp  # noqa: F401
import hybrid_ep_cpp  # noqa: F401

sys.exit(not version("deep_ep").endswith("+" + sys.argv[1]))
EOF
        log "DeepEP ${DEEPEP_COMMIT:0:12} already installed"
        return
    fi
    rm -f "${marker}"
    if [[ -d "${DEEPEP_DIR}/.git" ]]; then
        git -C "${DEEPEP_DIR}" checkout --quiet -- setup.py
    fi
    clone_at_commit "${DEEPEP_REPO}" "${DEEPEP_COMMIT}" "${DEEPEP_DIR}"
    local fix
    for fix in "${DEEPEP_BUILD_FIXES[@]}"; do
        log "Applying the setup.py change of DeepEP ${fix:0:12}"
        git -C "${DEEPEP_DIR}" show "${fix}" -- setup.py | git -C "${DEEPEP_DIR}" apply
    done
    if [[ ${multinode} == 1 ]]; then
        log "Building DeepEP with HybridEP RDMA support; this also builds its NCCL submodule"
    else
        log "Building DeepEP with HybridEP for single NVLink domains; install libibverbs-dev for RDMA support"
    fi
    # An empty RDMA_CORE_HOME uses the system headers and libraries, so the JIT has no install-root paths.
    env "$@" TORCH_CUDA_ARCH_LIST=10.0 HYBRID_EP_MULTINODE="${multinode}" RDMA_CORE_HOME= \
        python -m pip install --no-build-isolation --no-deps "${DEEPEP_DIR}"
    printf '%s\n' "${fingerprint}" > "${marker}"
}

# Hosts with a system cuDNN (for example DGX OS) otherwise mix it with the cuDNN wheel PyTorch
# uses: Transformer Engine loads the system libcudnn, and the wheel's sub-libraries resolve from
# /lib, so mismatched versions fail with undefined symbols or CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED.
prefer_wheel_cudnn() {
    log "Pointing CUDNN_PATH and LD_LIBRARY_PATH at the cuDNN wheel in ${VENV_DIR}/bin/activate"
    set_activate_block "prefer the cuDNN wheel" << EOF
for _llmb_cudnn in "\${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/cudnn; do
    [ -d "\${_llmb_cudnn}/lib" ] || continue
    export CUDNN_PATH="\${_llmb_cudnn}"
    case ":\${LD_LIBRARY_PATH:-}:" in
        *":\${_llmb_cudnn}/lib:"*) ;;
        *) export LD_LIBRARY_PATH="\${_llmb_cudnn}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" ;;
    esac
done
unset _llmb_cudnn
EOF
}

# Run-time compilers find the same toolkit the build used: Transformer Engine loads NVRTC from
# CUDA_HOME, and HybridEP runs CUDA_HOME/bin/nvcc.
export_cuda_home() {
    log "Setting CUDA_HOME=${CUDA_HOME} in ${VENV_DIR}/bin/activate"
    set_activate_block "CUDA toolkit" << EOF
_llmb_cuda=$(printf '%q' "${CUDA_HOME}")
export CUDA_HOME="\${_llmb_cuda}"
case ":\${PATH}:" in
    *":\${_llmb_cuda}/bin:"*) ;;
    *) export PATH="\${_llmb_cuda}/bin:\${PATH}" ;;
esac
unset _llmb_cuda
EOF
}

# Replace a marked block at the end of the activate script with standard input, then re-source it.
set_activate_block() {
    local activate="${VENV_DIR}/bin/activate"
    local begin="# >>> llmb: $1 >>>" end="# <<< llmb: $1 <<<"
    sed -i "\|^${begin}\$|,\|^${end}\$|d" "${activate}"
    {
        printf '%s\n' "${begin}"
        cat
        printf '%s\n' "${end}"
    } >> "${activate}"
    # shellcheck source=/dev/null
    source "${activate}"
}

download_hf_config() {
    log "Caching ${HF_MODEL_ID} config.json in ${HF_HOME}"
    HF_HOME="${HF_HOME}" python - "${HF_MODEL_ID}" << 'EOF'
import sys

from huggingface_hub import hf_hub_download

print(hf_hub_download(repo_id=sys.argv[1], filename="config.json"))
EOF
}

# Describe the installation for bundle.sh, which copies it to hosts without internet access.
write_install_manifest() {
    cat > "${LLMB_INSTALL}/.llmb-install.partial" << EOF
prefix=${LLMB_INSTALL}
cpu_arch=${CPU_ARCH}
os_version=${OS_VERSION}
glibc=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
cuda_home=${CUDA_HOME}
cuda_archs=${BUILD_ARCHS}
gpu_caps=107
gpu_family=rubin
EOF
    mv "${LLMB_INSTALL}/.llmb-install.partial" "${LLMB_INSTALL}/.llmb-install"
}

[[ ! -e ${LLMB_INSTALL}/cluster_config.yaml ]] \
    || die "${LLMB_INSTALL} is managed by llmb-install (cluster_config.yaml exists); choose a separate LLMB_INSTALL for the bare-metal workflow"

check_os

if [[ ${INSTALL_DRIVER} == true ]]; then
    require_root
    install_driver
    exit 0
fi

LLMB_INSTALL=$(mkdir -p "${LLMB_INSTALL}" && cd "${LLMB_INSTALL}" && pwd)
LOG_FILE="${LLMB_INSTALL}/logs/install_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-120} PIP_RETRIES=${PIP_RETRIES:-10}
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PROGRESS_BAR=off
# Keep several gigabytes of wheels out of small home-directory quotas on shared clusters.
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${LLMB_INSTALL}/.cache/pip}
VENV_DIR="${LLMB_INSTALL}/venvs/deepseek-v3-baremetal"
MCORE_DIR="${LLMB_INSTALL}/src/Megatron-LM"
APEX_DIR="${LLMB_INSTALL}/src/apex"
DEEPEP_DIR="${LLMB_INSTALL}/src/DeepEP"
TE_DIR="${LLMB_INSTALL}/src/TransformerEngine"
MBRIDGE_DIR="${LLMB_INSTALL}/workloads/pretrain_deepseek-v3/Megatron-Bridge"
HF_HOME="${LLMB_INSTALL}/.cache/huggingface"
CUDA_HOME=${LLMB_INSTALL}/cuda-${CUDA_REDIST_VERSION}

log "Installing DeepSeek V3 bare-metal benchmark into ${LLMB_INSTALL} (log: ${LOG_FILE})"
check_driver

if [[ ${SKIP_SYSTEM} == false ]]; then
    require_root
    install_system_packages
fi
check_build_tools
select_python
create_venv
install_cuda_redist "${CUDA_HOME}" "${CUDA_REDIST_VERSION}" "${CUDA_REDIST_OVERRIDES[@]}"

clone_at_commit "${MCORE_REPO}" "${MCORE_COMMIT}" "${MCORE_DIR}"
apply_mcore_patches
clone_at_commit "${MBRIDGE_REPO}" "${VALIDATED_MBRIDGE_COMMIT}" "${MBRIDGE_DIR}"
install_python_stack
prefer_wheel_cudnn
export_cuda_home

if [[ ${SKIP_HF_CONFIG} == false ]]; then
    download_hf_config
fi

log "Verifying the installation"
python "${SCRIPT_DIR}/check_env.py" --mbridge-dir "${MBRIDGE_DIR}"
write_install_manifest

log "Installation complete. Run the shortest benchmark with:"
printf '    LLMB_INSTALL=%q %q\n' "${LLMB_INSTALL}" "${SCRIPT_DIR}/launch.sh"
if [[ ${SELF_CONTAINED} == true ]]; then
    printf '\nTo copy it to hosts without internet access:\n    %q create --llmb-install %q\n' \
        "${SCRIPT_DIR}/bundle.sh" "${LLMB_INSTALL}"
fi
