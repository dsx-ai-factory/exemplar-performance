#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

# Copy an installation made by install.sh to hosts without internet access.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROG="$(basename "$0")"
source "${SCRIPT_DIR}/versions.sh"

BUNDLE_FORMAT=1
VENV_SUBDIR=venvs/deepseek-v3-baremetal
LLMB_INSTALL=${LLMB_INSTALL:-$HOME/llmb}
OUTPUT=
COMMAND=
BUNDLE=

usage() {
    cat << EOF
Usage:
  ${PROG} create [--llmb-install DIR] [--output FILE]
  ${PROG} install BUNDLE [--llmb-install DIR]

create   On a host with internet access, pack an installation made by install.sh --self-contained
         into one archive and write FILE.sha256 next to it. Build the installation on a Vera Rubin
         tray with the same Ubuntu release as the hosts that will run it.
install  On a host without internet access, unpack a bundle into a new install root, adapt its
         paths, and verify it with check_env.py. Needs neither root nor network access.

Options:
  --llmb-install DIR   Install root (default: \$LLMB_INSTALL or \$HOME/llmb)
  --output FILE        Bundle to write (default: ./llmb-deepseek-v3-<arch>-ubuntu<release>-<gpu>.tar.gz)
  -h, --help           Show this help and exit
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
        --llmb-install | --output)
            [[ $# -ge 2 && -n $2 ]] || die "option $1 requires a value"
            ;;&
        --llmb-install) LLMB_INSTALL=$2 ;;
        --output) OUTPUT=$2 ;;
        -h | --help)
            usage
            exit 0
            ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *)
            if [[ -z ${COMMAND} ]]; then
                COMMAND=$1
            elif [[ ${COMMAND} == install && -z ${BUNDLE} ]]; then
                BUNDLE=$1
            else
                die "unexpected argument '$1' (see --help)"
            fi
            shift
            continue
            ;;
    esac
    shift 2
done

require_tools() {
    local tool missing=()
    for tool in "$@"; do
        command -v "${tool}" > /dev/null || missing+=("${tool}")
    done
    ((${#missing[@]} == 0)) || die "missing tools: ${missing[*]}"
}

# True when path $1 is $2 or inside it.
is_under() {
    [[ $1 == "$2" || $1 == "$2"/* ]]
}

# Read key=value lines from standard input into MANIFEST and check that every field is present.
declare -A MANIFEST=()
read_manifest() {
    local key value
    while IFS='=' read -r key value; do
        if [[ ${key} =~ ^[a-z_]+$ ]]; then
            MANIFEST[${key}]=${value}
        fi
    done
    for key in prefix cpu_arch os_version glibc cuda_home cuda_archs gpu_caps gpu_family; do
        [[ -n ${MANIFEST[${key}]:-} ]] || die "the installation manifest has no ${key} entry"
    done
}

compressor() {
    if command -v pigz > /dev/null; then echo pigz; else echo gzip; fi
}

# Kilobytes used by the install root, excluding what create leaves out of the bundle.
bundle_size_kb() {
    local prefix=$1 total path
    shift
    total=$(du -sk "${prefix}" | cut -f1)
    for path in "$@"; do
        if [[ -e "${prefix}/${path}" ]]; then
            total=$((total - $(du -sk "${prefix}/${path}" | cut -f1)))
        fi
    done
    echo "${total}"
}

create_bundle() {
    require_tools du gzip python3 sha256sum tar
    local prefix
    prefix=$(cd "${LLMB_INSTALL}" 2> /dev/null && pwd) || die "install root ${LLMB_INSTALL} not found"
    local install_manifest="${prefix}/.llmb-install"
    [[ -f ${install_manifest} ]] \
        || die "${install_manifest} not found; run install.sh --self-contained (re-running it records an existing installation)"
    read_manifest < "${install_manifest}"
    [[ ${MANIFEST[prefix]} == "${prefix}" ]] || die "${install_manifest} describes ${MANIFEST[prefix]}, not ${prefix}"

    # A bundle may only depend on the driver and a C compiler on the target host.
    is_under "${MANIFEST[cuda_home]}" "${prefix}" \
        || die "the CUDA toolkit (${MANIFEST[cuda_home]}) is outside ${prefix}; re-run install.sh --self-contained"
    local venv="${prefix}/${VENV_SUBDIR}" python_home
    python_home=$(sed -n 's/^home = //p' "${venv}/pyvenv.cfg" 2> /dev/null)
    is_under "${python_home}" "${prefix}" \
        || die "the virtual environment uses Python from ${python_home:-an unknown location}; remove ${venv} and re-run install.sh --self-contained"
    [[ -n $(find "${prefix}/.cache/huggingface" -name config.json -print -quit 2> /dev/null) ]] \
        || die "the ${HF_MODEL_ID} configuration is not cached; re-run install.sh without --skip-hf-config"

    if [[ -z ${OUTPUT} ]]; then
        OUTPUT="${PWD}/llmb-deepseek-v3-${MANIFEST[cpu_arch]}-ubuntu${MANIFEST[os_version]}-${MANIFEST[gpu_family]}.tar.gz"
    fi
    mkdir -p "$(dirname "${OUTPUT}")"
    OUTPUT="$(cd "$(dirname "${OUTPUT}")" && pwd)/$(basename "${OUTPUT}")"
    [[ ! -e ${OUTPUT} ]] || die "${OUTPUT} already exists; remove it or pass --output"
    ! is_under "${OUTPUT}" "${prefix}" || die "the bundle cannot be written inside the install root ${prefix}"

    # Download caches, logs, results, and the APEX, DeepEP, and Transformer Engine build trees (all are
    # installed in the environment), plus the toolkit that compiled Transformer Engine.
    local excludes=(
        .cache/pip .cache/uv logs workloads/pretrain_deepseek-v3/experiments src/apex src/DeepEP src/TransformerEngine
        "cuda-${TRANSFORMER_ENGINE_CUDA_VERSION}"
    )
    local exclude_args=() path
    for path in "${excludes[@]}"; do
        exclude_args+=("--exclude=./${path}")
    done
    exclude_args+=("--exclude=./.cuda-redist.*" "--exclude=./*.partial")

    log "Measuring ${prefix}"
    local work
    work=$(mktemp -d)
    # shellcheck disable=SC2064  # Expand now: the trap runs after this function returns.
    trap "rm -rf $(printf '%q' "${work}")" EXIT
    {
        cat "${install_manifest}"
        echo "bundle_format=${BUNDLE_FORMAT}"
        echo "size_kb=$(bundle_size_kb "${prefix}" "${excludes[@]}")"
        echo "created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${work}/MANIFEST"

    log "Writing ${OUTPUT}; this takes several minutes"
    tar -C "${work}" -cf - MANIFEST \
        -C "${prefix}" "${exclude_args[@]}" --transform='s,^\./,root/,S' . \
        | python3 "${SCRIPT_DIR}/bundle_archive.py" normalize "${prefix}" \
        | "$(compressor)" > "${OUTPUT}.partial"
    mv "${OUTPUT}.partial" "${OUTPUT}"
    (cd "$(dirname "${OUTPUT}")" && sha256sum "$(basename "${OUTPUT}")" > "$(basename "${OUTPUT}").sha256")

    log "Bundle complete: ${OUTPUT} ($(du -h "${OUTPUT}" | cut -f1))"
    printf 'Copy it and %s to each host, together with this directory, then run:\n' "$(basename "${OUTPUT}").sha256"
    printf '    %q install %q --llmb-install DIR\n' "${SCRIPT_DIR}/bundle.sh" "$(basename "${OUTPUT}")"
}

check_target_host() {
    [[ $(uname -m) == "${MANIFEST[cpu_arch]}" ]] \
        || die "the bundle is for ${MANIFEST[cpu_arch]} hosts; this host is $(uname -m)"

    local glibc os_version
    glibc=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
    [[ $(printf '%s\n' "${MANIFEST[glibc]}" "${glibc}" | sort -V | head -n 1) == "${MANIFEST[glibc]}" ]] \
        || die "the bundle needs glibc ${MANIFEST[glibc]} or newer (Ubuntu ${MANIFEST[os_version]}); this host has ${glibc}"
    os_version=$(. /etc/os-release 2> /dev/null && echo "${VERSION_ID:-unknown}")
    if [[ ${os_version} != "${MANIFEST[os_version]}" ]]; then
        log "Warning: the bundle was built on Ubuntu ${MANIFEST[os_version]}; this host runs ${os_version}"
    fi

    command -v nvidia-smi > /dev/null || die "nvidia-smi not found; install the NVIDIA driver first"
    local driver
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 2> /dev/null) \
        || die "nvidia-smi could not query the GPU; check the driver installation"
    ((${driver%%.*} >= MIN_DRIVER_MAJOR)) \
        || die "driver ${driver} is too old; version ${MIN_DRIVER_MAJOR} or newer is required"

    [[ ${MANIFEST[gpu_family]} == rubin ]] || die "the bundle was built for ${MANIFEST[gpu_family]} GPUs, not Vera Rubin"
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | grep -qvx '10\.7' \
        && die "this host has GPUs other than Vera Rubin (compute capability 10.7)"

    # Triton compiles a small launcher with the host C compiler at run time.
    command -v gcc > /dev/null || command -v cc > /dev/null \
        || die "no C compiler found; Triton needs gcc at run time (install build-essential from the OS media)"
}

# Rewrite the original install root in text files (scripts, activate, pyvenv.cfg, editable-install
# hooks) and in absolute symbolic links. Compiled files contain it only in debug information.
relocate() {
    local root=$1 old=$2 new=$3
    [[ ${old} != "${new}" ]] || return 0
    log "Changing paths from ${old} to ${new}"
    local python
    python=$(find "${root}/tools/python" -path "*/bin/python${PYTHON_VERSION}" -type f -print -quit 2> /dev/null)
    [[ -n ${python} ]] || die "the bundle has no Python ${PYTHON_VERSION} interpreter under tools/python"
    { grep -rIlZF --exclude-dir=.git -e "${old}" "${root}" || true; } | "${python}" -c '
import os
import stat
import sys

old, new = (arg.encode() for arg in sys.argv[1:])
for name in filter(None, sys.stdin.buffer.read().split(b"\0")):
    mode = os.stat(name).st_mode
    os.chmod(name, mode | stat.S_IWUSR)
    with open(name, "rb") as f:
        data = f.read()
    with open(name, "wb") as f:
        f.write(data.replace(old, new))
    os.chmod(name, mode)
' "${old}" "${new}"

    local link target
    while IFS= read -r -d '' link; do
        target=$(readlink "${link}")
        if is_under "${target}" "${old}"; then
            ln -sfn "${new}${target#"${old}"}" "${link}"
        fi
    done < <(find "${root}" -type l -lname '/*' -print0)
}

install_bundle() {
    require_tools df find grep gzip mktemp python3 sha256sum tar
    [[ -n ${BUNDLE} ]] || die "install needs the bundle file (see --help)"
    [[ -f ${BUNDLE} ]] || die "bundle ${BUNDLE} not found"
    BUNDLE="$(cd "$(dirname "${BUNDLE}")" && pwd)/$(basename "${BUNDLE}")"

    if [[ -f "${BUNDLE}.sha256" ]]; then
        log "Verifying the checksum of ${BUNDLE}"
        local expected actual
        read -r expected _ < "${BUNDLE}.sha256" || true
        actual=$(sha256sum "${BUNDLE}" | cut -d' ' -f1)
        [[ ${actual} == "${expected}" ]] \
            || die "checksum mismatch; the bundle is damaged or incomplete, so copy it again"
    else
        log "Warning: ${BUNDLE}.sha256 not found; skipping the checksum check"
    fi

    python3 "${SCRIPT_DIR}/bundle_archive.py" validate "${BUNDLE}"

    # MANIFEST is the first member, so tar can stop reading early.
    local manifest
    manifest=$(tar -xzf "${BUNDLE}" --occurrence=1 -O MANIFEST 2> /dev/null || true)
    [[ -n ${manifest} ]] || die "${BUNDLE} is not a bundle created by ${PROG} create"
    read_manifest <<< "${manifest}"
    [[ ${MANIFEST[bundle_format]:-} == "${BUNDLE_FORMAT}" ]] \
        || die "${BUNDLE} has bundle format ${MANIFEST[bundle_format]:-unknown}; this script reads format ${BUNDLE_FORMAT}"
    check_target_host

    local dest
    dest=$(realpath -m "${LLMB_INSTALL}")
    if [[ -e ${dest} ]]; then
        [[ -d ${dest} && -z $(ls -A "${dest}") ]] \
            || die "${dest} is not empty; pass --llmb-install with a new directory"
    fi
    local parent=${dest}
    until [[ -d ${parent} ]]; do parent=$(dirname "${parent}"); done
    local free_kb need_kb=$((${MANIFEST[size_kb]:-0} * 11 / 10))
    free_kb=$(df -Pk "${parent}" | awk 'NR == 2 {print $4}')
    ((free_kb >= need_kb)) || die "${parent} has $((free_kb / 1048576)) GB free; the installation needs $((need_kb / 1048576)) GB"

    mkdir -p "$(dirname "${dest}")"
    local staging
    staging=$(mktemp -d "${dest}.partial.XXXXXXXXXX")
    # shellcheck disable=SC2064  # Expand now: the trap runs after this function returns.
    trap "rm -rf -- $(printf '%q' "${staging}")" EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    log "Unpacking ${BUNDLE} into ${dest}; this takes several minutes"
    tar -x --use-compress-program="$(compressor)" --no-same-owner --no-same-permissions -f "${BUNDLE}" -C "${staging}" --strip-components=1 root
    relocate "${staging}" "${MANIFEST[prefix]}" "${dest}"
    if [[ -d ${dest} ]]; then
        rmdir "${dest}"
    fi
    mv "${staging}" "${dest}"
    trap - EXIT INT TERM

    # Hugging Face lookups would wait for network timeouts; the model configuration is cached.
    local activate="${dest}/${VENV_SUBDIR}/bin/activate"
    sed -i '/^# >>> llmb: offline >>>$/,/^# <<< llmb: offline <<<$/d' "${activate}"
    cat >> "${activate}" << 'EOF'
# >>> llmb: offline >>>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# <<< llmb: offline <<<
EOF

    log "Verifying the installation"
    (
        # shellcheck source=/dev/null
        source "${activate}"
        LLMB_INSTALL=${dest} python "${SCRIPT_DIR}/check_env.py" \
            --skip-git \
            --mbridge-dir "${dest}/workloads/pretrain_deepseek-v3/Megatron-Bridge"
    )

    log "Installation complete. Run the shortest benchmark with:"
    printf '    LLMB_INSTALL=%q %q\n' "${dest}" "${SCRIPT_DIR}/launch.sh"
}

case ${COMMAND} in
    create) create_bundle ;;
    install) install_bundle ;;
    "")
        usage >&2
        exit 1
        ;;
    *) die "unknown command '${COMMAND}'; use create or install" ;;
esac
