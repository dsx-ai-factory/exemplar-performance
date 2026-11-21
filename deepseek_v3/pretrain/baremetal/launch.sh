#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

# Launch the DeepSeek V3 Megatron-Bridge pre-training benchmark with torchrun on bare-metal Vera Rubin
# trays installed by install.sh or bundle.sh. The supported scales come from
# ../megatron_bridge/metadata.yaml. The proxy and full-scale overrides follow the Vera Rubin
# recipe tested with the Megatron-Bridge revision in versions.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROG="$(basename "$0")"
source "${SCRIPT_DIR}/versions.sh"

usage() {
    cat << EOF
Usage: ${PROG} [options] [-- <extra Megatron-Bridge arguments>]

Run modes:
  --mode MODE             smoke | proxy | full (default: smoke)
                            smoke: scaled-down DeepSeek V3 (4 layers, hidden 1024) on 1-8 GPUs
                            proxy: reduced proxy (31 layers) at the proxy scales in metadata.yaml
                            full:  671B configuration at the scales in metadata.yaml

Precision and placement:
  --dtype DTYPE           bf16 | fp8 | nvfp4 (default: bf16)
  --gpu-type TYPE         auto | vr (default: auto; this recipe runs on Vera Rubin only)
  --dispatcher NAME       MoE token dispatcher: auto | hybridep | alltoall (default: auto, which uses the
                            recipe's HybridEP for proxy and full runs and all-to-all for smoke runs)
  --no-numa-bind          Do not bind each rank to the CPUs and memory of its GPU's NUMA node
                            (bound by default with 4 GPUs per tray in the default order)

Distributed launch (torchrun):
  --nproc-per-node N      GPUs per node (default: all visible GPUs)
  --nnodes N              Number of nodes (default: 1)
  --node-rank N           Rank of this node, 0..nnodes-1 (default: 0)
  --master-addr HOST      Address of node rank 0 (default: 127.0.0.1)
  --master-port PORT      Rendezvous port on node rank 0 (default: 29500)

Run control:
  --max-steps N           Training steps (default: 10 for smoke, 50 otherwise)
  --llmb-install DIR      Install root created by install.sh (default: \$LLMB_INSTALL or \$HOME/llmb)
  --log-dir DIR           Experiment output directory (default: auto-generated under \$LLMB_INSTALL)
  --offline               Do not contact Hugging Face; use the model config cached by install.sh
  --nsys                  Capture an Nsight Systems profile of rank 0 (steps 5-6 for smoke, 45-50 otherwise)
  --dry-run               Print the resolved command and write the training config without training
  -h, --help              Show this help and exit

Every option can also be set with the matching environment variable, for example
MODE=proxy, DTYPE=fp8, DISPATCHER=alltoall, NUMA_BIND=false, NNODES=2, MAX_STEPS=20.

Examples:
  ${PROG}                                # shortest run: 10-step smoke test on local GPUs
  ${PROG} --mode proxy --dtype fp8       # 4-GPU proxy on one tray
  ${PROG} --mode full --dtype fp8 --nnodes 32 --node-rank 0 --master-addr node001
EOF
}

die() {
    printf '%s: error: %s\n' "${PROG}" "$*" >&2
    exit 1
}

log() {
    printf '[%s] %s\n' "${PROG}" "$*"
}

require_value() {
    [[ $# -ge 2 && -n $2 && $2 != --* ]] || die "option $1 requires a value"
}

require_uint() {
    [[ $2 =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer (got '$2')"
}

MODE=${MODE:-smoke}
DTYPE=${DTYPE:-bf16}
GPU_TYPE=${GPU_TYPE:-auto}
DISPATCHER=${DISPATCHER:-auto}
NUMA_BIND=${NUMA_BIND:-auto}
NPROC_PER_NODE=${NPROC_PER_NODE:-auto}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
MAX_STEPS=${MAX_STEPS:-}
LLMB_INSTALL=${LLMB_INSTALL:-$HOME/llmb}
LOG_DIR=${LOG_DIR:-}
ENABLE_NSYS=${ENABLE_NSYS:-false}
DRY_RUN=${DRY_RUN:-false}
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode | --dtype | --gpu-type | --dispatcher | --nproc-per-node | --nnodes | --node-rank | --master-addr | --master-port | --max-steps | --llmb-install | --log-dir)
            require_value "$@"
            ;;&
        --mode) MODE=$2 ;;
        --dtype) DTYPE=$2 ;;
        --gpu-type) GPU_TYPE=$2 ;;
        --dispatcher) DISPATCHER=$2 ;;
        --nproc-per-node) NPROC_PER_NODE=$2 ;;
        --nnodes) NNODES=$2 ;;
        --node-rank) NODE_RANK=$2 ;;
        --master-addr) MASTER_ADDR=$2 ;;
        --master-port) MASTER_PORT=$2 ;;
        --max-steps) MAX_STEPS=$2 ;;
        --llmb-install) LLMB_INSTALL=$2 ;;
        --log-dir) LOG_DIR=$2 ;;
        --no-numa-bind | --offline | --nsys | --dry-run)
            case $1 in
                --no-numa-bind) NUMA_BIND=false ;;
                --offline) export HF_HUB_OFFLINE=1 ;;
                --nsys) ENABLE_NSYS=true ;;
                --dry-run) DRY_RUN=true ;;
            esac
            shift
            continue
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *) die "unknown option '$1' (see --help)" ;;
    esac
    shift 2
done

MODE=${MODE,,}
DTYPE=${DTYPE,,}
DISPATCHER=${DISPATCHER,,}

case ${MODE} in smoke | proxy | full) ;; *) die "--mode must be smoke, proxy, or full (got '${MODE}')" ;; esac
case ${DTYPE} in bf16 | fp8 | nvfp4) ;; *) die "--dtype must be bf16, fp8, or nvfp4 (got '${DTYPE}')" ;; esac
case ${GPU_TYPE,,} in auto | vr) ;; *) die "--gpu-type must be auto or vr (got '${GPU_TYPE}'); this recipe runs on Vera Rubin only" ;; esac
case ${DISPATCHER} in auto | hybridep | alltoall) ;; *) die "--dispatcher must be auto, hybridep, or alltoall (got '${DISPATCHER}')" ;; esac
case ${NUMA_BIND} in auto | true | false) ;; *) die "NUMA_BIND must be auto, true, or false (got '${NUMA_BIND}')" ;; esac
require_uint --nnodes "${NNODES}"
require_uint --node-rank "${NODE_RANK}"
require_uint --master-port "${MASTER_PORT}"
((NNODES >= 1)) || die "--nnodes must be at least 1"
((NODE_RANK < NNODES)) || die "--node-rank (${NODE_RANK}) must be less than --nnodes (${NNODES})"

VENV_DIR="${LLMB_INSTALL}/venvs/deepseek-v3-baremetal"
[[ -f "${VENV_DIR}/bin/activate" ]] || die "virtual environment not found at ${VENV_DIR}; run install.sh first"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

command -v nvidia-smi > /dev/null || die "nvidia-smi not found; install the NVIDIA driver first"
command -v torchrun > /dev/null || die "torchrun not found in ${VENV_DIR}"
[[ $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2> /dev/null) == 10.7 ]] \
    || die "this recipe supports Vera Rubin GPUs (compute capability 10.7) only"

if [[ ${NPROC_PER_NODE} == auto ]]; then
    if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
        IFS=',' read -ra VISIBLE <<< "${CUDA_VISIBLE_DEVICES}"
        NPROC_PER_NODE=${#VISIBLE[@]}
    else
        NPROC_PER_NODE=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)
    fi
fi
require_uint --nproc-per-node "${NPROC_PER_NODE}"
((NPROC_PER_NODE >= 1)) || die "no GPUs are visible on this node"
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
if [[ ${MODE} != smoke && ${NPROC_PER_NODE} -ne 4 ]]; then
    log "Warning: the Vera Rubin recipes are tuned for 4 GPUs per tray; this node runs ${NPROC_PER_NODE}."
fi

case ${DTYPE} in
    fp8) COMPUTE_DTYPE=fp8_mx ;;
    *) COMPUTE_DTYPE=${DTYPE} ;;
esac

contains_word() {
    [[ " $1 " == *" $2 "* ]]
}

OVERRIDES=()
HYDRA_OVERRIDES=()
SEQ_LENGTH=4096

case ${MODE} in
    smoke)
        SEQ_LENGTH=1024
        ((TOTAL_GPUS <= 8 && NNODES == 1)) || die "smoke mode runs on a single node with 1-8 GPUs (requested ${TOTAL_GPUS} GPUs on ${NNODES} nodes)"
        # Each expert-parallel rank needs at least two experts, and top-8 routing over
        # four selected groups needs at least two experts per router group.
        NUM_EXPERTS=$((TOTAL_GPUS * 2))
        while ((NUM_EXPERTS < 8)); do NUM_EXPERTS=$((NUM_EXPERTS + TOTAL_GPUS * 2)); done
        ROUTER_GROUPS=$((NUM_EXPERTS / 2 < 8 ? NUM_EXPERTS / 2 : 8))
        OVERRIDES+=(
            -tp 1 -pp 1 -cp 1 -vp None -ep "${TOTAL_GPUS}"
            -mb 1 -gb $((TOTAL_GPUS * 4))
            --seq_length "${SEQ_LENGTH}"
            --hidden_size 1024
            --num_layers 4
            --first_k_dense_replace 1
            --num_moe_experts "${NUM_EXPERTS}"
            --cuda_graph_impl none
        )
        HYDRA_OVERRIDES+=(
            model.vocab_size=32000
            model.num_attention_heads=32
            model.num_query_groups=32
            "model.moe_router_num_groups=${ROUTER_GROUPS}"
            mixed_precision.grad_reduce_in_fp32=true
            ddp.grad_reduce_in_fp32=true
            # Overlapping the parameter all-gather with the optimizer step needs
            # virtual pipeline chunks, which the single-chunk smoke model lacks.
            optimizer.overlap_param_gather_with_optimizer_step=false
            comm_overlap.overlap_param_gather_with_optimizer_step=false
        )
        ;;
    proxy)
        SCALES=$(read_metadata python3 proxy-scales fp8) || die "could not read ${METADATA_FILE}"
        contains_word "${SCALES}" "${TOTAL_GPUS}" \
            || die "proxy mode supports ${SCALES} GPUs (requested ${TOTAL_GPUS})"
        [[ ${DTYPE} == fp8 ]] || log "Warning: the published proxy configurations are validated with --dtype fp8."
        if ((TOTAL_GPUS % 72 == 0)); then
            ep=72 num_moe_experts=144
        elif ((TOTAL_GPUS == 4)); then
            # The default proxy configuration hits a Transformer Engine assertion.
            ep=4 num_moe_experts=16
        else
            ep=${TOTAL_GPUS} num_moe_experts=$((TOTAL_GPUS * 2))
        fi
        OVERRIDES+=(-tp 1 -pp 1 -vp None -ep "${ep}" -mb $((TOTAL_GPUS == 4 ? 1 : 2)))
        OVERRIDES+=(--num_layers 31 --num_moe_experts "${num_moe_experts}")
        ;;
    full)
        SCALES=$(read_metadata python3 scales "${DTYPE}") || die "could not read ${METADATA_FILE}"
        contains_word "${SCALES}" "${TOTAL_GPUS}" \
            || die "full mode with ${DTYPE} supports ${SCALES:-no} GPU counts (requested ${TOTAL_GPUS})"
        if ((TOTAL_GPUS == 128)); then
            OVERRIDES+=(-pp 4 -vp 4 -ep 32 --recompute_modules mla_up_proj)
        fi
        ;;
esac

if [[ -z ${MAX_STEPS} ]]; then
    MAX_STEPS=$([[ ${MODE} == smoke ]] && echo 10 || echo 50)
fi
require_uint --max-steps "${MAX_STEPS}"
((MAX_STEPS >= 2)) || die "--max-steps must be at least 2"

# Average the same window as llmb-run (iterations 35-44 of 50); short runs use the second half.
if ((MAX_STEPS >= 50)); then
    TIMING_START=35 TIMING_END=44
else
    TIMING_START=$((MAX_STEPS / 2 + 1)) TIMING_END=${MAX_STEPS}
fi

if [[ ${ENABLE_NSYS} == true ]]; then
    command -v nsys > /dev/null || die "--nsys requires Nsight Systems (nsys) on PATH"
    if [[ ${MODE} == smoke ]]; then
        NSYS_START=${PROFILE_START_STEP:-5} NSYS_STOP=${PROFILE_STOP_STEP:-6}
    else
        NSYS_START=${PROFILE_START_STEP:-45} NSYS_STOP=${PROFILE_STOP_STEP:-50}
    fi
    ((NSYS_STOP <= MAX_STEPS)) || die "profiling stop step ${NSYS_STOP} exceeds --max-steps ${MAX_STEPS}"
    HYDRA_OVERRIDES+=(
        profiling.use_nsys_profiler=true
        "profiling.profile_step_start=${NSYS_START}"
        "profiling.profile_step_end=${NSYS_STOP}"
        "profiling.profile_ranks=[0]"
    )
fi

LLMB_WORKLOAD="${LLMB_INSTALL}/workloads/pretrain_deepseek-v3"
MBRIDGE_DIR="${LLMB_WORKLOAD}/Megatron-Bridge"
BOOTSTRAP="${MBRIDGE_DIR}/scripts/performance/bootstrap.py"
[[ -f ${BOOTSTRAP} ]] || die "Megatron-Bridge performance scripts not found at ${BOOTSTRAP}; run install.sh"
INSTALLED_COMMIT=$(git -C "${MBRIDGE_DIR}" rev-parse HEAD 2> /dev/null || true)
[[ -z ${INSTALLED_COMMIT} || ${INSTALLED_COMMIT} == "${VALIDATED_MBRIDGE_COMMIT}" ]] \
    || die "${MBRIDGE_DIR} is at ${INSTALLED_COMMIT:0:12}, not the validated ${VALIDATED_MBRIDGE_COMMIT:0:12}; re-run install.sh"

EXPERIMENT_NAME="pretrain_deepseek_v3_671b_${DTYPE}_${TOTAL_GPUS}gpu_vr_${MODE}"
if [[ -z ${LOG_DIR} ]]; then
    LOG_DIR="${LLMB_WORKLOAD}/experiments/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/log-${EXPERIMENT_NAME}_node${NODE_RANK}.out"

# Put the installed checkout ahead of any Megatron-Bridge copy in the environment.
export PYTHONPATH="${MBRIDGE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${HF_HOME:-${LLMB_INSTALL}/.cache/huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-False}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# The published proxy and full configurations dispatch MoE tokens with the recipe's HybridEP.
if [[ ${DISPATCHER} == auto ]]; then
    DISPATCHER=$([[ ${MODE} == smoke ]] && echo alltoall || echo recipe)
fi
case ${DISPATCHER} in
    alltoall)
        OVERRIDES+=(--moe_flex_dispatcher_backend None)
        [[ ${MODE} == smoke ]] \
            || log "Warning: all-to-all MoE dispatcher selected; results are not comparable to published numbers."
        ;;
    hybridep | recipe)
        [[ ${DISPATCHER} == hybridep ]] && OVERRIDES+=(--moe_flex_dispatcher_backend hybridep)
        # HybridEP compiles its kernels with CUDA_HOME/bin/nvcc. The compiled kernels are scratch files
        # of each process, so keep them off shared file systems.
        [[ -x ${CUDA_HOME:-}/bin/nvcc ]] \
            || die "HybridEP compiles its kernels with \$CUDA_HOME/bin/nvcc; set CUDA_HOME to the toolkit install.sh used (CUDA_HOME='${CUDA_HOME:-}')"
        export HYBRID_EP_CACHE_DIR=${HYBRID_EP_CACHE_DIR:-${TMPDIR:-/tmp}/${USER:-llmb}-hybrid-ep}
        ;;
esac

TRAIN_ARGS=(
    --model_family_name deepseek
    --model_recipe_name deepseek_v3
    --gpu vr200
    --compute_dtype "${COMPUTE_DTYPE}"
    --num_gpus "${TOTAL_GPUS}"
    --max_steps "${MAX_STEPS}"
    --data mock
    --log_dir "${LOG_DIR}"
    # Without this, Megatron-Bridge writes the resolved config under /nemo_run, which only root can create.
    --save_config_filepath "${LOG_DIR}/ConfigContainer.yaml"
    "${OVERRIDES[@]}"
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    ${HYDRA_OVERRIDES[@]+"${HYDRA_OVERRIDES[@]}"}
)

for ((i = 0; i < ${#TRAIN_ARGS[@]}; i++)); do
    case ${TRAIN_ARGS[i]} in
        --seq_length | -sl) SEQ_LENGTH=${TRAIN_ARGS[i + 1]:-} ;;
        --seq_length=* | -sl=*) SEQ_LENGTH=${TRAIN_ARGS[i]#*=} ;;
    esac
done
[[ ${SEQ_LENGTH} =~ ^[1-9][0-9]*$ ]] || die "--seq_length must be a positive integer (got '${SEQ_LENGTH}')"

# Vera Rubin trays have two GPUs per CPU socket; the published runs bind each rank (LOCAL_RANK / 2)
# to its socket, which holds only for the default GPU order.
if [[ ${NUMA_BIND} == auto ]]; then
    NUMA_BIND=false
    if ((NPROC_PER_NODE == 4)); then
        if [[ ${CUDA_VISIBLE_DEVICES-0,1,2,3} != 0,1,2,3 ]]; then
            log "Warning: CUDA_VISIBLE_DEVICES changes the default GPU order; automatic NUMA binding disabled."
        elif command -v numactl > /dev/null; then
            NUMA_BIND=true
        else
            log "Warning: numactl not found; automatic NUMA binding disabled. Install numactl for the published binding."
        fi
    fi
fi
RANK_CMD=(python3 "${BOOTSTRAP}")
if [[ ${NUMA_BIND} == true ]]; then
    command -v numactl > /dev/null || die "NUMA binding needs numactl (apt install numactl), or pass --no-numa-bind"
    # shellcheck disable=SC2016  # Each rank expands LOCAL_RANK, which torchrun sets.
    RANK_CMD=(bash -c 'node=$((LOCAL_RANK / 2)); exec numactl --cpunodebind="${node}" --membind="${node}" "$@"' numa-bind "${RANK_CMD[@]}")
fi

TORCHRUN_CMD=(
    torchrun
    --nnodes "${NNODES}"
    --nproc-per-node "${NPROC_PER_NODE}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
    --no-python
    "${RANK_CMD[@]}"
    "${TRAIN_ARGS[@]}"
)

# The published runs lock the memory clock at its maximum (nvidia-smi -lmc, which needs root).
read -r MEM_CLOCK MAX_MEM_CLOCK < <(nvidia-smi --query-gpu=clocks.mem,clocks.max.mem --format=csv,noheader,nounits -i 0 2> /dev/null | tr -d ',') || true
if [[ -n ${MAX_MEM_CLOCK:-} && ${MEM_CLOCK:-} != "${MAX_MEM_CLOCK}" ]]; then
    log "Warning: memory clock is ${MEM_CLOCK} MHz, below the ${MAX_MEM_CLOCK} MHz of the published runs; lock it with: sudo nvidia-smi -lmc ${MAX_MEM_CLOCK},${MAX_MEM_CLOCK}"
fi

if [[ ${ENABLE_NSYS} == true ]]; then
    TORCHRUN_CMD=(
        nsys profile -s none -t "cuda,nvtx"
        -o "${LOG_DIR}/profile_${EXPERIMENT_NAME}_node${NODE_RANK}"
        --force-overwrite true
        --capture-range=cudaProfilerApi --capture-range-end=stop
        "${TORCHRUN_CMD[@]}"
    )
fi

log "Mode: ${MODE} | Precision: ${COMPUTE_DTYPE} | Megatron-Bridge ${VALIDATED_MBRIDGE_COMMIT:0:12}"
log "GPUs: ${TOTAL_GPUS} (${NNODES} node(s) x ${NPROC_PER_NODE}) | Steps: ${MAX_STEPS} | NUMA binding: ${NUMA_BIND}"
log "MoE dispatcher: ${DISPATCHER/recipe/recipe (HybridEP)}"
log "Log file: ${LOG_FILE}"

if [[ ${DRY_RUN} == true ]]; then
    printf '[%s] Command:' "${PROG}"
    printf ' %q' "${TORCHRUN_CMD[@]}"
    printf '\n'
    WORLD_SIZE=${TOTAL_GPUS} RANK=0 python3 "${MBRIDGE_DIR}/scripts/performance/run_script.py" \
        "${TRAIN_ARGS[@]}" --dryrun \
        > "${LOG_DIR}/dryrun.log" 2>&1 || {
        tail -n 40 "${LOG_DIR}/dryrun.log" >&2
        die "dry run failed; full output in ${LOG_DIR}/dryrun.log"
    }
    log "Dry run succeeded; resolved configuration: ${LOG_DIR}/ConfigContainer.yaml"
    exit 0
fi

{
    printf 'Command:'
    printf ' %q' "${TORCHRUN_CMD[@]}"
    printf '\n'
} > "${LOG_FILE}"

"${TORCHRUN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

# The last rank prints the iteration lines and rank 0 the throughput lines, so summarize on the
# last node and add node 0's log when the log directory is shared.
if ((NODE_RANK == NNODES - 1)); then
    PARSE_LOGS=("${LOG_FILE}")
    NODE0_LOG="${LOG_DIR}/log-${EXPERIMENT_NAME}_node0.out"
    if [[ ${NODE0_LOG} != "${LOG_FILE}" && -f ${NODE0_LOG} ]]; then
        PARSE_LOGS+=("${NODE0_LOG}")
    fi
    python3 "${SCRIPT_DIR}/parse_results.py" "${PARSE_LOGS[@]}" \
        --start "${TIMING_START}" --end "${TIMING_END}" \
        --seq-length "${SEQ_LENGTH}" \
        --json "${LOG_DIR}/metrics.json"
fi
