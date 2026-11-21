# DeepSeek V3 Pre-training on Bare-Metal Vera Rubin

**Experimental alpha.** This workflow uses the bare-metal stack originally validated with Megatron-Bridge `4a0b7d31`. The [Slurm recipe](../megatron_bridge/README.md) is the reference for this preview and uses a newer stack.

This directory runs the Vera Rubin DeepSeek V3 pre-training benchmark directly on Ubuntu trays, without containers, Slurm, or Kubernetes. `install.sh` builds the software stack of the published VR results, and `launch.sh` starts training with `torchrun`.

- `versions.sh` pins Megatron-Bridge and the rest of the software stack. The supported GPU counts come from the [Slurm recipe metadata](../megatron_bridge/metadata.yaml).
- The proxy and full-scale parallelism settings follow the original Vera Rubin validation.
- `parse_results.py` summarizes training logs and writes `metrics.json`, importing `llmb-run`'s pre-training log parser module (`cli/llmb-run/src/llmb_run/pretrain_log_parser.py`) for step time and TFLOPS.

Only pre-training is covered. For reinforcement learning, see [RL README](../../rl/README.md).

## Requirements

| Item       | Requirement                                                                                                                                                                |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hosts      | Vera Rubin trays (4 GPUs, compute capability 10.7), Ubuntu 24.04 or 26.04 LTS, aarch64                                                                                     |
| Driver     | 580 or newer                                                                                                                                                               |
| Disk       | About 40 GB for the install root, on a file system that all trays of a multi-tray run share                                                                                |
| Memory     | 32 GB of host memory or more while Transformer Engine compiles                                                                                                             |
| Network    | Internet access during installation (PyPI, GitHub, NVIDIA, Hugging Face), or use a bundle                                                                                  |
| Multi-tray | InfiniBand or RoCE with working RDMA, and the same `LLMB_INSTALL` path on every tray                                                                                       |
| Packages   | `build-essential`, `curl`, `git`, and `xz-utils` (plus `numactl` for NUMA binding and `libibverbs-dev` / `rdma-core` for multi-domain HybridEP when using `--skip-system`) |

## Quick Start

```bash
export LLMB_INSTALL=/shared/llmb             # install root, on a shared file system for multi-tray runs
./install.sh --skip-system                   # about 40 minutes, half of it the Transformer Engine build
./launch.sh                                  # 10-step smoke test on the local GPUs
./launch.sh --mode proxy --dtype fp8         # 4-GPU proxy
```

Run `install.sh` on a Vera Rubin tray, for example inside a Slurm job (`srun -N 1 --exclusive -t 02:00:00 ./install.sh --skip-system`). Without `--skip-system`, it also installs the build tools, `numactl`, and RDMA libraries with `apt`. It is safe to re-run; completed steps are skipped. The installer log is in `$LLMB_INSTALL/logs`, and `source "$LLMB_INSTALL/venvs/deepseek-v3-baremetal/bin/activate" && python check_env.py` verifies an installation at any time.

| Install option       | Description                                                                         |
| -------------------- | ----------------------------------------------------------------------------------- |
| `--llmb-install DIR` | Install root (default: `$LLMB_INSTALL` or `$HOME/llmb`); not an `llmb-install` root |
| `--install-driver`   | Installs the NVIDIA datacenter driver and Fabric Manager, then exits (reboot after) |
| `--driver-branch N`  | Driver branch for `--install-driver` (default: `580`)                               |
| `--skip-system`      | Skips `apt`, so no root is needed                                                   |
| `--self-contained`   | Installs Python under the install root, as `bundle.sh` requires                     |
| `--skip-hf-config`   | Does not cache the Hugging Face model configuration                                 |
| `--max-jobs N`       | Parallel compile jobs; lower it if the build runs out of memory                     |

## Running the Benchmark

| Mode              | Model                           | GPUs                                                   |
| ----------------- | ------------------------------- | ------------------------------------------------------ |
| `smoke` (default) | 4 layers, hidden size 1024      | 1-8 on one tray                                        |
| `proxy`           | 31 layers, reduced expert count | `fp8` `proxy_scales` in `metadata.yaml` (4 to 576)     |
| `full`            | Full 671B DeepSeek V3           | `scales` in `metadata.yaml` for the dtype (128 to 512) |

Proxy and full runs dispatch MoE tokens with HybridEP and, with 4 GPUs per tray in the default order and `numactl` installed, bind each rank to the NUMA node of its GPU, as the published runs do. `--dtype` is `bf16`, `fp8` (MXFP8), or `nvfp4`. The averaging window is iterations 35-44 of 50, as in `llmb-run`. The published runs lock the GPU memory clock at its maximum with `sudo nvidia-smi -lmc`; `launch.sh` does not change clocks but warns when the memory clock is below its maximum. `./launch.sh --help` lists every option.

For a multi-tray run, start `launch.sh` once per tray with the same arguments, a shared `--log-dir`, and that tray's rank, for example with `mpirun` or `srun`:

```bash
# On every tray; NODE_RANK is 0 on the first tray, 1 on the next, and so on.
./launch.sh --mode proxy --dtype fp8 --nnodes 8 --nproc-per-node 4 \
    --node-rank "${NODE_RANK}" --master-addr tray001 --log-dir /shared/runs/proxy32
```

The last tray combines its log with tray 0's log in `--log-dir`, prints the summary, and writes `metrics.json`. Single-tray runs write `log-<experiment>_node0.out` to `$LLMB_INSTALL/workloads/pretrain_deepseek-v3/experiments/` unless `--log-dir` is given. To summarize logs again, pass the last tray's log and then tray 0's log:

```bash
python parse_results.py log-<experiment>_node7.out log-<experiment>_node0.out --seq-length 4096
```

## Hosts Without Internet Access

Install once on a connected Vera Rubin tray with `--self-contained`, pack it, and unpack it on each offline tray. Offline trays need the driver, `gcc`, `python3`, `numactl`, and (for multi-tray runs) the RDMA userspace libraries (`ibverbs-providers`).

```bash
./install.sh --skip-system --self-contained --llmb-install "$HOME/llmb-stage"   # connected tray
./bundle.sh create --llmb-install "$HOME/llmb-stage"                            # writes a 6.8 GB bundle
./bundle.sh install llmb-deepseek-v3-aarch64-ubuntu26.04-rubin.tar.gz --llmb-install "$HOME/llmb"
```

Copy the bundle, its `.sha256` file, and this repository to the offline trays. `bundle.sh install` checks the host, rejects archive members or links that point outside the bundle, unpacks it into a new directory, rewrites its paths, and runs `check_env.py`.

## Validation

This directory was validated on Vera Rubin trays (Ubuntu 26.04, driver 620, no root, install root on Lustre) with these runs, which all completed successfully:

| Run                                         | GPUs |
| ------------------------------------------- | ---- |
| `--mode proxy --dtype fp8`, one tray        | 4    |
| `--mode proxy --dtype fp8`, 8 trays, 1 rack | 32   |
| `--mode full --dtype bf16`, 32 trays        | 128  |
| `--mode full --dtype fp8`, 32 trays         | 128  |

A clean installation took about 35 minutes, and `bundle.sh` was checked end to end on a tray without network access.

Run `python -m pytest tests` with PyYAML and pytest installed. The Slurm launcher parity tests in `tests/test_recipe_alignment.py` are skipped when `../megatron_bridge/metadata.yaml` pins a different Megatron-Bridge commit than `versions.sh`.

## Pinned Versions

`versions.sh` records the software versions used for the original Vera Rubin validation and explains each choice:

| Component          | Version                                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| Megatron-Bridge    | `4a0b7d31`, from `versions.sh`                                                                    |
| Megatron Core      | `9be4a574`, with `patches/megatron-lm-full-iteration-rng.patch` (upstream `b4d72b79`, `d6316eb4`) |
| Transformer Engine | 2.20.0.dev0 at `80a89adc`, built with CUDA 13.4.3, with upstream fixes for exit, memory, and FP8  |
| DeepEP (HybridEP)  | `17cfb817`, with the `setup.py` change of `f725d296`                                              |
| PyTorch            | 2.14.1 (CUDA 13.2 wheels), with CUDA 13.2.2 and NVRTC 13.4                                        |
| Libraries          | cuBLAS 13.8.0.4, NCCL 2.31.2, NVSHMEM 3.8.0, cuDNN 9.26.1.1, Triton 3.9.0                         |
