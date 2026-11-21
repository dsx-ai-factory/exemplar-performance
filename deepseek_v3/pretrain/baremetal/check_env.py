#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that every dependency of the DeepSeek V3 bare-metal benchmark is installed and usable."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 12)
RESULTS: list[tuple[str, bool, str]] = []
# Checks reported as SKIP rather than FAIL when they do not pass.
OPTIONAL = {"Flash Attention (optional)"}


def record(name: str, ok: bool, detail: str) -> bool:
    RESULTS.append((name, ok, detail))
    return ok


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python() -> None:
    version = sys.version_info
    record(
        "Python",
        version[:2] == REQUIRED_PYTHON,
        f"{platform.python_version()} (requires {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.x)",
    )


def check_driver() -> None:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        record("NVIDIA driver", False, "nvidia-smi not found on PATH")
        return
    query = [smi, "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"]
    try:
        output = subprocess.run(query, check=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except (subprocess.SubprocessError, OSError) as error:
        record("NVIDIA driver", False, f"nvidia-smi failed: {error}")
        return
    first_gpu = output.splitlines()[0] if output else "no GPUs reported"
    record("NVIDIA driver", bool(output), first_gpu)


def check_torch() -> bool:
    try:
        import torch
    except (ImportError, OSError) as error:
        return record("PyTorch", False, f"import failed: {error}")

    if not record("PyTorch", torch.cuda.is_available(), f"{torch.__version__} (CUDA {torch.version.cuda})"):
        return False

    count = torch.cuda.device_count()
    major, minor = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)
    record("CUDA devices", count > 0, f"{count} visible, device 0 is {device_name} (sm_{major}{minor})")
    record("BF16 support", torch.cuda.is_bf16_supported(), "required for all DeepSeek V3 recipes")

    # torch.cuda.nccl.version() is the version PyTorch was built with, not the library it loads.
    built_with = ".".join(str(part) for part in torch.cuda.nccl.version())
    wheel = package_version("nvidia-nccl-cu13") or package_version("nvidia-nccl-cu12")
    record("NCCL", True, f"{wheel} (PyTorch built with {built_with})" if wheel else built_with)

    cudnn = torch.backends.cudnn.version()
    record("cuDNN", cudnn is not None, str(cudnn))
    return True


def check_transformer_engine(run_kernel: bool) -> None:
    try:
        import transformer_engine.pytorch as te
    except (ImportError, OSError) as error:
        record("Transformer Engine", False, f"import failed: {error}")
        return

    version = package_version("transformer_engine") or getattr(te, "__version__", "unknown")
    if not run_kernel:
        record("Transformer Engine", True, version)
        return

    import torch

    try:
        layer = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
        inputs = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        layer(inputs).sum().backward()
        # Normalization kernels are compiled at run time with NVRTC.
        norm = te.RMSNorm(256, params_dtype=torch.bfloat16).cuda()
        norm(inputs).sum().backward()
        # Fused attention exercises the cuDNN sub-libraries, which a Linear layer never loads.
        qkv = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        attention = te.DotProductAttention(4, 64, qkv_format="bshd", attn_mask_type="causal")
        attention(qkv, qkv, qkv).sum().backward()
        torch.cuda.synchronize()
    except Exception as error:
        record("Transformer Engine", False, f"{version}, BF16 forward/backward failed: {error}")
        return
    record("Transformer Engine", True, f"{version}, BF16 Linear, RMSNorm, and attention forward/backward OK")


def check_torch_compile() -> None:
    """Megatron fuses elementwise kernels with torch.compile, which needs a Triton that knows the GPU."""
    import torch

    version = package_version("triton") or "unknown"

    @torch.compile
    def fused(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x) * x

    try:
        inputs = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
        fused(inputs)
        torch.cuda.synchronize()
    except Exception as error:
        first_line = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
        record("Triton (torch.compile)", False, f"{version}, compile failed: {first_line}")
        return
    record("Triton (torch.compile)", True, f"{version}, kernel compiled and ran")


def check_import(name: str, module: str, distribution: str | None = None) -> None:
    try:
        imported = importlib.import_module(module)
    except Exception as error:
        record(name, False, f"import {module} failed: {error}")
        return
    version = package_version(distribution) if distribution else None
    version = version or getattr(imported, "__version__", "installed")
    record(name, True, str(version))


def check_hybrid_ep() -> None:
    """HybridEP compiles its kernels at run time with CUDA_HOME/bin/nvcc against the toolkit headers."""
    name = "DeepEP HybridEP"
    try:
        import torch  # noqa: F401  (loads the CUDA libraries the extensions link against)

        importlib.import_module("deep_ep")
    except Exception as error:
        record(name, False, f"import failed: {error}")
        return
    version = package_version("deep_ep") or "installed"
    cuda_home = os.environ.get("CUDA_HOME", "")
    needed = ("bin/nvcc", "include/nvml.h", "lib64/libcudart.so")
    missing = [part for part in needed if not (Path(cuda_home) / part).exists()] if cuda_home else ["CUDA_HOME"]
    if missing:
        record(name, False, f"{version}; the run-time compiler needs {', '.join(missing)} (CUDA_HOME={cuda_home})")
        return
    record(name, True, f"{version}, run-time compiler {cuda_home}/bin/nvcc")


def check_mbridge_scripts(mbridge_dir: Path) -> None:
    bootstrap = mbridge_dir / "scripts" / "performance" / "bootstrap.py"
    record("Megatron-Bridge scripts", bootstrap.is_file(), str(bootstrap))


def check_tools(skip_git: bool = False) -> None:
    for tool in ("torchrun",) if skip_git else ("torchrun", "git"):
        path = shutil.which(tool)
        record(tool, path is not None, path or "not found on PATH")


def main(argv: list[str] | None = None) -> int:
    default_install = Path(os.environ.get("LLMB_INSTALL", Path.home() / "llmb"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mbridge-dir",
        type=Path,
        default=default_install / "workloads" / "pretrain_deepseek-v3" / "Megatron-Bridge",
        help="Megatron-Bridge checkout to verify (default: %(default)s).",
    )
    parser.add_argument("--no-gpu-test", action="store_true", help="Skip the GPU kernel tests.")
    parser.add_argument("--skip-git", action="store_true", help="Do not require git when verifying an offline bundle.")
    args = parser.parse_args(argv)

    check_python()
    check_driver()
    torch_ok = check_torch()
    check_transformer_engine(run_kernel=torch_ok and not args.no_gpu_test)
    if torch_ok and not args.no_gpu_test:
        check_torch_compile()
    check_import("Megatron Core", "megatron.core", "megatron-core")
    check_import("Megatron-Bridge", "megatron.bridge", "megatron-bridge")
    check_import("NeMo Run", "nemo_run", "nemo-run")
    check_import("APEX fused kernels", "fused_weight_gradient_mlp_cuda", "apex")
    check_import("Flash Attention (optional)", "flash_attn", "flash-attn")
    check_hybrid_ep()
    check_mbridge_scripts(args.mbridge_dir)
    check_tools(skip_git=args.skip_git)

    width = max(len(name) for name, _, _ in RESULTS)
    failures = 0
    for name, ok, detail in RESULTS:
        if ok:
            label = "OK"
        elif name in OPTIONAL:
            label = "SKIP"
        else:
            label = "FAIL"
            failures += 1
        print(f"[{label:>4}] {name:<{width}}  {detail}")

    if failures:
        print(f"\n{failures} required check(s) failed.", file=sys.stderr)
        return 1
    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
