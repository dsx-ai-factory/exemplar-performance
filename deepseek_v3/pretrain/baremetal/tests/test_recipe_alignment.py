# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that launch.sh runs the Vera Rubin configurations of ../megatron_bridge.

Both launchers run with stub binaries, and the parallelism arguments they pass to Megatron-Bridge are
compared for every scale in metadata.yaml.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

BAREMETAL = Path(__file__).resolve().parents[1]
SLURM = BAREMETAL.parent / "megatron_bridge"
METADATA = yaml.safe_load((SLURM / "metadata.yaml").read_text())
VR_CONFIG = METADATA["run"]["gpu_configs"]["vr"]["model_configs"][0]["dtypes"]
PINNED_COMMIT = METADATA["repositories"]["by_gpu"]["vr"]["megatron_bridge"]["commit"]
VALIDATED_COMMIT = re.search(
    r"^VALIDATED_MBRIDGE_COMMIT=(\w+)$", (BAREMETAL / "versions.sh").read_text(), re.MULTILINE
).group(1)

COMPARED = {
    "-tp", "-pp", "-vp", "-ep", "-mb", "-E", "--num_layers", "--num_moe_experts", "--recompute_modules",
    "--cuda_graph_impl", "--compute_dtype", "--gpu", "--num_gpus", "--pipeline_model_parallel_layout",
}  # fmt: skip
CAPTURED = "CAPTURED"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt", reason="needs Linux with bash")
requires_matching_slurm_pin = pytest.mark.skipif(
    PINNED_COMMIT != VALIDATED_COMMIT,
    reason=f"Slurm uses Megatron-Bridge {PINNED_COMMIT[:12]} and bare metal uses {VALIDATED_COMMIT[:12]}",
)


def _write_exec(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture(scope="module")
def stubs(tmp_path_factory) -> Path:
    """An install root and a bin directory whose python3 prints the Megatron-Bridge arguments."""
    root = tmp_path_factory.mktemp("llmb")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    _write_exec(
        bin_dir / "python3",
        f"""#!/usr/bin/env bash
case $1 in
    *run_script.py | *setup_experiment.py) shift; echo {CAPTURED}; printf '%s\\n' "$@" ;;
    *) exec {sys.executable} "$@" ;;
esac
""",
    )
    _write_exec(
        bin_dir / "nvidia-smi",
        """#!/usr/bin/env bash
case $* in
    *compute_cap*) echo 10.7 ;;
    *index*) printf '0\\n1\\n2\\n3\\n' ;;
    *clocks*) echo '3000, 3000' ;;
esac
""",
    )
    _write_exec(bin_dir / "torchrun", "#!/usr/bin/env bash\n")
    _write_exec(bin_dir / "nvcc", "#!/usr/bin/env bash\n")
    venv = root / "venvs" / "deepseek-v3-baremetal" / "bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text("")
    scripts = root / "workloads" / "pretrain_deepseek-v3" / "Megatron-Bridge" / "scripts" / "performance"
    scripts.mkdir(parents=True)
    (scripts / "bootstrap.py").write_text("")
    return root


def _env(root: Path, **extra: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith(("SLURM", "LLMB"))}
    env.update(PATH=f"{root / 'bin'}:{env['PATH']}", LLMB_INSTALL=str(root), CUDA_HOME=str(root), **extra)
    return env


def _parallelism(output: str) -> dict[str, list[str]]:
    """Map each compared option to its values; accepts both "--key value" and "--key=value"."""
    words = output.split(CAPTURED, 1)[1].split()
    found: dict[str, list[str]] = {}
    index = 0
    while index < len(words):
        key, _, value = words[index].partition("=")
        if key in COMPARED:
            if not value:
                index += 1
                value = words[index]
            found.setdefault(key, []).append(value)
        elif "=" in words[index] and not key.startswith("-"):
            found.setdefault("hydra", []).append(words[index])
        index += 1
    return found


def _slurm(root: Path, dtype: str, gpus: int, proxy: bool) -> dict[str, list[str]]:
    env = _env(
        root, GPU_TYPE="vr", MBRIDGE_GPU_TYPE="vr200", DTYPE=dtype, JOB_TOTAL_GPUS=str(gpus),
        PROXY_WORKLOAD=str(proxy).lower(), SBATCH_ACCOUNT="account", SBATCH_PARTITION="partition",
    )  # fmt: skip
    result = subprocess.run(
        ["bash", str(SLURM / "launch.sh")], cwd=SLURM, env=env, capture_output=True, text=True, check=True
    )
    return _parallelism(result.stdout)


def _baremetal(root: Path, dtype: str, gpus: int, mode: str) -> dict[str, list[str]]:
    log_dir = root / "runs" / f"{mode}_{dtype}_{gpus}"
    command = [
        "bash", str(BAREMETAL / "launch.sh"), "--mode", mode, "--dtype", dtype, "--nproc-per-node", "4",
        "--nnodes", str(gpus // 4), "--no-numa-bind", "--dry-run", "--log-dir", str(log_dir),
    ]  # fmt: skip
    result = subprocess.run(command, env=_env(root), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return _parallelism((log_dir / "dryrun.log").read_text())


PROXY_CASES = [pytest.param("fp8", n, id=f"proxy-fp8-{n}") for n in VR_CONFIG["fp8"]["proxy_scales"]]
FULL_CASES = [
    pytest.param(dtype, n, id=f"full-{dtype}-{n}") for dtype, config in VR_CONFIG.items() for n in config["scales"]
]


@requires_matching_slurm_pin
@pytest.mark.parametrize(("dtype", "gpus"), PROXY_CASES)
def test_proxy_matches_slurm_recipe(stubs: Path, dtype: str, gpus: int) -> None:
    assert _baremetal(stubs, dtype, gpus, "proxy") == _slurm(stubs, dtype, gpus, proxy=True)


@requires_matching_slurm_pin
@pytest.mark.parametrize(("dtype", "gpus"), FULL_CASES)
def test_full_matches_slurm_recipe(stubs: Path, dtype: str, gpus: int) -> None:
    assert _baremetal(stubs, dtype, gpus, "full") == _slurm(stubs, dtype, gpus, proxy=False)


def test_unsupported_scale_is_refused(stubs: Path) -> None:
    command = ["bash", str(BAREMETAL / "launch.sh"), "--mode", "proxy", "--dtype", "fp8", "--nnodes", "3", "--dry-run"]
    result = subprocess.run(command, env=_env(stubs), capture_output=True, text=True)
    assert result.returncode != 0
    assert "proxy mode supports" in result.stderr
