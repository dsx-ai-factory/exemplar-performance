# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Installation configuration prompts for LLMB installer."""

import os
from pathlib import Path
from typing import Optional

from llmb_install.downloads.image import is_enroot_installed
from llmb_install.ui.interface import UIInterface


def prompt_install_location(ui: UIInterface, default: Optional[str] = None) -> Optional[str]:
    """Prompt the user for the installation location.

    Args:
        ui: UI interface for user interaction
        default: Default installation path to suggest

    Returns:
        Optional[str]: Installation path, or None if cancelled
    """
    ui.log("\nInstallation Location")
    ui.log("--------------------")
    cwd = os.getcwd()
    ui.log(f"Current directory: {cwd}")
    ui.log("Note: Installation can exceed 100GB with multiple workloads. Use high-performance storage (Lustre, GPFS).")
    ui.log("Provide an absolute path (e.g., /lustre/scratch/username/llmb or ~/llmb-workloads)\n")

    def validate_path(path: str) -> str:
        if not path:
            return "invalid:Path required. Enter an absolute path (e.g., /scratch/username/llmb)"

        # Remove any quotes that might be present
        path = path.strip('"\'')

        # Expand ~ before checking if absolute (supports ~/llmb)
        expanded_path = Path(path).expanduser()

        # Reject root directory
        if expanded_path == Path("/"):
            return "invalid:Cannot install to root directory"

        # Require absolute path (after tilde expansion)
        if not expanded_path.is_absolute():
            return "invalid:Path must be absolute (start with / or ~)"

        # Check write permission on existing parent directory
        parent_dir = expanded_path.parent
        if parent_dir.exists() and not os.access(parent_dir, os.W_OK):
            return "invalid:No write permission in the parent directory"
        return "valid"

    location = ui.prompt_path("Enter installation directory path:", default=default or "", validate=validate_path)

    if location is None:
        return None

    # Remove any quotes that might be present
    location = location.strip('"\'')

    # Expand user path and get absolute path using pathlib
    location = str(Path(location).expanduser().resolve())

    ui.log(f"Installation location: {location}")
    return location


def prompt_cluster_type(ui: UIInterface, default_cluster_type: Optional[str] = None) -> Optional[str]:
    """Prompt the user to choose between Slurm and Kubernetes as the cluster type.

    Args:
        ui: UI interface for user interaction
        default_cluster_type: Previously saved cluster type ('slurm' or 'kubernetes')
            used to derive a sensible default.

    Returns:
        'slurm' or 'kubernetes', or None if cancelled.
    """
    ui.log("\nCluster Type")
    ui.log("------------")
    ui.log(" 'slurm':      Jobs are submitted to a SLURM-managed cluster.")
    ui.log(" 'kubernetes': Jobs run as Kubernetes WorkloadRun custom resources (nvcre/nvcrectl).")

    default = default_cluster_type or 'slurm'
    selected = ui.prompt_select("Select cluster type:", choices=['slurm', 'kubernetes'], default=default)
    return selected


def prompt_image_download_method(
    ui: UIInterface,
    default: Optional[str] = None,
    express_mode: bool = False,
    allow_kubernetes: bool = False,
) -> Optional[str]:
    """Prompt the user to select where container image downloads run for Slurm installs.

    Args:
        ui: UI interface for user interaction
        default: Default value from system config (if available)
        express_mode: Whether this is being called from express mode (shows default messages)
        allow_kubernetes: Unused; retained for call-site compatibility.

    Returns:
        Optional[str]: 'local' (download on the current node) or 'slurm' (submit as a
        Slurm CPU-partition job), or None if the user cancels.
    """
    ui.log("\nImage Download Method")
    ui.log("---------------------")
    ui.log("Please select where container images should be downloaded.")
    ui.log(" 'local': Images are downloaded on the current machine. Requires enroot.")
    ui.log(
        " 'slurm': Images are downloaded via SLURM jobs. Requires compute nodes with internet access and enroot. "
        "Recommended when interactive nodes have limited resources or network access."
    )

    # Check if we're running within a SLURM job
    running_in_slurm_job = 'SLURM_JOB_ID' in os.environ

    enroot_available = is_enroot_installed()

    if running_in_slurm_job:
        ui.log("\nDetected: Running within a SLURM job (SLURM_JOB_ID found in environment).")
        ui.log("SLURM image download method is not supported when running within a SLURM job.")

        if not enroot_available:
            ui.log("\nError: Cannot proceed with installation.")
            ui.log("You are running within a SLURM job, but enroot is not available on this system.")
            ui.log("The local image download method requires enroot.")
            ui.log("Please either:")
            ui.log("  1. Run the installer from outside a SLURM job, or")
            ui.log("  2. Ensure enroot is available on the compute nodes")
            raise SystemExit(1)

        ui.log("Forcing local image download method.")
        return "local"

    # Determine available options
    options = []
    if enroot_available:
        options.append("local")
    options.append("slurm")  # SLURM is always available as an option
    if allow_kubernetes:
        options.append("kubernetes")
        ui.log(
            " 'kubernetes': Workloads run as nvcre WorkloadRun custom resources. No local "
            "image/dataset/venv setup; assets are provisioned in-cluster."
        )

    if not enroot_available and not allow_kubernetes:
        ui.log("\nNote: Enroot is not available on this system.")
        ui.log("Local image download method is not available. Using SLURM method.")
        return "slurm"

    # Prompt for choice
    # Use provided default if valid, otherwise no default
    if default and default in options:
        selected_default = default
        if express_mode:
            ui.log(f"Using saved default: {default}")
    else:
        selected_default = None

    selected = ui.prompt_select("Select image download method:", options, default=selected_default)

    return selected
