# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Git utilities for LLMB installation.

This module provides functions for checking git and git-lfs availability
and configuration, and for reading the commit of a checkout.
"""

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def is_git_lfs_installed() -> bool:
    """Check if Git LFS is installed and available in the system PATH.

    Returns:
        bool: True if git-lfs command is available, False otherwise
    """
    return shutil.which('git-lfs') is not None


def ensure_git_lfs_configured() -> None:
    """Ensure Git LFS is configured for the current user.

    Checks if Git LFS is already configured by looking for the lfs filter
    in git config. If not configured, runs 'git lfs install --skip-repo' to set it up.

    Uses --skip-repo to configure global filters without modifying the llmb-repo itself,
    since the installer runs from within the llmb-repo.

    This function is idempotent and safe to call multiple times.

    Raises:
        subprocess.CalledProcessError: If git lfs install fails
    """
    try:
        # Check if already configured by looking for the lfs process filter
        result = subprocess.run(
            ['git', 'config', '--global', '--get', 'filter.lfs.process'],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            logger.debug("Git LFS already configured")
            return

        # Not configured, run git lfs install
        logger.debug("Configuring Git LFS")
        subprocess.run(
            ['git', 'lfs', 'install', '--skip-repo'],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug("Git LFS configured successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to configure Git LFS: {e.stderr if e.stderr else str(e)}")
        raise


def describe_checkout(repo_path: str) -> dict:
    """Return git_commit, git_dirty (a tracked file differs from HEAD) and git_error (why either is None)."""
    record = {'git_commit': None, 'git_dirty': None, 'git_error': None}
    # Without its own .git, git would describe an enclosing checkout instead.
    if not (Path(repo_path) / '.git').exists():
        record['git_error'] = 'not a git checkout'
        return record
    # GIT_OPTIONAL_LOCKS=0 keeps `git status` from rewriting the index.
    env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}
    run = {'cwd': repo_path, 'env': env, 'capture_output': True, 'text': True, 'check': True}
    try:
        record['git_commit'] = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], **run).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], **run)
        record['git_dirty'] = bool(status.stdout.strip())
    except subprocess.CalledProcessError as e:
        record['git_error'] = e.stderr.strip() or str(e)
    except OSError as e:
        record['git_error'] = str(e)
    return record
