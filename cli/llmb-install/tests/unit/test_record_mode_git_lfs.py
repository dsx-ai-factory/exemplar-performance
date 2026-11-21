# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Git LFS is required only by runs that can clone a recipe repository."""

import argparse

import pytest
import yaml

from llmb_install.config.models import InstallConfig, SlurmConfig
from llmb_install.core.installer import Installer


def _args(**overrides) -> argparse.Namespace:
    base = {'play': None, 'record': None, 'command': None, 'verbose': False, 'ui_mode': 'simple', 'dev_mode': False}
    return argparse.Namespace(**{**base, **overrides})


def _config(install_path: str) -> InstallConfig:
    return InstallConfig(
        install_path=install_path,
        venv_type='uv',
        gpu_type='h100',
        node_architecture='x86_64',
        selected_workloads=['nemotron'],
        slurm=SlurmConfig(account='acct', gpu_partition='gpu', cpu_partition='cpu'),
    )


LIST = {'command': 'express', 'list_workloads': True}


@pytest.mark.parametrize(
    "kwargs, without_state",
    [
        ({'record': 'out.yaml'}, False),
        (LIST, False),
        ({}, True),
        ({'play': 'config.yaml'}, True),
        ({'command': 'express', 'list_workloads': False}, True),
        ({'play': 'config.yaml', **LIST}, True),
    ],
    ids=['record', 'list-workloads', 'interactive', 'play', 'express', 'play-list-workloads'],
)
def test_needs_git_lfs(kwargs, without_state):
    assert Installer()._needs_git_lfs(_args(**kwargs), None) is without_state
    assert Installer()._needs_git_lfs(_args(**kwargs), ('saved',)) is True


@pytest.mark.parametrize("lfs_installed", [False, True])
def test_record_skips_git_lfs(tmp_path, monkeypatch, lfs_installed):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('llmb_install.core.installer.find_llmb_repo_root', lambda: str(tmp_path))
    monkeypatch.setattr('llmb_install.core.installer.build_workload_dict', lambda root: {'nemotron': {}})
    monkeypatch.setattr('llmb_install.core.installer.load_install_state', lambda: None)
    monkeypatch.setattr(Installer, '_collect_configuration', lambda self, **kw: _config(str(tmp_path / 'install')))
    monkeypatch.setattr('llmb_install.core.installer.is_git_lfs_installed', lambda: lfs_installed)
    configured = []
    monkeypatch.setattr('llmb_install.core.installer.ensure_git_lfs_configured', lambda: configured.append(True))
    playfile = tmp_path / 'recorded.yaml'

    Installer().run(_args(record=str(playfile)))

    assert yaml.safe_load(playfile.read_text())['selected_workloads'] == ['nemotron']
    assert configured == []
