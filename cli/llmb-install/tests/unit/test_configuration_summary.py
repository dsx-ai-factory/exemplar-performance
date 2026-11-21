# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the installation configuration summary."""

from unittest.mock import MagicMock, patch

from llmb_install.config.models import InstallConfig, KubernetesConfig, SlurmConfig
from llmb_install.core.installer import Installer


def _summary_output(config, capsys):
    ui = MagicMock()
    ui.prompt_select.return_value = 'yes'
    with patch.object(Installer, 'create_ui', return_value=ui):
        Installer()._show_configuration_summary(config)
    return capsys.readouterr().out


def _config(**kwargs):
    return InstallConfig(
        install_path='/tmp/llmb',
        venv_type='uv',
        gpu_type='h100',
        node_architecture='x86_64',
        selected_workloads=['test_workload'],
        image_download_method='local',
        **kwargs,
    )


def test_kubernetes_summary_shows_cluster_type_and_namespace(capsys):
    out = _summary_output(_config(kubernetes=KubernetesConfig(namespace='bench')), capsys)

    assert 'Cluster Type: kubernetes' in out
    assert 'Namespace: bench' in out
    assert 'Image Download Method' not in out


def test_slurm_summary_shows_cluster_type_and_image_download_method(capsys):
    slurm = SlurmConfig(account='acct', gpu_partition='gpu', cpu_partition='cpu')
    out = _summary_output(_config(slurm=slurm), capsys)

    assert 'Cluster Type: slurm' in out
    assert 'Image Download Method: local' in out
    assert 'Namespace' not in out
