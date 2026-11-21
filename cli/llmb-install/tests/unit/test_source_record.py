# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the record of the source commit that llmb-install writes into its repository copy."""

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from llmb_install.core.installer import Installer
from llmb_install.utils.git import describe_checkout

_METADATA = """\
general:
  workload: test_workload
  workload_type: pretrain
  framework: nemo
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _source(path: Path) -> Path:
    """A repository tree with one recipe."""
    (path / 'recipe').mkdir(parents=True)
    (path / 'recipe' / 'metadata.yaml').write_text(_METADATA)
    return path


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A git checkout with its recipe committed."""
    if shutil.which('git') is None:
        pytest.skip('git is not installed')
    repo = _source(tmp_path / 'checkout')
    _git(repo, 'init', '-q')
    _git(repo, 'add', '.')
    _git(
        repo,
        *('-c', 'user.name=Test', '-c', 'user.email=test@example.com', '-c', 'commit.gpgsign=false'),
        *('commit', '-qm', 'x'),
    )
    return repo


def _install(source: Path, tmp_path: Path) -> Path:
    installer = Installer()
    installer.root_dir = str(source)
    return Path(installer._handle_repository_setup(str(tmp_path / 'install'), dev_mode=False))


def test_clean_checkout(checkout):
    head = _git(checkout, 'rev-parse', 'HEAD')

    assert describe_checkout(str(checkout)) == {'git_commit': head, 'git_dirty': False, 'git_error': None}


def test_only_tracked_changes_make_a_checkout_dirty(checkout):
    (checkout / 'recipe' / 'notes.txt').write_text('untracked\n')
    assert describe_checkout(str(checkout))['git_dirty'] is False

    (checkout / 'recipe' / 'metadata.yaml').write_text(_METADATA + '# edited\n')
    assert describe_checkout(str(checkout))['git_dirty'] is True


def test_folder_inside_a_checkout_is_not_a_checkout(checkout):
    record = describe_checkout(str(checkout / 'recipe'))

    assert record == {'git_commit': None, 'git_dirty': None, 'git_error': 'not a git checkout'}


def test_git_failure_is_recorded(tmp_path):
    source = _source(tmp_path / 'source')
    (source / '.git').write_text('not a gitdir\n')

    record = describe_checkout(str(source))

    assert record['git_commit'] is None
    assert record['git_error']


def test_install_records_the_checkout_in_its_copy(checkout, tmp_path):
    repo = _install(checkout, tmp_path)

    assert json.loads((repo / '.llmb_source.json').read_text())['git_commit'] == _git(checkout, 'rev-parse', 'HEAD')


def test_record_failure_only_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr('llmb_install.utils.filesystem.describe_checkout', Mock(side_effect=OSError('boom')))

    repo = _install(_source(tmp_path / 'source'), tmp_path)

    assert (repo / '.llmb_repo_copy_complete').exists()
    assert "Warning: Could not write .llmb_source.json: boom" in capsys.readouterr().out
