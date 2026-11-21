#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for git caching functionality using bare repositories."""

import io
import os
import subprocess
from unittest.mock import Mock, patch

import pytest

from llmb_install.core.dependency import (
    _cleanup_invalid_git_repo,
    _clone_or_get_from_cache,
    _commit_exists_in_repo,
    _derive_cache_name_from_url,
    _derive_repo_name_from_url,
    _extract_git_progress_status,
    _get_default_branch,
    _get_git_cache_dir,
    _has_required_remotes,
    _is_valid_git_repo,
    _iter_stream_lines,
    _repo_uses_lfs,
    _run_git_with_progress,
    clone_git_repos,
)


def _make_mock_popen(stderr_bytes: bytes = b'', returncode: int = 0) -> Mock:
    """Create a mock Popen context manager with a BufferedReader stderr stream."""
    proc = Mock()
    proc.stderr = io.BufferedReader(io.BytesIO(stderr_bytes))
    proc.wait.return_value = returncode
    proc.__enter__ = Mock(return_value=proc)
    proc.__exit__ = Mock(return_value=False)
    return proc


class TestGitCaching:
    """Test git caching functionality with bare repositories."""

    def test_derive_repo_name_from_url(self):
        """Test deriving repository names from various URL formats."""
        test_cases = [
            ("https://github.com/NVIDIA/NeMo.git", "NeMo"),
            ("https://github.com/user/repo", "repo"),
            ("git@github.com:user/project.git", "project"),
            ("https://gitlab.com/group/subgroup/name.git", "name"),
        ]

        for url, expected in test_cases:
            result = _derive_repo_name_from_url(url)
            assert result == expected, f"Failed for URL: {url}"

    def test_derive_repo_name_from_url_error_handling(self):
        """Test error handling for malformed URLs in repository name derivation."""
        error_cases = [
            ("", "Invalid repository URL"),  # Empty string
            (None, "Invalid repository URL"),  # None value
            ("https://github.com/user/.git", "Cannot derive repository name"),  # Empty repo name after removing .git
            (123, "Invalid repository URL"),  # Non-string type
        ]

        for url, expected_error in error_cases:
            with pytest.raises(ValueError, match=expected_error):
                _derive_repo_name_from_url(url)

    def test_derive_cache_name_from_url(self):
        """Test deriving unique cache names from URLs."""
        test_cases = [
            ("https://github.com/NVIDIA/NeMo.git", "github.com_NVIDIA_NeMo"),
            ("git@gitlab.com:user/project.git", "gitlab.com_user_project"),
            ("https://bitbucket.org/team/repo", "bitbucket.org_team_repo"),
        ]

        for url, expected in test_cases:
            result = _derive_cache_name_from_url(url)
            assert result == expected, f"Failed for URL: {url}"

    def test_derive_cache_name_collision_avoidance(self):
        """Test that different URLs with same repo name get different cache names."""
        url1 = "https://github.com/user1/repo.git"
        url2 = "https://github.com/user2/repo.git"

        cache_name1 = _derive_cache_name_from_url(url1)
        cache_name2 = _derive_cache_name_from_url(url2)

        # Should have different cache names but same repo name
        assert cache_name1 != cache_name2
        assert _derive_repo_name_from_url(url1) == _derive_repo_name_from_url(url2)

    def test_get_git_cache_dir(self):
        """Test cache directory path generation."""
        install_path = "/path/to/install"
        expected = os.path.join(install_path, ".cache", "git_repos")
        result = _get_git_cache_dir(install_path)
        assert result == expected

    @patch('subprocess.run')
    def test_commit_exists_in_repo_success(self, mock_subprocess):
        """Test commit existence check when commit exists."""
        mock_subprocess.return_value = Mock(returncode=0)

        result = _commit_exists_in_repo("/path/to/repo", "abc123")

        mock_subprocess.assert_called_once_with(
            ['git', 'cat-file', '-e', 'abc123'], cwd="/path/to/repo", check=True, capture_output=True, text=True
        )
        assert result is True

    @patch('subprocess.run')
    def test_commit_exists_in_repo_failure(self, mock_subprocess):
        """Test commit existence check when commit doesn't exist."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, ['git', 'cat-file'])

        result = _commit_exists_in_repo("/path/to/repo", "nonexistent")

        assert result is False

    @patch('llmb_install.core.dependency._repo_uses_lfs')
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_clone_or_get_from_cache_new_repo(
        self, mock_makedirs, mock_exists, mock_subprocess, mock_popen, mock_uses_lfs
    ):
        """Test caching behavior when repository doesn't exist in cache."""
        # Setup mocks
        mock_exists.side_effect = lambda path: False  # Neither cache nor target exist
        mock_popen.return_value = _make_mock_popen()
        mock_subprocess.return_value = Mock(returncode=0, stdout='')
        mock_uses_lfs.return_value = False  # Non-LFS repo

        result = _clone_or_get_from_cache(
            name="test_repo",
            repo_url="https://github.com/test/repo.git",
            install_path="/install",
            target_dir="/target",
            commit="abc123",
        )

        # Verify the mirror clone goes through Popen with --progress
        mock_popen.assert_called_once()
        popen_call = mock_popen.call_args
        assert popen_call.args[0] == [
            'git',
            'clone',
            '--mirror',
            '--progress',
            'https://github.com/test/repo.git',
            '/install/.cache/git_repos/github.com_test_repo.git',
        ]
        assert popen_call.kwargs['env']['GIT_TERMINAL_PROMPT'] == '0'

        # Verify local git operations: clone from cache, remote setup
        expected_calls = [
            (
                [
                    'git',
                    'clone',
                    '--no-checkout',
                    '-o',
                    'cache',
                    '/install/.cache/git_repos/github.com_test_repo.git',
                    '/target/repo',
                ],
                {'check': True, 'capture_output': True, 'text': True},
            ),
            (
                ['git', 'remote', 'add', 'origin', 'https://github.com/test/repo.git'],
                {'cwd': '/target/repo', 'check': True, 'capture_output': True, 'text': True},
            ),
        ]

        assert mock_subprocess.call_count == len(expected_calls)
        for i, (expected_args, expected_kwargs) in enumerate(expected_calls):
            actual_call = mock_subprocess.call_args_list[i]
            assert actual_call.args[0] == expected_args
            for key, value in expected_kwargs.items():
                assert actual_call.kwargs[key] == value

        assert result == "/target/repo"

    @patch('llmb_install.core.dependency._repo_uses_lfs')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_clone_or_get_from_cache_existing_cache(self, mock_makedirs, mock_exists, mock_subprocess, mock_uses_lfs):
        """Test behavior when cache exists but target doesn't."""

        # Setup mocks - cache exists, target doesn't
        def mock_exists_side_effect(path):
            return '/install/.cache/git_repos/github.com_test_repo.git' in path

        mock_exists.side_effect = mock_exists_side_effect
        mock_subprocess.return_value = Mock(returncode=0, stdout='')
        mock_uses_lfs.return_value = False  # Non-LFS repo

        result = _clone_or_get_from_cache(
            name="test_repo",
            repo_url="https://github.com/test/repo.git",
            install_path="/install",
            target_dir="/target",
            commit="abc123",
        )

        # Should validate cache, check commit exists, then clone from cache to target
        expected_calls = [
            (
                ['git', '--git-dir', '/install/.cache/git_repos/github.com_test_repo.git', 'rev-parse', '--git-dir'],
                {
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                ['git', 'cat-file', '-e', 'abc123'],
                {
                    'cwd': '/install/.cache/git_repos/github.com_test_repo.git',
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                [
                    'git',
                    'clone',
                    '--no-checkout',
                    '-o',
                    'cache',
                    '/install/.cache/git_repos/github.com_test_repo.git',
                    '/target/repo',
                ],
                {'check': True, 'capture_output': True, 'text': True},
            ),
            (
                ['git', 'remote', 'add', 'origin', 'https://github.com/test/repo.git'],
                {'cwd': '/target/repo', 'check': True, 'capture_output': True, 'text': True},
            ),
        ]

        assert mock_subprocess.call_count == len(expected_calls)
        for i, (expected_args, expected_kwargs) in enumerate(expected_calls):
            actual_call = mock_subprocess.call_args_list[i]
            assert actual_call.args[0] == expected_args
            for key, value in expected_kwargs.items():
                assert actual_call.kwargs[key] == value

        assert result == "/target/repo"

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_clone_or_get_from_cache_existing_target(self, mock_makedirs, mock_exists, mock_subprocess):
        """Test behavior when target repo already exists and commit exists."""
        # Setup mocks - both cache and target exist
        mock_exists.return_value = True

        # Mock git commands in order: git rev-parse (cache), cat-file (commit check in cache),
        # git rev-parse (target), git remote (remote check), cat-file (commit check in target)
        # Note: With our optimization, we only fetch if commit is missing
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # git rev-parse for cache validation
            Mock(returncode=0),  # git cat-file for commit existence in cache
            Mock(returncode=0),  # git rev-parse for target validation
            Mock(stdout="origin\ncache\n", returncode=0),  # git remote for target
            Mock(returncode=0),  # git cat-file for commit existence in target (exists, so no fetch needed)
        ]

        result = _clone_or_get_from_cache(
            name="test_repo",
            repo_url="https://github.com/test/repo.git",
            install_path="/install",
            target_dir="/target",
            commit="abc123",
        )

        # Should validate cache, check commit exists in cache, validate target (including remotes),
        # check commit exists in target (it does, so no fetch)
        expected_calls = [
            (
                ['git', '--git-dir', '/install/.cache/git_repos/github.com_test_repo.git', 'rev-parse', '--git-dir'],
                {
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                ['git', 'cat-file', '-e', 'abc123'],
                {
                    'cwd': '/install/.cache/git_repos/github.com_test_repo.git',
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                ['git', '--git-dir', '/target/repo/.git', 'rev-parse', '--git-dir'],
                {
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                ['git', 'remote'],
                {
                    'cwd': '/target/repo',
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                ['git', 'cat-file', '-e', 'abc123'],
                {
                    'cwd': '/target/repo',
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
        ]

        assert mock_subprocess.call_count == len(expected_calls)
        for i, (expected_args, expected_kwargs) in enumerate(expected_calls):
            actual_call = mock_subprocess.call_args_list[i]
            assert actual_call.args[0] == expected_args
            for key, value in expected_kwargs.items():
                assert actual_call.kwargs[key] == value

        assert result == "/target/repo"


class TestCloneGitReposIntegration:
    """Integration tests for clone_git_repos function."""

    @patch('llmb_install.core.dependency._repo_uses_lfs')
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('llmb_install.core.dependency._commit_exists_in_repo')
    def test_clone_git_repos_with_caching(
        self, mock_commit_exists, mock_makedirs, mock_exists, mock_subprocess, mock_popen, mock_uses_lfs
    ):
        """Test clone_git_repos with caching enabled."""
        git_deps = {'nemo': {'url': 'https://github.com/NVIDIA/NeMo.git', 'commit': 'abc123'}}

        # Setup mocks
        mock_exists.side_effect = lambda path: False  # Nothing exists initially
        mock_commit_exists.return_value = True  # Commit exists, no need to fetch
        mock_popen.return_value = _make_mock_popen()
        mock_subprocess.return_value = Mock(returncode=0, stdout='')
        mock_uses_lfs.return_value = False  # Non-LFS repo

        with patch('builtins.print'):  # Suppress output
            clone_git_repos(git_deps, "/target", "/install")

        # Mirror clone and checkout run via Popen; clone from cache and remote setup via subprocess.run
        assert mock_popen.call_count == 2
        assert mock_subprocess.call_count == 2

        # The checkout of the pinned commit is the last thing that runs, and reports progress
        assert mock_popen.call_args.args[0] == ['git', 'checkout', '-f', '--progress', 'abc123']

    @patch('os.environ.get')
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('llmb_install.core.dependency._clone_or_get_from_cache')
    def test_clone_git_repos_cache_disabled(
        self, mock_cache_clone, mock_exists, mock_subprocess, mock_popen, mock_env_get
    ):
        """Test clone_git_repos with caching disabled by environment variable."""
        git_deps = {'nemo': {'url': 'https://github.com/NVIDIA/NeMo.git', 'commit': 'abc123'}}

        # Setup mocks
        mock_env_get.return_value = 'true'  # LLMB_DISABLE_GIT_CACHE=true
        mock_exists.return_value = False
        mock_popen.return_value = _make_mock_popen()
        mock_subprocess.return_value = Mock(returncode=0)

        with patch('builtins.print'):  # Suppress output
            clone_git_repos(git_deps, "/target", "/install")

        # Should NOT call cache function when disabled
        mock_cache_clone.assert_not_called()

        # Direct clone (cache disabled) and the checkout both go through Popen
        assert [call.args[0] for call in mock_popen.call_args_list] == [
            ['git', 'clone', '--no-checkout', '--progress', 'https://github.com/NVIDIA/NeMo.git', '/target/NeMo'],
            ['git', 'checkout', '-f', '--progress', 'abc123'],
        ]
        assert mock_popen.call_args_list[0].kwargs['env']['GIT_TERMINAL_PROMPT'] == '0'

    def test_clone_git_repos_empty_deps(self):
        """Test clone_git_repos with empty dependencies."""
        # Should return immediately without error
        result = clone_git_repos({}, "/target", "/install")
        assert result is None

    def test_clone_git_repos_none_deps(self):
        """Test clone_git_repos with None dependencies."""
        # Should return immediately without error
        result = clone_git_repos(None, "/target", "/install")
        assert result is None

    @patch('llmb_install.core.dependency._repo_uses_lfs')
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('llmb_install.core.dependency._commit_exists_in_repo')
    def test_git_error_handling(
        self, mock_commit_exists, mock_makedirs, mock_exists, mock_subprocess, mock_popen, mock_uses_lfs
    ):
        """Test error handling when git commands fail."""
        git_deps = {'nemo': {'url': 'https://github.com/NVIDIA/NeMo.git', 'commit': 'abc123'}}

        # Setup mocks
        mock_exists.side_effect = lambda path: False
        mock_commit_exists.return_value = True
        mock_popen.return_value = _make_mock_popen()
        mock_uses_lfs.return_value = False  # Non-LFS repo

        def popen_side_effect(cmd, **kwargs):
            # The checkout of the pinned commit is what fails here
            return _make_mock_popen(returncode=1 if cmd[0:2] == ['git', 'checkout'] else 0)

        mock_popen.side_effect = popen_side_effect
        mock_subprocess.return_value = Mock(returncode=0, stdout='')

        with pytest.raises(subprocess.CalledProcessError):
            with patch('builtins.print'):  # Suppress output
                clone_git_repos(git_deps, "/target", "/install")


class TestGitRemoteValidation:
    """Test git remote validation functionality."""

    @patch('subprocess.run')
    def test_has_required_remotes_success(self, mock_subprocess):
        """Test checking required remotes when all exist."""
        mock_subprocess.return_value = Mock(stdout="origin\ncache\nupstream\n", returncode=0)

        result = _has_required_remotes("/path/to/repo", ["origin", "cache"])

        mock_subprocess.assert_called_once_with(
            ['git', 'remote'], cwd="/path/to/repo", check=True, capture_output=True, text=True
        )
        assert result is True

    @patch('subprocess.run')
    def test_has_required_remotes_missing(self, mock_subprocess):
        """Test checking required remotes when some are missing."""
        mock_subprocess.return_value = Mock(stdout="origin\n", returncode=0)

        result = _has_required_remotes("/path/to/repo", ["origin", "cache"])

        mock_subprocess.assert_called_once_with(
            ['git', 'remote'], cwd="/path/to/repo", check=True, capture_output=True, text=True
        )
        assert result is False

    @patch('subprocess.run')
    def test_has_required_remotes_empty_output(self, mock_subprocess):
        """Test checking required remotes when no remotes exist."""
        mock_subprocess.return_value = Mock(stdout="", returncode=0)

        result = _has_required_remotes("/path/to/repo", ["origin"])

        assert result is False

    @patch('subprocess.run')
    def test_has_required_remotes_git_error(self, mock_subprocess):
        """Test checking required remotes when git command fails."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, ['git', 'remote'])

        result = _has_required_remotes("/path/to/repo", ["origin"])

        assert result is False

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_is_valid_git_repo_with_required_remotes(self, mock_exists, mock_subprocess):
        """Test git repo validation with required remotes check."""
        mock_exists.return_value = True
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # git rev-parse call
            Mock(stdout="origin\ncache\n", returncode=0),  # git remote call
        ]

        result = _is_valid_git_repo("/path/to/repo", is_bare=False, required_remotes=["cache"])

        assert result is True
        # Should call both git rev-parse and git remote
        assert mock_subprocess.call_count == 2

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_is_valid_git_repo_missing_required_remotes(self, mock_exists, mock_subprocess):
        """Test git repo validation fails when required remotes are missing."""
        mock_exists.return_value = True
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # git rev-parse call
            Mock(stdout="origin\n", returncode=0),  # git remote call (missing 'cache')
        ]

        result = _is_valid_git_repo("/path/to/repo", is_bare=False, required_remotes=["cache"])

        assert result is False

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_is_valid_git_repo_no_remote_check_when_none_required(self, mock_exists, mock_subprocess):
        """Test git repo validation without remote check when no remotes required."""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)  # git rev-parse call

        result = _is_valid_git_repo("/path/to/repo", is_bare=False, required_remotes=None)

        assert result is True
        # Should only call git rev-parse, not git remote
        assert mock_subprocess.call_count == 1
        assert mock_subprocess.call_args[0][0] == ['git', '--git-dir', '/path/to/repo/.git', 'rev-parse', '--git-dir']

    @patch('llmb_install.core.dependency._is_valid_git_repo')
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_cleanup_invalid_git_repo_with_required_remotes(self, mock_rmtree, mock_exists, mock_is_valid):
        """Test cleanup when repo exists but is invalid due to missing remotes."""
        mock_exists.return_value = True
        mock_is_valid.return_value = False

        _cleanup_invalid_git_repo("/path/to/repo", is_bare=False, required_remotes=["cache"])

        mock_is_valid.assert_called_once_with("/path/to/repo", is_bare=False, required_remotes=["cache"])
        mock_rmtree.assert_called_once_with("/path/to/repo")

    @patch('llmb_install.core.dependency._is_valid_git_repo')
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_cleanup_valid_git_repo_not_removed(self, mock_rmtree, mock_exists, mock_is_valid):
        """Test cleanup when repo is valid - should not be removed."""
        mock_exists.return_value = True
        mock_is_valid.return_value = True

        _cleanup_invalid_git_repo("/path/to/repo", is_bare=False, required_remotes=["cache"])

        mock_is_valid.assert_called_once_with("/path/to/repo", is_bare=False, required_remotes=["cache"])
        mock_rmtree.assert_not_called()


class TestGitLfsDetection:
    """Test git-lfs detection functionality."""

    @patch('subprocess.run')
    def test_repo_uses_lfs_with_lfs_files(self, mock_subprocess):
        """Test detection of repositories using git-lfs."""
        # Mock .gitattributes content with LFS filters
        gitattributes_content = """*.psd filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
"""
        mock_subprocess.return_value = Mock(returncode=0, stdout=gitattributes_content)

        result = _repo_uses_lfs("/path/to/repo", "abc123")

        assert result is True
        mock_subprocess.assert_called_once_with(
            ['git', 'show', 'abc123:.gitattributes'],
            cwd="/path/to/repo",
            capture_output=True,
            text=True,
            check=False,
        )

    @patch('subprocess.run')
    def test_repo_uses_lfs_without_lfs_files(self, mock_subprocess):
        """Test detection of repositories not using git-lfs."""
        # Mock .gitattributes without LFS filters
        gitattributes_content = """* text=auto
*.py text
*.md text
"""
        mock_subprocess.return_value = Mock(returncode=0, stdout=gitattributes_content)

        result = _repo_uses_lfs("/path/to/repo", "abc123")

        assert result is False

    @patch('subprocess.run')
    def test_repo_uses_lfs_no_gitattributes(self, mock_subprocess):
        """Test detection when .gitattributes doesn't exist."""
        # Git returns non-zero when file doesn't exist
        mock_subprocess.return_value = Mock(returncode=128, stdout='', stderr='fatal: path not in commit')

        result = _repo_uses_lfs("/path/to/repo", "abc123")

        assert result is False

    @patch('subprocess.run')
    def test_repo_uses_lfs_handles_exceptions(self, mock_subprocess):
        """Test that exceptions are handled gracefully."""
        mock_subprocess.side_effect = OSError("Command not found")

        result = _repo_uses_lfs("/path/to/repo", "abc123")

        assert result is False


class TestDefaultBranchDetection:
    """Test default branch detection functionality."""

    @patch('subprocess.run')
    def test_get_default_branch_main(self, mock_subprocess):
        """Test getting default branch when it's 'main'."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="refs/remotes/cache/main\n")

        result = _get_default_branch("/path/to/repo", "cache")

        assert result == "main"
        mock_subprocess.assert_called_once_with(
            ['git', 'symbolic-ref', 'refs/remotes/cache/HEAD'],
            cwd="/path/to/repo",
            capture_output=True,
            text=True,
            check=True,
        )

    @patch('subprocess.run')
    def test_get_default_branch_master(self, mock_subprocess):
        """Test getting default branch when it's 'master'."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="refs/remotes/origin/master\n")

        result = _get_default_branch("/path/to/repo", "origin")

        assert result == "master"

    @patch('subprocess.run')
    def test_get_default_branch_custom(self, mock_subprocess):
        """Test getting custom default branch names."""
        test_cases = [
            ("develop", "refs/remotes/cache/develop"),
            ("trunk", "refs/remotes/cache/trunk"),
            ("release", "refs/remotes/cache/release"),
        ]

        for expected_branch, symbolic_ref_output in test_cases:
            mock_subprocess.return_value = Mock(returncode=0, stdout=f"{symbolic_ref_output}\n")
            result = _get_default_branch("/path/to/repo", "cache")
            assert result == expected_branch, f"Failed for branch: {expected_branch}"

    @patch('subprocess.run')
    def test_get_default_branch_not_found(self, mock_subprocess):
        """Test when symbolic-ref command fails."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, ['git', 'symbolic-ref'])

        result = _get_default_branch("/path/to/repo", "cache")

        assert result is None

    @patch('subprocess.run')
    def test_get_default_branch_malformed_output(self, mock_subprocess):
        """Test handling of malformed symbolic-ref output."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="malformed-output")

        result = _get_default_branch("/path/to/repo", "cache")

        assert result is None


class TestGitProgress:
    """Tests for git progress status parsing and live progress bar updates."""

    def test_extract_git_progress_status_recognizes_phases(self):
        """Progress lines (with or without 'remote:' prefix) return cleaned status text."""
        assert (
            _extract_git_progress_status("remote: Enumerating objects: 420, done.") == "Enumerating objects: 420, done."
        )
        assert (
            _extract_git_progress_status("remote: Counting objects:  45% (189/420)")
            == "Counting objects:  45% (189/420)"
        )
        assert (
            _extract_git_progress_status("remote: Compressing objects: 100% (200/200), done.")
            == "Compressing objects: 100% (200/200), done."
        )
        assert (
            _extract_git_progress_status("Receiving objects:  42% (51832/123410), 84.50 MiB | 12.34 MiB/s")
            == "Receiving objects:  42% (51832/123410), 84.50 MiB | 12.34 MiB/s"
        )
        assert (
            _extract_git_progress_status("Resolving deltas:  75% (6000/8000)") == "Resolving deltas:  75% (6000/8000)"
        )
        assert _extract_git_progress_status("Updating files:  90% (900/1000)") == "Updating files:  90% (900/1000)"
        assert (
            _extract_git_progress_status("Filtering content:  50% (2/4), 10.00 MiB | 5.00 MiB/s")
            == "Filtering content:  50% (2/4), 10.00 MiB | 5.00 MiB/s"
        )

    def test_extract_git_progress_status_ignores_non_progress_lines(self):
        """Non-progress stderr output returns None so it does not clutter the progress bar."""
        assert _extract_git_progress_status("Cloning into bare repository '/cache/repo.git'...") is None
        assert _extract_git_progress_status("From https://github.com/NVIDIA/NeMo") is None
        assert _extract_git_progress_status(" * [new branch]      main       -> origin/main") is None
        assert (
            _extract_git_progress_status("fatal: unable to access 'https://example.invalid/': Operation too slow")
            is None
        )
        assert _extract_git_progress_status("") is None
        assert _extract_git_progress_status("error: 1234 bytes of body are still expected") is None
        assert _extract_git_progress_status("remote: Total 3 (delta 1), reused 0 (delta 0), pack-reused 0") is None
        assert _extract_git_progress_status("Submodule path 'sub': checked out 'abc123'") is None

    def test_extract_git_progress_status_matches_shape_not_phase_name(self):
        """Progress is recognised by its shape, so older git wording, other locales and count-only phases match."""
        assert _extract_git_progress_status("Checking out files:  50% (1/2)") == "Checking out files:  50% (1/2)"
        assert _extract_git_progress_status("Empfange Objekte:  42% (42/100)") == "Empfange Objekte:  42% (42/100)"
        assert _extract_git_progress_status("remote: Enumerating objects: 12345   ") == "Enumerating objects: 12345"

    def test_iter_stream_lines_splits_on_cr_and_keeps_multibyte_chars_split_across_reads(self):
        """A UTF-8 character straddling two read1() chunks is decoded intact, and the unterminated tail is flushed."""
        chunks = iter([b"caf\xc3", b"\xa9\rnext\r\nlast", b""])
        stream = Mock(read1=Mock(side_effect=lambda _n: next(chunks)))

        assert list(_iter_stream_lines(stream)) == ["caf\u00e9", "next", "last"]

    @patch('llmb_install.core.dependency.Progress')
    @patch('subprocess.Popen')
    def test_run_git_with_progress_streams_cr_delimited_updates(self, mock_popen, mock_progress_cls):
        """CR-delimited stderr progress lines update the rich task status."""
        stderr_stream = (
            b"Cloning into bare repository '/cache/repo.git'...\n"
            b"remote: Counting objects:  50% (5/10)\r"
            b"Receiving objects:  42% (42/100), 12.00 MiB | 6.00 MiB/s\r"
            b"Resolving deltas: 100% (10/10), done.\n"
        )
        mock_popen.return_value = _make_mock_popen(stderr_bytes=stderr_stream, returncode=0)
        mock_progress = Mock()
        mock_progress.add_task.return_value = 1
        mock_progress_cls.return_value.__enter__.return_value = mock_progress

        _run_git_with_progress(
            ['clone', '--mirror', '--progress', 'https://example.invalid/r.git', '/cache/r.git'], "Cloning repo"
        )

        mock_progress.add_task.assert_called_once_with("Cloning repo", total=None, status="")
        statuses = [call.kwargs['status'] for call in mock_progress.update.call_args_list]
        assert statuses == [
            "Counting objects:  50% (5/10)",
            "Receiving objects:  42% (42/100), 12.00 MiB | 6.00 MiB/s",
            "Resolving deltas: 100% (10/10), done.",
        ]

    @patch('llmb_install.core.dependency.Progress')
    @patch('subprocess.Popen')
    def test_run_git_with_progress_fallback_when_no_parseable_output(self, mock_popen, mock_progress_cls):
        """When git emits no progress lines, the indeterminate task runs and completes cleanly."""
        mock_popen.return_value = _make_mock_popen(stderr_bytes=b"Cloning into '/target/repo'...\n", returncode=0)
        mock_progress = Mock()
        mock_progress.add_task.return_value = 1
        mock_progress_cls.return_value.__enter__.return_value = mock_progress

        _run_git_with_progress(['fetch', '--progress', 'origin'], "Fetching repo", cwd="/target/repo")

        mock_progress.add_task.assert_called_once_with("Fetching repo", total=None, status="")
        mock_progress.update.assert_not_called()
        assert mock_popen.call_args.kwargs['cwd'] == "/target/repo"
        assert mock_popen.call_args.args[0] == ['git', 'fetch', '--progress', 'origin']

    @patch('subprocess.Popen')
    def test_run_git_with_progress_error_preserves_fatal_stderr_without_progress_noise(self, mock_popen):
        """On failure, CalledProcessError.stderr includes fatal lines but strips CR progress spam."""
        stderr_stream = (
            b"Cloning into bare repository '/cache/repo.git'...\n"
            b"Receiving objects:  10% (10/100), 1.00 MiB | 512.00 KiB/s\r"
            b"fatal: RPC failed; curl 28 Operation too slow. Less than 1024 bytes/sec transferred the last 60 seconds\n"
        )
        mock_popen.return_value = _make_mock_popen(stderr_bytes=stderr_stream, returncode=128)

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            _run_git_with_progress(
                ['clone', '--mirror', '--progress', 'https://example.invalid/r.git', '/cache/r.git'], "Cloning repo"
            )

        assert exc_info.value.returncode == 128
        assert "Operation too slow" in exc_info.value.stderr
        assert "Receiving objects" not in exc_info.value.stderr
        assert exc_info.value.cmd[:4] == ['git', 'clone', '--mirror', '--progress']

    @patch('llmb_install.core.dependency.Progress')
    @patch('subprocess.Popen')
    def test_run_git_with_progress_reassembles_lines_split_across_chunks(self, mock_popen, mock_progress_cls):
        """A progress or error line split across read1() chunks is reassembled before parsing."""
        proc = _make_mock_popen(returncode=0)
        chunks = iter([b"remote: Receiving obj", b"ects:  75% (75/100), 5.00 MiB | 2.50 MiB/s\r", b""])
        proc.stderr = Mock(read1=Mock(side_effect=lambda _n: next(chunks)))
        mock_popen.return_value = proc
        mock_progress = Mock()
        mock_progress.add_task.return_value = 1
        mock_progress_cls.return_value.__enter__.return_value = mock_progress

        _run_git_with_progress(['fetch', '--progress', 'origin'], "Fetching repo")

        mock_progress.update.assert_called_once_with(
            1, status="Receiving objects:  75% (75/100), 5.00 MiB | 2.50 MiB/s"
        )

    @patch('llmb_install.core.dependency._get_default_branch', return_value='main')
    @patch('llmb_install.core.dependency._repo_uses_lfs', return_value=True)
    @patch('llmb_install.core.dependency._commit_exists_in_repo', side_effect=[False, True])
    @patch('llmb_install.core.dependency._cleanup_invalid_git_repo')
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_cache_fetch_and_lfs_fetch_report_progress(
        self, mock_exists, mock_run, mock_popen, _mock_cleanup, _mock_commit_exists, _mock_lfs, _mock_branch
    ):
        """Both cache-miss fetch and LFS origin fetch pass --progress."""
        mock_exists.side_effect = lambda path: path.endswith('.git')  # Cache exists, target does not
        mock_popen.side_effect = lambda *args, **kwargs: _make_mock_popen()
        mock_run.return_value = Mock(returncode=0, stdout='')

        _clone_or_get_from_cache("test_repo", "https://github.com/test/repo.git", "/install", "/target", "abc123")

        assert mock_popen.call_count == 2
        cache_fetch_call, lfs_fetch_call = mock_popen.call_args_list
        assert cache_fetch_call.args[0] == ['git', 'fetch', '--progress', 'origin']
        assert cache_fetch_call.kwargs['cwd'] == '/install/.cache/git_repos/github.com_test_repo.git'
        assert lfs_fetch_call.args[0] == ['git', 'fetch', '--progress', 'origin']
        assert lfs_fetch_call.kwargs['cwd'] == '/target/repo'

    @patch('llmb_install.core.dependency._run_git_with_progress')
    @patch('subprocess.run')
    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    def test_clone_line_and_bar_use_different_wording(
        self, _mock_makedirs, _mock_exists, mock_run, mock_run_with_progress, capsys
    ):
        """The printed line names the operation; the bar names the phase, so they don't read as a duplicate."""
        mock_run.return_value = Mock(returncode=0, stdout='')

        _clone_or_get_from_cache("nemo", "https://github.com/NVIDIA/NeMo.git", "/install", "/target", "abc123")

        assert capsys.readouterr().out.startswith("  Cloning nemo from https://github.com/NVIDIA/NeMo.git")
        assert mock_run_with_progress.call_args.args[1] == "Downloading nemo"

    @patch('llmb_install.core.dependency._interactive_output', return_value=False)
    @patch('llmb_install.core.dependency._commit_exists_in_repo', return_value=True)
    @patch('llmb_install.core.dependency._cleanup_invalid_git_repo')
    @patch('os.path.exists', return_value=True)
    def test_cached_repo_is_announced_when_no_live_display(
        self, _mock_exists, _mock_cleanup, _mock_commit_exists, _mock_interactive, capsys
    ):
        """A cached repo does no network work, so without this line the log looks stuck."""
        _clone_or_get_from_cache("nemo", "https://github.com/NVIDIA/NeMo.git", "/install", "/target", "abc123")

        assert "  Using cached nemo" in capsys.readouterr().out

    @patch('llmb_install.core.dependency._interactive_output', return_value=False)
    @patch('llmb_install.core.dependency.Progress')
    @patch('subprocess.Popen')
    def test_run_git_with_progress_disables_live_display_when_not_a_terminal(
        self, mock_popen, mock_progress_cls, _mock_interactive
    ):
        """Non-TTY output gets no bar frames (hence no control codes, no stray newlines)."""
        mock_popen.return_value = _make_mock_popen(stderr_bytes=b"Receiving objects:  42% (42/100)\r")

        _run_git_with_progress(['fetch', '--progress', 'origin'], "Fetching repo")

        assert mock_progress_cls.call_args.kwargs['disable'] is True

    @patch('llmb_install.core.dependency._interactive_output', return_value=True)
    @patch('llmb_install.core.dependency.Progress')
    @patch('subprocess.Popen')
    def test_run_git_with_progress_enables_live_display_on_a_terminal(
        self, mock_popen, mock_progress_cls, _mock_interactive
    ):
        """A terminal keeps the live display, which clears itself when done."""
        mock_popen.return_value = _make_mock_popen(stderr_bytes=b"Receiving objects:  42% (42/100)\r")

        _run_git_with_progress(['fetch', '--progress', 'origin'], "Fetching repo")

        assert mock_progress_cls.call_args.kwargs['disable'] is False
        assert mock_progress_cls.call_args.kwargs['transient'] is True

    @patch('llmb_install.core.dependency._repo_uses_lfs')
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_clone_of_uncached_repo_announces_name_and_url(
        self, mock_makedirs, mock_exists, mock_run, mock_popen, mock_uses_lfs, capsys
    ):
        """Name and URL are printed up front, so logs show what a long silence refers to."""
        mock_exists.side_effect = lambda path: False
        mock_popen.return_value = _make_mock_popen()
        mock_run.return_value = Mock(returncode=0, stdout='')
        mock_uses_lfs.return_value = False

        _clone_or_get_from_cache("nemo", "https://github.com/NVIDIA/NeMo.git", "/install", "/target", "abc123")

        assert "  Cloning nemo from https://github.com/NVIDIA/NeMo.git" in capsys.readouterr().out

    @patch('llmb_install.core.dependency._interactive_output', return_value=False)
    @patch('llmb_install.core.dependency._get_default_branch', return_value='main')
    @patch('llmb_install.core.dependency._repo_uses_lfs', return_value=True)
    @patch('llmb_install.core.dependency._commit_exists_in_repo', side_effect=[False, True])
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_fetch_is_announced_when_no_live_display(
        self, mock_exists, mock_run, mock_popen, _commit_exists, _lfs, _branch, _interactive, capsys
    ):
        """A logged install still records the fetches, which the bar would otherwise be the only trace of."""
        mock_exists.side_effect = lambda path: path.endswith('.git')
        mock_popen.side_effect = lambda *args, **kwargs: _make_mock_popen()
        mock_run.return_value = Mock(returncode=0, stdout='')

        _clone_or_get_from_cache("nemo", "https://github.com/NVIDIA/NeMo.git", "/install", "/target", "abc123")

        assert capsys.readouterr().out.count("  Fetching nemo from https://github.com/NVIDIA/NeMo.git...") == 2

    @patch('llmb_install.core.dependency._interactive_output', return_value=True)
    @patch('llmb_install.core.dependency._get_default_branch', return_value='main')
    @patch('llmb_install.core.dependency._repo_uses_lfs', return_value=True)
    @patch('llmb_install.core.dependency._commit_exists_in_repo', side_effect=[False, True])
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_fetch_is_not_announced_when_the_bar_shows_it(
        self, mock_exists, mock_run, mock_popen, _commit_exists, _lfs, _branch, _interactive, capsys
    ):
        """On a terminal the 'Downloading nemo' bar covers the fetch, so no 'Fetching' line is printed."""
        mock_exists.side_effect = lambda path: path.endswith('.git')
        mock_popen.side_effect = lambda *args, **kwargs: _make_mock_popen()
        mock_run.return_value = Mock(returncode=0, stdout='')

        _clone_or_get_from_cache("nemo", "https://github.com/NVIDIA/NeMo.git", "/install", "/target", "abc123")

        assert "Fetching nemo" not in capsys.readouterr().out

    @patch('llmb_install.core.dependency._run_git_with_progress')
    @patch('llmb_install.core.dependency._clone_or_get_from_cache', return_value='/target/NeMo-RL')
    def test_submodule_update_reports_progress(self, _mock_cache_clone, mock_run_with_progress):
        """Submodules can be large network fetches, so they go through the progress runner too."""
        git_deps = {'nemo_rl': {'url': 'https://github.com/NVIDIA/NeMo-RL.git', 'commit': 'abc123', 'submodules': True}}

        with patch('builtins.print'):
            clone_git_repos(git_deps, "/target", "/install")

        assert mock_run_with_progress.call_args.args[0] == [
            'submodule',
            'update',
            '--init',
            '--recursive',
            '--progress',
        ]
        assert mock_run_with_progress.call_args.kwargs['cwd'] == '/target/NeMo-RL'
