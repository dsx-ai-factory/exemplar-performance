# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HuggingFace download/verify workflow."""

import os
import resource
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmb_install.downloads import huggingface as hf


@pytest.fixture(autouse=True)
def isolated_download_settings():
    # Each test represents a fresh installer process, independent of the test host
    # and of any settings selected by earlier downloads in this test session.
    with (
        patch.object(hf, "_download_settings", None),
        patch.object(hf, "_download_settings_announced", False),
        patch.object(resource, "getrlimit", return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY)),
        patch.dict(os.environ),
    ):
        for name in list(os.environ):
            if name.startswith(("HF_XET_", "HF_HUB_", "LLMB_HF_")):
                os.environ.pop(name)
        yield


REPO_ID = "meta-llama/Meta-Llama-3-70B"
NEMOTRON_REPO_ID = "nvidia/Nemotron-4-340B-Base"
WORKLOAD_KEY = "workload1"


def _workloads_with_downloads(downloads):
    return {WORKLOAD_KEY: {"downloads": downloads}}


def _asset_specs(repo_id, *assets):
    return (hf.HuggingFaceAssetSpec(repo_id=repo_id, assets=frozenset(assets)),)


def _assert_hf_env(test_case, cache_dir):
    test_case.assertEqual(os.environ.get("HF_HOME"), cache_dir)
    test_case.assertEqual(os.environ.get("HF_HUB_CACHE"), os.path.join(cache_dir, "hub"))


def _assert_offline_call(test_case, mocked_loader, repo_id):
    args, kwargs = mocked_loader.from_pretrained.call_args
    test_case.assertEqual(args[0], repo_id)
    test_case.assertTrue(kwargs["local_files_only"])
    test_case.assertTrue(kwargs["trust_remote_code"])


class TestNormalizeHfToken(unittest.TestCase):
    """Tests for HuggingFace token normalization."""

    def test_strips_whitespace(self):
        self.assertEqual(hf._normalize_hf_token("  hf_token  "), "hf_token")
        self.assertEqual(hf._normalize_hf_token("\thf_token\t"), "hf_token")

    def test_strips_newlines(self):
        self.assertEqual(hf._normalize_hf_token("hf_token\n"), "hf_token")
        self.assertEqual(hf._normalize_hf_token("hf_token\r\n"), "hf_token")

    def test_strips_quotes(self):
        self.assertEqual(hf._normalize_hf_token('"hf_token"'), "hf_token")
        self.assertEqual(hf._normalize_hf_token("'hf_token'"), "hf_token")

    def test_strips_quotes_with_whitespace(self):
        self.assertEqual(hf._normalize_hf_token('  "hf_token"  '), "hf_token")

    def test_returns_none_for_whitespace_only(self):
        self.assertIsNone(hf._normalize_hf_token("   "))
        self.assertIsNone(hf._normalize_hf_token("\n"))

    def test_preserves_valid_token(self):
        self.assertEqual(hf._normalize_hf_token("hf_abcd1234"), "hf_abcd1234")


class TestHuggingFaceDownloadPlan(unittest.TestCase):
    """Tests for HuggingFace download plan parsing."""

    def test_cache_assets_default_to_tokenizer_and_config(self):
        workloads = _workloads_with_downloads({"huggingface": [{"repo_id": REPO_ID}]})

        plan = hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

        self.assertEqual(len(plan.cache), 1)
        self.assertEqual(plan.cache[0].repo_id, REPO_ID)
        self.assertEqual(plan.cache[0].assets, frozenset({"tokenizer", "config"}))

    def test_map_form_cache_entries_match_legacy_list_form(self):
        list_form = _workloads_with_downloads({"huggingface": [{"repo_id": REPO_ID, "assets": ["config"]}]})
        map_form = _workloads_with_downloads({"huggingface": {"cache": [{"repo_id": REPO_ID, "assets": ["config"]}]}})

        list_plan = hf.build_hf_download_plan(list_form, [WORKLOAD_KEY])
        map_plan = hf.build_hf_download_plan(map_form, [WORKLOAD_KEY])

        self.assertEqual(list_plan, map_plan)

    def test_repo_entry_defaults(self):
        workloads = _workloads_with_downloads(
            {"huggingface": {"repos": [{"repo_id": "org/model-name", "repo_type": "model"}]}}
        )

        plan = hf.build_hf_download_plan(workloads, [WORKLOAD_KEY], gpu_type="h100")

        self.assertEqual(plan.cache, ())
        self.assertEqual(len(plan.repos), 1)
        repo = plan.repos[0]
        self.assertEqual(repo.repo_id, "org/model-name")
        self.assertEqual(repo.repo_type, "model")
        self.assertIsNone(repo.revision)
        self.assertEqual(repo.target, "workload")
        self.assertEqual(repo.name, "model-name")

    def test_repo_entry_explicit_fields(self):
        workloads = _workloads_with_downloads(
            {
                "huggingface": {
                    "repos": [{"repo_id": "org/data", "repo_type": "dataset", "revision": "abc123", "name": "dataset"}]
                }
            }
        )

        (repo,) = hf.build_hf_download_plan(workloads, [WORKLOAD_KEY]).repos

        self.assertEqual(repo.repo_type, "dataset")
        self.assertEqual(repo.revision, "abc123")
        self.assertEqual(repo.target, "workload")
        self.assertEqual(repo.name, "dataset")

    def test_repo_entry_requires_repo_type(self):
        workloads = _workloads_with_downloads({"huggingface": {"repos": [{"repo_id": "org/model-name"}]}})

        with self.assertRaisesRegex(ValueError, "repo_type"):
            hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

    def test_repo_gpu_filtering_happens_in_plan(self):
        workloads = _workloads_with_downloads(
            {"huggingface": {"repos": [{"repo_id": REPO_ID, "repo_type": "model", "when": {"gpu": ["gb200"]}}]}}
        )

        plan = hf.build_hf_download_plan(workloads, [WORKLOAD_KEY], gpu_type="h100")

        self.assertEqual(plan.cache, ())
        self.assertEqual(plan.repos, ())

    def test_repo_entry_validated_even_when_gpu_filtered(self):
        # A malformed entry must fail on every GPU type, not only on matching ones.
        workloads = _workloads_with_downloads(
            {
                "huggingface": {
                    "repos": [
                        {
                            "repo_id": REPO_ID,
                            "repo_type": "model",
                            "target": "bogus",
                            "when": {"gpu": ["gb200"]},
                        }
                    ]
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "target"):
            hf.build_hf_download_plan(workloads, [WORKLOAD_KEY], gpu_type="h100")

    def test_cache_assets_merge_by_revision(self):
        workloads = _workloads_with_downloads(
            {
                "huggingface": [
                    {"repo_id": REPO_ID, "revision": "abc123", "assets": ["tokenizer"]},
                    {"repo_id": REPO_ID, "revision": "abc123", "assets": ["config"]},
                ]
            }
        )

        plan = hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

        self.assertEqual(len(plan.cache), 1)
        self.assertEqual(plan.cache[0].assets, frozenset({"tokenizer", "config"}))
        self.assertEqual(plan.cache[0].revision, "abc123")

    def test_mixed_optional_fields_for_same_repo_do_not_break_sorting(self):
        # Two entries sharing a repo_id where revision/include are set on one but not the
        # other previously crashed the sort with a None-vs-value TypeError.
        workloads = {
            "workload1": {
                "downloads": {
                    "huggingface": {
                        "cache": [{"repo_id": REPO_ID, "assets": ["tokenizer"]}],
                        "repos": [{"repo_id": REPO_ID, "repo_type": "model", "name": "snap"}],
                    }
                }
            },
            "workload2": {
                "downloads": {
                    "huggingface": {
                        "cache": [{"repo_id": REPO_ID, "revision": "v1", "assets": ["config"]}],
                        "repos": [{"repo_id": REPO_ID, "repo_type": "model", "name": "snap", "include": ["*.json"]}],
                    }
                }
            },
        }

        plan = hf.build_hf_download_plan(workloads, ["workload1", "workload2"])

        self.assertEqual({(s.revision or "") for s in plan.cache}, {"", "v1"})
        self.assertEqual({(s.include or ()) for s in plan.repos}, {(), ("*.json",)})

    def test_repo_exact_dedup(self):
        item = {"repo_id": REPO_ID, "repo_type": "model", "target": "shared", "name": "shared-copy"}
        workloads = _workloads_with_downloads({"huggingface": {"repos": [item, dict(item)]}})

        plan = hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

        self.assertEqual(len(plan.repos), 1)

    def test_cache_entry_rejects_repo_download_keys(self):
        workloads = _workloads_with_downloads(
            {"huggingface": [{"repo_id": REPO_ID, "snapshot": {"name": "model-files"}}]}
        )

        with self.assertRaisesRegex(ValueError, "downloads.huggingface.repos"):
            hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

    def test_huggingface_map_rejects_unknown_keys(self):
        workloads = _workloads_with_downloads({"huggingface": {"snapshots": [{"repo_id": REPO_ID}]}})

        with self.assertRaisesRegex(ValueError, "cache.*repos"):
            hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

    def test_same_repo_destination_with_different_source_fails(self):
        workloads = {
            "workload1": {
                "downloads": {
                    "huggingface": {
                        "repos": [
                            {"repo_id": "org/source-a", "repo_type": "model", "target": "shared", "name": "shared"}
                        ]
                    }
                }
            },
            "workload2": {
                "downloads": {
                    "huggingface": {
                        "repos": [
                            {"repo_id": "org/source-b", "repo_type": "model", "target": "shared", "name": "shared"}
                        ]
                    }
                }
            },
        }
        plan = hf.build_hf_download_plan(workloads, ["workload1", "workload2"])

        with tempfile.TemporaryDirectory() as install_path:
            with self.assertRaises(ValueError):
                hf._dedup_and_validate_repo_destinations(plan.repos, install_path)

    def test_same_repo_destination_with_same_source_collapses(self):
        workloads = {
            "workload1": {
                "downloads": {
                    "huggingface": {
                        "repos": [{"repo_id": "org/source", "repo_type": "model", "target": "shared", "name": "shared"}]
                    }
                }
            },
            "workload2": {
                "downloads": {
                    "huggingface": {
                        "repos": [{"repo_id": "org/source", "repo_type": "model", "target": "shared", "name": "shared"}]
                    }
                }
            },
        }
        plan = hf.build_hf_download_plan(workloads, ["workload1", "workload2"])

        with tempfile.TemporaryDirectory() as install_path:
            deduped = hf._dedup_and_validate_repo_destinations(plan.repos, install_path)

        self.assertEqual(len(deduped), 1)

    def test_shared_target_resolves_to_datasets_dir(self):
        spec = hf.HuggingFaceRepoSpec(
            workload_key=WORKLOAD_KEY, repo_id="org/data", repo_type="dataset", target="shared", name="data-copy"
        )
        with tempfile.TemporaryDirectory() as install_path:
            local_dir = hf.resolve_huggingface_repo_local_dir(spec, install_path)

            self.assertEqual(local_dir, str((Path(install_path) / "datasets" / "data-copy").resolve()))

    def test_repo_destination_must_be_subdirectory(self):
        for name in (".", "..", "../escape"):
            spec = hf.HuggingFaceRepoSpec(
                workload_key=WORKLOAD_KEY, repo_id="org/data", repo_type="model", target="shared", name=name
            )
            with tempfile.TemporaryDirectory() as install_path:
                with self.assertRaises(ValueError):
                    hf.resolve_huggingface_repo_local_dir(spec, install_path)


class TestHuggingFaceDownloadPhase(unittest.TestCase):
    """Tests for HuggingFace download orchestration."""

    def test_cache_download_uses_expected_patterns(self):
        specs = _asset_specs(REPO_ID, "tokenizer")
        with tempfile.TemporaryDirectory() as cache_dir:
            with (
                patch("huggingface_hub.snapshot_download") as mock_snapshot,
                patch("huggingface_hub.login") as mock_login,
                patch.dict(os.environ, {"LLMB_HF_MAX_WORKERS": "4"}),
            ):
                mock_snapshot.return_value = cache_dir
                # Avoid network access: the download phase may try to write tokenizer
                # metadata if it's not present in the snapshot output.
                Path(os.path.join(cache_dir, "tokenizer_config.json")).write_text("{}")

                hf.download_huggingface_cache_assets(specs, cache_dir)
                _assert_hf_env(self, cache_dir)

        mock_login.assert_not_called()
        mock_snapshot.assert_called_once()
        snapshot_kwargs = mock_snapshot.call_args.kwargs
        self.assertEqual(snapshot_kwargs["repo_id"], REPO_ID)
        self.assertIn("allow_patterns", snapshot_kwargs)
        self.assertIn("ignore_patterns", snapshot_kwargs)
        self.assertEqual(snapshot_kwargs["cache_dir"], os.path.join(cache_dir, "hub"))
        self.assertEqual(snapshot_kwargs["max_workers"], 4)
        self.assertIsInstance(snapshot_kwargs["allow_patterns"], list)
        self.assertIsInstance(snapshot_kwargs["ignore_patterns"], list)
        self.assertTrue(snapshot_kwargs["allow_patterns"])
        self.assertTrue(snapshot_kwargs["ignore_patterns"])
        self.assertTrue(all(isinstance(item, str) for item in snapshot_kwargs["allow_patterns"]))
        self.assertTrue(all(isinstance(item, str) for item in snapshot_kwargs["ignore_patterns"]))
        self.assertIn("*.jinja", snapshot_kwargs["allow_patterns"])
        self.assertIn("chat_template*", snapshot_kwargs["allow_patterns"])
        self.assertIn("chat_templates/*", snapshot_kwargs["allow_patterns"])

    def test_cache_download_scopes_chat_template_patterns_to_tokenizers(self):
        specs = _asset_specs(REPO_ID, "config")
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch("huggingface_hub.snapshot_download") as mock_snapshot:
                mock_snapshot.return_value = cache_dir

                hf.download_huggingface_cache_assets(specs, cache_dir)

        allow_patterns = mock_snapshot.call_args.kwargs["allow_patterns"]
        self.assertNotIn("*.jinja", allow_patterns)
        self.assertNotIn("chat_template*", allow_patterns)
        self.assertNotIn("chat_templates/*", allow_patterns)

    def test_cache_download_authenticates_with_token(self):
        """Token is passed explicitly to snapshot_download (no login() call)."""
        specs = _asset_specs(REPO_ID, "tokenizer")
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch("huggingface_hub.snapshot_download") as mock_snapshot:
                mock_snapshot.return_value = cache_dir
                # Avoid network access (see note in other download-phase test).
                Path(os.path.join(cache_dir, "tokenizer_config.json")).write_text("{}")

                hf.download_huggingface_cache_assets(specs, cache_dir, token="hf_test_token")

        # Token should be passed explicitly to snapshot_download
        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args.kwargs
        self.assertEqual(call_kwargs.get("token"), "hf_test_token")

    def test_repo_download_uses_configured_paths_and_cache(self):
        spec = hf.HuggingFaceRepoSpec(
            workload_key=WORKLOAD_KEY,
            repo_id="org/dataset-repo",
            repo_type="dataset",
            revision="main",
            target="shared",
            name="dataset-copy",
            include=("*.jsonl", "*.txt"),
            exclude=("scratch/*",),
        )
        with tempfile.TemporaryDirectory() as install_path:
            cache_dir = os.path.join(install_path, ".cache", "huggingface")
            with (
                patch("huggingface_hub.snapshot_download") as mock_snapshot,
                patch.dict(os.environ, {"LLMB_HF_MAX_WORKERS": "4"}),
            ):
                mock_snapshot.return_value = os.path.join(install_path, "datasets", "dataset-copy")

                hf.download_huggingface_repos((spec,), install_path, cache_dir, token="hf_test_token")

        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args.kwargs
        self.assertEqual(call_kwargs["repo_id"], "org/dataset-repo")
        self.assertEqual(call_kwargs["repo_type"], "dataset")
        self.assertEqual(call_kwargs["revision"], "main")
        self.assertEqual(call_kwargs["cache_dir"], os.path.join(cache_dir, "hub"))
        self.assertEqual(call_kwargs["token"], "hf_test_token")
        self.assertEqual(call_kwargs["max_workers"], 4)
        expected_local_dir = str((Path(install_path) / "datasets" / "dataset-copy").resolve())
        self.assertEqual(call_kwargs["local_dir"], expected_local_dir)
        self.assertEqual(call_kwargs["allow_patterns"], ["*.jsonl", "*.txt"])
        self.assertEqual(call_kwargs["ignore_patterns"], ["scratch/*"])


class TestFinalizeTokenizerSnapshot(unittest.TestCase):
    def test_skips_when_tokenizer_config_present(self):
        with tempfile.TemporaryDirectory() as snapshot_dir:
            Path(snapshot_dir, "tokenizer_config.json").write_text("{}")
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
                hf._maybe_finalize_tokenizer_snapshot(REPO_ID, {"tokenizer"}, snapshot_dir, None)

        mock_from_pretrained.assert_not_called()

    def test_generates_metadata_when_tokenizer_config_missing(self):
        with tempfile.TemporaryDirectory() as snapshot_dir:
            mock_tokenizer = MagicMock()
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                hf._maybe_finalize_tokenizer_snapshot(REPO_ID, {"tokenizer"}, snapshot_dir, None)

        mock_tokenizer.save_pretrained.assert_called_once_with(snapshot_dir)


class TestHuggingFaceVerifyPhase(unittest.TestCase):
    """Tests for HuggingFace offline verification."""

    def test_verify_calls_expected_assets_with_offline_flags(self):
        specs = _asset_specs(REPO_ID, "tokenizer", "config")
        with tempfile.TemporaryDirectory() as cache_dir:
            with (
                patch("transformers.AutoTokenizer") as mock_auto_tokenizer,
                patch("transformers.AutoConfig") as mock_auto_config,
            ):
                hf.verify_huggingface_cache_assets(specs, cache_dir)

        _assert_offline_call(self, mock_auto_tokenizer, REPO_ID)
        _assert_offline_call(self, mock_auto_config, REPO_ID)
        _assert_hf_env(self, cache_dir)

    def test_nemotron_tokenizer_only_skips_config_verification(self):
        specs = _asset_specs(NEMOTRON_REPO_ID, "tokenizer")
        with tempfile.TemporaryDirectory() as cache_dir:
            with (
                patch("transformers.AutoTokenizer") as mock_auto_tokenizer,
                patch("transformers.AutoConfig") as mock_auto_config,
            ):
                hf.verify_huggingface_cache_assets(specs, cache_dir)

        mock_auto_tokenizer.from_pretrained.assert_called_once()
        mock_auto_config.from_pretrained.assert_not_called()


class TestXetConcurrencyNotPinned(unittest.TestCase):
    """Adaptive Xet streams stay enabled unless the user explicitly pins them."""

    _VAR = "HF_XET_FIXED_DOWNLOAD_CONCURRENCY"

    def test_does_not_pin_stream_concurrency_by_default(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            hf.set_hf_environment(cache_dir)
        self.assertIsNone(os.environ.get(self._VAR))

    def test_leaves_user_override_untouched(self):
        os.environ[self._VAR] = "8"
        with tempfile.TemporaryDirectory() as cache_dir:
            hf.set_hf_environment(cache_dir)
        self.assertEqual(os.environ.get(self._VAR), "8")


class TestAddressSpaceLimit(unittest.TestCase):
    def _limit(self, address_space, data, hard=resource.RLIM_INFINITY):
        limits = {
            resource.RLIMIT_AS: (address_space, hard),
            resource.RLIMIT_DATA: (data, hard),
        }
        with patch.object(resource, "getrlimit", side_effect=lambda res: limits[res]):
            return hf._address_space_limit()

    def test_unlimited(self):
        self.assertIsNone(self._limit(resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    def test_address_space_limit(self):
        self.assertEqual(self._limit(8 * 1024**3, resource.RLIM_INFINITY), 8 * 1024**3)

    def test_data_limit(self):
        self.assertEqual(self._limit(resource.RLIM_INFINITY, 8 * 1024**3), 8 * 1024**3)

    def test_tighter_limit_wins(self):
        self.assertEqual(self._limit(16 * 1024**3, 8 * 1024**3), 8 * 1024**3)

    def test_soft_limit_is_used(self):
        self.assertEqual(self._limit(8 * 1024**3, 16 * 1024**3, hard=32 * 1024**3), 8 * 1024**3)

    def test_zero_is_a_finite_limit(self):
        self.assertEqual(self._limit(0, resource.RLIM_INFINITY), 0)


class TestHfDownloadSettings(unittest.TestCase):
    """Cap Xet's settings without duplicating its host/cgroup memory detection."""

    _GIB = 1024**3

    @staticmethod
    def _buffers(base, per_file, ceiling):
        return {
            "reconstruction.download_buffer_size": base,
            "reconstruction.download_buffer_perfile_size": per_file,
            "reconstruction.download_buffer_limit": ceiling,
        }

    def setUp(self):
        # Representative 64 GiB-visible / 8 GiB-process-limit host.
        self.derived = self._buffers(4 * self._GIB, self._GIB, 16 * self._GIB)
        self.caps = self._buffers(self._GIB // 2, self._GIB // 8, 2 * self._GIB)
        self.effective = dict(self.caps)
        self.configs = []
        config_patcher = patch("hf_xet.XetConfig", side_effect=self._config)
        self.config_constructor = config_patcher.start()
        self.addCleanup(config_patcher.stop)
        limit_patcher = patch.object(resource, "getrlimit", return_value=(8 * self._GIB, resource.RLIM_INFINITY))
        self.getrlimit = limit_patcher.start()
        self.addCleanup(limit_patcher.stop)

    @staticmethod
    def _env_name(key):
        return "HF_XET_" + key.upper().replace(".", "_")

    def _config(self):
        # Supply only the initial and post-capping values needed by production.
        # Xet's own environment parsing is covered by the real-session test below.
        values = self.derived if not self.configs else self.effective
        self.configs.append(values)
        return values

    def _resolve(self):
        self.configs = []
        hf.set_hf_environment("/test/hf-cache")
        return hf._get_download_settings()

    def _assert_caps_in_environment(self):
        for key, value in self.caps.items():
            self.assertEqual(os.environ[self._env_name(key)], str(value))

    def _assert_capped(self):
        settings = self._resolve()
        self.assertEqual(settings.max_workers, 8)
        self.assertEqual(settings.buffer_sizes, tuple(self.caps.values()))
        self._assert_caps_in_environment()

    def _assert_unchanged(self):
        settings = self._resolve()
        self.assertEqual(settings.max_workers, 8)
        self.assertIsNone(settings.buffer_sizes)
        for key in self.derived:
            self.assertNotIn(self._env_name(key), os.environ)

    def test_64g_visible_8g_process_limit_caps_buffers_not_workers(self):
        self._assert_capped()

    def test_30g_visible_8g_process_limit_also_caps_buffers(self):
        # A target below the whole AS limit can still leave too little headroom.
        self.derived = self._buffers(1_872_000_000, 464_000_000, 7_496_000_000)
        self._assert_capped()

    def test_9g_visible_8g_process_limit_caps_only_oversized_values(self):
        self.derived = self._buffers(600_000_000, 128_000_000, 2_408_000_000)
        expected = dict(self.caps)
        expected["reconstruction.download_buffer_perfile_size"] = 128_000_000
        self.effective = dict(expected)
        settings = self._resolve()
        self.assertIsNotNone(settings.buffer_sizes)
        for key, value in expected.items():
            if key == "reconstruction.download_buffer_perfile_size":
                self.assertNotIn(self._env_name(key), os.environ)
            else:
                self.assertEqual(os.environ[self._env_name(key)], str(value))

    def test_smaller_cgroup_buffers_are_never_increased(self):
        self.derived = self._buffers(264_000_000, 64_000_000, 1_072_000_000)
        self._assert_unchanged()

    def test_upstream_process_aware_buffers_need_no_overrides(self):
        self.derived = dict(self.caps)
        self._assert_unchanged()
        self.config_constructor.assert_called_once()

    def test_unlimited_host_keeps_defaults_without_probing_xet(self):
        self.getrlimit.return_value = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        self._assert_unchanged()
        self.config_constructor.assert_not_called()

    def test_matching_large_process_limit_keeps_defaults(self):
        self.getrlimit.return_value = (64 * self._GIB, resource.RLIM_INFINITY)
        self._assert_unchanged()

    def test_zero_limit_is_not_treated_as_unlimited(self):
        self.getrlimit.return_value = (0, resource.RLIM_INFINITY)
        self.effective = self._buffers(0, 0, 0)
        self.assertIsNotNone(self._resolve().buffer_sizes)
        for key in self.effective:
            self.assertEqual(os.environ[self._env_name(key)], "0")

    def test_fraction_caps_use_exact_bytes_without_rounding_up(self):
        process_limit = 8 * self._GIB + 123
        self.getrlimit.return_value = (process_limit, resource.RLIM_INFINITY)
        expected = self._buffers(process_limit // 16, process_limit // 64, process_limit // 4)
        self.effective = dict(expected)
        self._resolve()
        for key, value in expected.items():
            self.assertEqual(os.environ[self._env_name(key)], str(value))

    def test_each_buffer_is_capped_independently(self):
        for key in self.derived:
            with self.subTest(key=key), patch.object(hf, "_download_settings", None), patch.dict(os.environ):
                self.derived = self._buffers(64_000_000, 16_000_000, 264_000_000)
                self.derived[key] = self.caps[key] * 2
                self.effective = dict(self.derived)
                self.effective[key] = self.caps[key]
                settings = self._resolve()
                self.assertIsNotNone(settings.buffer_sizes)
                set_names = {self._env_name(k) for k in self.derived if self._env_name(k) in os.environ}
                self.assertEqual(set_names, {self._env_name(key)})
                self.assertEqual(os.environ[self._env_name(key)], str(self.caps[key]))

    def test_worker_overrides_do_not_change_buffer_caps(self):
        for workers in (1, 4, 8, 64):
            with self.subTest(workers=workers), patch.object(hf, "_download_settings", None), patch.dict(os.environ):
                os.environ["LLMB_HF_MAX_WORKERS"] = str(workers)
                settings = self._resolve()
                self.assertEqual(settings.max_workers, workers)
                self._assert_caps_in_environment()

    def test_invalid_workers_use_eight_even_when_buffers_are_capped(self):
        for raw in ("not-an-int", "0", "-1"):
            with self.subTest(raw=raw), patch.object(hf, "_download_settings", None), patch.dict(os.environ):
                os.environ["LLMB_HF_MAX_WORKERS"] = raw
                with patch.object(hf.logger, "warning") as warning:
                    self.assertEqual(self._resolve().max_workers, 8)
                warning.assert_called_once()

    def test_invalid_workers_use_eight_on_unlimited_host(self):
        self.getrlimit.return_value = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        os.environ["LLMB_HF_MAX_WORKERS"] = "0"
        self.assertEqual(self._resolve().max_workers, 8)

    def test_explicit_per_buffer_values_are_preserved_above_or_below_caps(self):
        for key, cap in self.caps.items():
            for value in (cap // 2, cap * 2):
                with (
                    self.subTest(key=key, value=value),
                    patch.object(hf, "_download_settings", None),
                    patch.dict(os.environ),
                ):
                    name = self._env_name(key)
                    os.environ[name] = str(value)
                    self._resolve()
                    self.assertEqual(os.environ[name], str(value))

    def test_all_explicit_buffer_values_are_left_alone(self):
        for key, value in self.derived.items():
            os.environ[self._env_name(key)] = str(value)
        settings = self._resolve()
        self.assertIsNone(settings.buffer_sizes)
        for key, value in self.derived.items():
            self.assertEqual(os.environ[self._env_name(key)], str(value))

    def test_disabled_xet_is_not_probed_or_capped(self):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        settings = self._resolve()
        self.assertEqual(settings.max_workers, 8)
        self.assertIsNone(settings.buffer_sizes)
        self.config_constructor.assert_not_called()
        for key in self.derived:
            self.assertNotIn(self._env_name(key), os.environ)

    def test_cache_environment_precedes_config_inspection(self):
        def config_after_cache_setup():
            self.assertEqual(os.environ["HF_HOME"], "/test/hf-cache")
            self.assertEqual(os.environ["HF_HUB_CACHE"], "/test/hf-cache/hub")
            return self._config()

        self.config_constructor.side_effect = config_after_cache_setup
        with patch("builtins.print") as output:
            self._resolve()
        output.assert_not_called()

    def test_settings_are_resolved_once_and_caps_are_retained(self):
        settings = self._resolve()
        env = {self._env_name(key): os.environ[self._env_name(key)] for key in self.caps}
        os.environ["LLMB_HF_MAX_WORKERS"] = "64"
        hf.set_hf_environment("/test/another-cache")
        self.assertIs(hf._get_download_settings(), settings)
        self.assertEqual(hf._hf_max_workers(), 8)
        self.assertEqual({name: os.environ[name] for name in env}, env)
        self.assertEqual(self.getrlimit.call_count, 2)
        self.assertEqual(self.config_constructor.call_count, 2)

    def test_normal_announcement_is_one_line_printed_once(self):
        self.getrlimit.return_value = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        self._resolve()
        with patch("builtins.print") as output:
            hf._announce_download_settings()
            hf._announce_download_settings()
        self.assertEqual(output.call_count, 1)
        self.assertEqual(
            output.call_args_list[0].args[0],
            "HuggingFace downloads: up to 8 workers (adjust with LLMB_HF_MAX_WORKERS).",
        )
        self.assertIn("LLMB_HF_MAX_WORKERS", str(output.call_args_list))

    def test_capped_announcement_is_concise_and_diagnostics_are_debug_only(self):
        self._resolve()
        with self.assertLogs(hf.logger, level="DEBUG") as diagnostics, patch("builtins.print") as output:
            hf._announce_download_settings()
            hf._announce_download_settings()
        self.assertEqual(output.call_count, 1)
        self.assertEqual(
            output.call_args_list[0].args[0],
            "HuggingFace downloads: up to 8 workers (adjust with LLMB_HF_MAX_WORKERS).",
        )
        user_output = str(output.call_args_list)
        self.assertNotIn("MiB", user_output)
        self.assertNotIn("GiB", user_output)
        self.assertNotIn("Xet", user_output)
        self.assertIn("LLMB_HF_MAX_WORKERS", user_output)
        self.assertEqual(len(diagnostics.records), 1)
        self.assertEqual(
            diagnostics.records[0].getMessage(),
            "HuggingFace downloads: detected process memory limit 8 GiB; "
            "capped Xet buffers to shared=512 MiB, per-file=128 MiB, ceiling=2048 MiB.",
        )
        self.assertEqual(self.config_constructor.call_count, 2)  # Announcing never probes Xet again.

    def test_info_level_does_not_emit_memory_diagnostics(self):
        self._resolve()
        with self.assertNoLogs(hf.logger, level="INFO"), patch("builtins.print") as output:
            hf._announce_download_settings()
        self.assertEqual(output.call_count, 1)
        self.assertNotIn("MiB", str(output.call_args_list))

    def test_announcement_reports_worker_override(self):
        os.environ["LLMB_HF_MAX_WORKERS"] = "64"
        self._resolve()
        with patch("builtins.print") as output:
            hf._announce_download_settings()
        self.assertIn("up to 64 workers", output.call_args_list[0].args[0])

    def test_cache_and_repo_downloads_share_one_decision_and_announcement(self):
        repo = hf.HuggingFaceRepoSpec(
            workload_key=WORKLOAD_KEY, repo_id="openai/gpt-oss-120b", repo_type="model", name="gpt-oss-120b"
        )
        with tempfile.TemporaryDirectory() as install_path:
            cache_dir = os.path.join(install_path, ".cache", "huggingface")
            with (
                patch("huggingface_hub.snapshot_download", return_value=cache_dir) as snapshot,
                patch("builtins.print") as output,
            ):
                hf.download_huggingface_cache_assets(_asset_specs(REPO_ID, "config"), cache_dir)
                hf.download_huggingface_repos((repo,), install_path, cache_dir)
        self.assertEqual(snapshot.call_count, 2)
        self.assertEqual([call.kwargs["max_workers"] for call in snapshot.call_args_list], [8, 8])
        announcements = [
            call.args[0]
            for call in output.call_args_list
            if call.args and str(call.args[0]).startswith("HuggingFace downloads:")
        ]
        self.assertEqual(len(announcements), 1)
        self.assertEqual(self.config_constructor.call_count, 2)
        self.assertEqual(self.getrlimit.call_count, 2)


class TestXetSessionBufferCaps(unittest.TestCase):
    def test_actual_hub_session_receives_caps_without_network_or_allocator_tuning(self):
        # A fresh interpreter avoids the mocks/imports/cached session in other tests.
        # Only simulate the limit reading: no real AS cap or large data allocation.
        code = textwrap.dedent(r"""
            import os
            import re
            import tempfile
            from llmb_install.downloads import huggingface as hf

            hf._address_space_limit = lambda: 8 * 1024**3
            # Force a known large baseline even on a small test host. This flag
            # remains set, but its non-explicit buffer values must be capped.
            os.environ["HF_XET_DISABLE_MEMORY_DERIVED_DOWNLOAD_BUFFERS"] = "1"
            with tempfile.TemporaryDirectory() as cache:
                hf.set_hf_environment(cache)
                from huggingface_hub.utils._xet import get_xet_session, abort_xet_session

                session = get_xet_session()
                actual = dict(re.findall(r"(reconstruction[.]download_buffer_\w+)=(\d+)", repr(session)))
                assert actual == {
                    "reconstruction.download_buffer_size": str(512 * 1024**2),
                    "reconstruction.download_buffer_perfile_size": str(128 * 1024**2),
                    "reconstruction.download_buffer_limit": str(2 * 1024**3),
                }, actual
                assert "client.ac_max_download_concurrency=64" in repr(session)
                assert hf._hf_max_workers() == 8
                assert "HF_XET_FIXED_DOWNLOAD_CONCURRENCY" not in os.environ
                abort_xet_session()
            """)
        env = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("HF_XET_", "HF_HUB_", "LLMB_HF_")) and name != "MALLOC_ARENA_MAX"
        }
        env["PYTHONPATH"] = str(Path(hf.__file__).resolve().parents[2])
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["TOKIO_WORKER_THREADS"] = "2"  # Keep this startup-only test small on large CI hosts.
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
