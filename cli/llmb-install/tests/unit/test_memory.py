# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early allocator policy and fresh-process checks; no model downloads required."""

import ctypes
import logging
import os
import platform
import resource
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llmb_install.utils import memory


@pytest.fixture
def allocator(monkeypatch):
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(memory, "_arena_limit_requested", False)
    monkeypatch.setattr(memory, "_allocator_diagnostic", "automatic arena setup not requested")
    monkeypatch.setattr(memory, "process_memory_limit", Mock(return_value=8 * 1024**3))
    for name in ("MALLOC_ARENA_MAX", "MALLOC_ARENA_TEST", "GLIBC_TUNABLES"):
        monkeypatch.delenv(name, raising=False)
    libc = SimpleNamespace(gnu_get_libc_version=object(), mallopt=Mock(return_value=1))
    loader = Mock(return_value=libc)
    monkeypatch.setattr(ctypes, "CDLL", loader)
    return loader, libc


def test_requests_two_arenas_once_without_exporting_environment(allocator, capsys):
    loader, libc = allocator
    before = dict(os.environ)
    assert memory.configure_allocator()
    assert memory.configure_allocator()
    loader.assert_called_once_with(None)
    libc.mallopt.assert_called_once_with(-8, 2)
    assert libc.mallopt.argtypes == [ctypes.c_int, ctypes.c_int]
    assert libc.mallopt.restype is ctypes.c_int
    assert dict(os.environ) == before
    assert not capsys.readouterr().out
    assert memory.allocator_diagnostic() == "glibc arena cap requested=2 (process limit: 8 GiB)"


@pytest.mark.parametrize("limit", [None, 8 * 1024**3 + 1, 64 * 1024**3])
def test_leaves_unlimited_or_generous_limits_unchanged(allocator, limit):
    loader, _ = allocator
    memory.process_memory_limit.return_value = limit
    assert not memory.configure_allocator()
    loader.assert_not_called()


@pytest.mark.parametrize("platform_name", ["darwin", "win32"])
def test_non_linux_is_untouched(allocator, platform_name):
    loader, _ = allocator
    memory.sys.platform = platform_name
    assert not memory.configure_allocator()
    loader.assert_not_called()
    memory.process_memory_limit.assert_not_called()


@pytest.mark.parametrize("name", ["MALLOC_ARENA_MAX", "MALLOC_ARENA_TEST"])
@pytest.mark.parametrize("value", ["", "0", "8"])
def test_preserves_explicit_arena_environment(allocator, monkeypatch, name, value):
    loader, _ = allocator
    monkeypatch.setenv(name, value)
    assert not memory.configure_allocator()
    assert os.environ[name] == value
    loader.assert_not_called()


@pytest.mark.parametrize("setting", ["glibc.malloc.arena_max", "glibc.malloc.arena_test"])
def test_preserves_explicit_glibc_arena_tunable(allocator, monkeypatch, setting):
    loader, _ = allocator
    value = f"glibc.malloc.tcache_count=4:{setting}=0:glibc.malloc.trim_threshold=1024"
    monkeypatch.setenv("GLIBC_TUNABLES", value)
    assert not memory.configure_allocator()
    assert os.environ["GLIBC_TUNABLES"] == value
    loader.assert_not_called()


def test_unrelated_glibc_tunable_is_preserved_without_disabling_setup(allocator, monkeypatch):
    monkeypatch.setenv("GLIBC_TUNABLES", "glibc.malloc.tcache_count=4")
    assert memory.configure_allocator()
    assert os.environ["GLIBC_TUNABLES"] == "glibc.malloc.tcache_count=4"


def test_non_glibc_is_untouched(allocator):
    loader, _ = allocator
    loader.return_value = SimpleNamespace()
    assert not memory.configure_allocator()


def test_unavailable_native_function_is_nonfatal(allocator):
    loader, _ = allocator
    loader.return_value = SimpleNamespace(gnu_get_libc_version=object())
    assert not memory.configure_allocator()


def test_missing_ctypes_is_nonfatal(allocator, monkeypatch):
    loader, _ = allocator
    monkeypatch.setitem(sys.modules, "ctypes", None)
    assert not memory.configure_allocator()
    loader.assert_not_called()


def test_libc_load_failure_is_nonfatal(allocator):
    loader, _ = allocator
    loader.side_effect = OSError("unavailable")
    assert not memory.configure_allocator()


def test_failed_mallopt_does_not_report_success(allocator, capsys):
    _, libc = allocator
    libc.mallopt.return_value = 0
    assert not memory.configure_allocator()
    assert not memory._arena_limit_requested
    assert not capsys.readouterr().out


@pytest.mark.parametrize("error", [OSError("unreadable"), ValueError("unsupported")])
def test_unreadable_process_limit_is_nonfatal(monkeypatch, error):
    def fail(_res):
        raise error

    monkeypatch.setattr(resource, "getrlimit", fail)
    assert memory.process_memory_limit() is None


@pytest.mark.parametrize("verbose", [False, True])
def test_allocator_diagnostics_are_logged_after_setup_only_when_verbose(monkeypatch, caplog, capsys, verbose):
    from llmb_install.core import installer as core

    diagnostic = "glibc arena cap requested=2 (process limit: 8 GiB)"
    monkeypatch.setattr(memory, "_allocator_diagnostic", diagnostic)
    instance = core.Installer.__new__(core.Installer)
    instance.logger = logging.getLogger("allocator_diagnostic_test")
    events = []

    def setup(level):
        events.append("logging")
        caplog.set_level(level, logger=instance.logger.name)

    def stop():
        events.append("resume state")
        raise RuntimeError("stop after logging")

    monkeypatch.setattr(core, "setup_logging", setup)
    monkeypatch.setattr(core, "load_install_state", stop)
    with pytest.raises(RuntimeError, match="stop after logging"):
        instance.run(SimpleNamespace(verbose=verbose))
    assert events == ["logging", "resume state"]
    assert caplog.messages == ([f"Allocator setup: {diagnostic}"] if verbose else [])
    assert not capsys.readouterr().out


@pytest.mark.skipif(sys.platform != "linux" or platform.libc_ver()[0] != "glibc", reason="glibc-only native check")
@pytest.mark.parametrize("entry", ["console", "module"])
def test_cli_requests_arena_limit_before_installer_import_and_native_threads(entry):
    # Only simulate the limit reading; do not impose a real AS limit on CI.
    # Verify the actual allocator with malloc_info rather than trusting mallopt's
    # return value. A fresh interpreter is essential: pytest has already imported
    # native libraries and might already have initialized the arena ceiling.
    code = textwrap.dedent(r"""
        import builtins
        import ctypes
        import os
        import runpy
        import sys
        import tempfile
        import threading
        import xml.etree.ElementTree as ET
        from llmb_install.utils import memory

        memory.process_memory_limit = lambda: 8 * 1024**3
        original_import = builtins.__import__
        installer_imported = False
        def check_import(name, *args, **kwargs):
            global installer_imported
            if name == "llmb_install.core.installer":
                assert memory._arena_limit_requested, "Allocator setup ran too late"
                installer_imported = True
            return original_import(name, *args, **kwargs)

        builtins.__import__ = check_import
        entry = sys.argv[1]
        sys.argv = ["llmb-install", "--help"]
        try:
            if entry == "console":
                from llmb_install.cli import main
                main()
            else:
                runpy.run_module("llmb_install", run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 0, exc.code
        finally:
            builtins.__import__ = original_import
        assert installer_imported
        assert "MALLOC_ARENA_MAX" not in os.environ

        libc = ctypes.CDLL(None)
        for name, args, result in [
            ("malloc", [ctypes.c_size_t], ctypes.c_void_p),
            ("free", [ctypes.c_void_p], None),
            ("fopen", [ctypes.c_char_p, ctypes.c_char_p], ctypes.c_void_p),
            ("fclose", [ctypes.c_void_p], ctypes.c_int),
            ("malloc_info", [ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
        ]:
            fn = getattr(libc, name)
            fn.argtypes = args
            fn.restype = result
        barrier = threading.Barrier(17)
        def allocate():
            barrier.wait()
            blocks = [libc.malloc(32768) for _ in range(40)]
            assert all(blocks)
            barrier.wait()
            for pointer in blocks:
                libc.free(pointer)
        threads = [threading.Thread(target=allocate) for _ in range(16)]
        for thread in threads:
            thread.start()
        barrier.wait()
        barrier.wait()
        for thread in threads:
            thread.join()
        with tempfile.NamedTemporaryFile() as info:
            fp = libc.fopen(os.fsencode(info.name), b"w")
            assert fp
            assert libc.malloc_info(0, fp) == 0
            libc.fclose(fp)
            arenas = len(ET.parse(info.name).findall("heap"))
        assert arenas <= 2, f"Arena cap ineffective: {arenas} arenas"
        print(f"Native arena count verified: {arenas}")
        """)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"MALLOC_ARENA_MAX", "MALLOC_ARENA_TEST", "GLIBC_TUNABLES", "LD_PRELOAD"}
    }
    env["PYTHONPATH"] = str(Path(memory.__file__).resolve().parents[2])
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    result = subprocess.run([sys.executable, "-c", code, entry], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "glibc arena cap requested" not in result.stdout
    assert "Native arena count verified:" in result.stdout
