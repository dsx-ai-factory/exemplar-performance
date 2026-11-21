# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight process-memory setup, safe to import before the main installer."""

import os
import sys
from typing import Optional

# Keep allocator tuning scoped to the tight limits exercised in cluster QA.
# A finite but generous limit is not a reason to restrict allocator parallelism.
_ARENA_PROCESS_LIMIT_CEILING = 8 * 1024**3
_ARENA_LIMIT = 2
_M_ARENA_MAX = -8  # glibc malloc.h; this is not a portable libc parameter.
_arena_limit_requested = False
_allocator_diagnostic = "automatic arena setup not requested"


def process_memory_limit() -> Optional[int]:
    """Tightest finite soft RLIMIT_AS / RLIMIT_DATA, if supported and readable."""
    try:
        import resource
    except ImportError:
        return None

    try:
        limits = []
        for res in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            soft, _hard = resource.getrlimit(res)
            if soft != resource.RLIM_INFINITY:
                limits.append(soft)
        return min(limits) if limits else None
    except (AttributeError, OSError, ValueError):
        return None


def configure_allocator() -> bool:
    """Request fewer glibc arenas early on tightly process-limited Linux hosts.

    Call before importing Installer, NumPy, hf_xet, or other threaded libraries.
    glibc can cache its arena ceiling: a late mallopt call may return success
    without constraining new arenas, and cannot reclaim existing arenas. Python
    startup hooks may already have initialized it; this is best-effort, not proof
    that the effective arena count is bounded. The startup environment setting
    remains the workaround if this call is too late on a particular system.

    Only the current process is tuned. No environment variable is exported to
    Git, package installers, or other executed subprocesses. Explicit arena
    settings take precedence. Unsupported systems retain allocator defaults.
    """
    global _arena_limit_requested, _allocator_diagnostic
    if _arena_limit_requested:
        return True
    if sys.platform != "linux":
        _allocator_diagnostic = "allocator defaults retained (non-Linux platform)"
        return False
    if any(name in os.environ for name in ("MALLOC_ARENA_MAX", "MALLOC_ARENA_TEST")):
        _allocator_diagnostic = "explicit MALLOC_ARENA settings preserved"
        return False
    if any(
        entry.partition("=")[0] in {"glibc.malloc.arena_max", "glibc.malloc.arena_test"}
        for entry in os.environ.get("GLIBC_TUNABLES", "").split(":")
    ):
        _allocator_diagnostic = "explicit glibc arena tunables preserved"
        return False
    limit = process_memory_limit()
    if limit is None or limit > _ARENA_PROCESS_LIMIT_CEILING:
        _allocator_diagnostic = f"allocator defaults retained (process limit: {limit} bytes)"
        return False

    try:
        import ctypes

        libc = ctypes.CDLL(None)
        if getattr(libc, "gnu_get_libc_version", None) is None:
            _allocator_diagnostic = "allocator defaults retained (glibc not detected)"
            return False
        mallopt = libc.mallopt
        mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
        mallopt.restype = ctypes.c_int
        if mallopt(_M_ARENA_MAX, _ARENA_LIMIT) != 1:
            _allocator_diagnostic = "glibc arena cap request failed"
            return False
    except (ImportError, AttributeError, OSError) as exc:
        _allocator_diagnostic = f"glibc arena setup unavailable ({type(exc).__name__})"
        return False

    _arena_limit_requested = True
    _allocator_diagnostic = f"glibc arena cap requested={_ARENA_LIMIT} (process limit: {limit / 1024**3:g} GiB)"
    return True


def allocator_diagnostic() -> str:
    """Return the early setup result for debug logging after logging is configured."""
    return _allocator_diagnostic
