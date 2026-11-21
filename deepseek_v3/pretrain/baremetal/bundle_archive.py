#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize bundle links during creation, or validate all members before extraction."""

import argparse
import posixpath
import sys
import tarfile


def normalize(prefix: str) -> None:
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as source:
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as target:
            for member in source:
                if member.issym() and posixpath.isabs(member.linkname):
                    if not member.linkname.startswith(prefix.rstrip("/") + "/"):
                        raise ValueError(f"external symbolic link: {member.name} -> {member.linkname}")
                    destination = "root/" + member.linkname[len(prefix.rstrip("/")) + 1 :]
                    member.linkname = posixpath.relpath(destination, posixpath.dirname(member.name))
                    member.pax_headers.pop("linkpath", None)
                target.addfile(member, source.extractfile(member) if member.isfile() else None)


def validate(archive: str) -> None:
    with tarfile.open(archive, "r:gz") as source:
        members = {}
        for member in source:
            name = member.name
            if posixpath.isabs(name) or ".." in name.split("/"):
                raise ValueError(f"unsafe member path: {name}")
            name = posixpath.normpath(name)
            if name == "." and member.isdir():
                continue
            if name != "MANIFEST" and name != "root" and not name.startswith("root/"):
                raise ValueError(f"unexpected bundle member: {name}")
            if name in members:
                raise ValueError(f"duplicate bundle member: {name}")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError(f"unsupported member type: {name}")
            if name == "MANIFEST" and not member.isfile():
                raise ValueError("MANIFEST must be a regular file")
            if name == "root" and not member.isdir():
                raise ValueError("root must be a directory")
            if (member.issym() or member.islnk()) and posixpath.isabs(member.linkname):
                raise ValueError(f"absolute link target: {name} -> {member.linkname}")
            members[name] = member

        def resolve(parts: list[str], depth: int = 0) -> str:
            if depth > 40:
                raise ValueError("cyclic or excessively nested archive links")
            resolved = []
            while parts:
                part, *parts = parts
                if part in ("", "."):
                    continue
                if part == "..":
                    if len(resolved) <= 1:
                        raise ValueError("link escapes bundle root")
                    resolved.pop()
                    continue
                resolved.append(part)
                if resolved[0] != "root":
                    raise ValueError("link escapes bundle root")
                entry = members.get("/".join(resolved))
                if entry and (entry.issym() or entry.islnk()):
                    base = resolved[:-1] if entry.issym() else []
                    return resolve(base + entry.linkname.split("/") + parts, depth + 1)
            return "/".join(resolved)

        for name, member in members.items():
            parent = posixpath.dirname(name)
            while parent:
                if parent in members and not members[parent].isdir():
                    raise ValueError(f"non-directory ancestor of {name}: {parent}")
                parent = posixpath.dirname(parent)
            if member.islnk():
                target = members.get(member.linkname)
                if target is None or not target.isfile():
                    raise ValueError(f"hard link does not target a regular bundle file: {name}")
            elif member.issym():
                resolve(name.split("/"))
        if "MANIFEST" not in members:
            raise ValueError("bundle has no MANIFEST")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("normalize", "validate"))
    parser.add_argument("path", help="Original install prefix for normalize; archive filename for validate.")
    args = parser.parse_args()
    try:
        if args.command == "normalize":
            normalize(args.path)
        else:
            validate(args.path)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Unsafe bundle: {error}\n")


if __name__ == "__main__":
    main()
