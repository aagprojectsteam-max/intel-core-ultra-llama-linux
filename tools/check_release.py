"""Reject private paths, binary/model artifacts, oversized files, and unsafe links."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_SUFFIXES = {".md", ".py", ".sh", ".conf", ".yml", ".yaml"}
ALLOWED_NAMES = {"LICENSE", ".gitignore", "aag-llama-control"}
PRIVATE_PATH = re.compile(rb"/home/[a-zA-Z0-9_.-]+|/(?:mnt|media)/[a-zA-Z0-9_.-]+")
SECRET = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def check(name, data, mode=None):
    path = PurePosixPath(name)
    issues = []
    if len(data) > 1024 * 1024:
        issues.append("larger than 1 MiB")
    if mode == "120000":
        target = PurePosixPath(data.decode())
        if target.is_absolute() or ".." in target.parts:
            issues.append("unsafe symlink")
    elif path.name not in ALLOWED_NAMES and path.suffix not in ALLOWED_SUFFIXES:
        issues.append("file type outside source allowlist")
    if any(part.lower() in {"evidence", "storage", "logs", "models", ".ssh"} for part in path.parts):
        issues.append("excluded directory")
    if data.startswith((b"GGUF", b"\x7fELF", b"PK\x03\x04")) or b"\0" in data:
        issues.append("binary/model payload")
    if PRIVATE_PATH.search(data):
        issues.append("machine-specific path")
    if SECRET.search(data):
        issues.append("possible secret")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        issues.append("non-text content")
    return [{"path": name, "issue": issue} for issue in issues]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    findings = []
    count = 0
    for directory, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [name for name in dirs if name != ".git"]
        for filename in files:
            path = Path(directory) / filename
            data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
            findings += check(str(path.relative_to(ROOT)), data, "120000" if path.is_symlink() else None)
            count += 1
    history_count = 0
    if args.history:
        seen = set()
        for commit in git("rev-list", "--all").splitlines():
            findings += check("commit.md", git("cat-file", "commit", commit.decode()))
            for entry in git("ls-tree", "-rz", "-r", commit.decode()).split(b"\0"):
                if not entry:
                    continue
                header, name = entry.split(b"\t", 1)
                mode, kind, oid = header.decode().split()
                key = (name, oid)
                if key in seen:
                    continue
                seen.add(key)
                if kind != "blob":
                    findings.append({"path": name.decode(), "issue": "unexpected git object"})
                    continue
                findings += check(name.decode(), git("cat-file", "blob", oid), mode)
                history_count += 1
        # Include unreachable blobs so an amended-away artifact cannot be overlooked.
        for line in git("cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype) %(objectsize)").decode().splitlines():
            oid, kind, size = line.split()
            if int(size) > 1024 * 1024:
                findings.append({"object": oid, "issue": "large history object"})
            if kind in {"blob", "commit", "tag"}:
                findings += check("object.md", git("cat-file", kind, oid))
    print(json.dumps({"status": "FAIL" if findings else "PASS", "files": count, "history_entries": history_count, "findings": findings}, indent=2))
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
