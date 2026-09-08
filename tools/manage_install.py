"""User-local installation with file ownership receipts and guarded removal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

PROJECT = "intel-core-ultra-llama-linux"
SOURCE = Path(__file__).resolve().parent.parent
COMMANDS = [f"aag-llama-{mode}-{action}" for mode in ("server", "cli") for action in ("start", "stop", "status")]


def fingerprint(path):
    if path.is_symlink():
        return {"link": os.readlink(path)}
    if path.is_file():
        return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mode": path.stat().st_mode & 0o777}
    return None


def safe_parents(path):
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"Refusing symlink directory: {parent}")


def write(path, data, mode=0o644):
    safe_parents(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".install-tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
    temporary.chmod(mode)
    temporary.replace(path)


def detect():
    distro = {}
    os_release = Path("/etc/os-release")
    if os_release.exists():
        for line in os_release.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                distro[key] = value.strip('"')
    cpu = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    devices = list(Path("/sys/class/drm").glob("card[0-9]*/device/vendor"))
    intel = any(path.read_text().strip() == "0x8086" for path in devices)
    print("Linux distribution:", distro.get("PRETTY_NAME", sys.platform))
    print("Intel Core Ultra:", "detected" if "Core(TM) Ultra" in cpu or "Core Ultra" in cpu else "not identified")
    print("Intel DRM GPU:", "detected" if intel else "not detected (launcher installation is still allowed)")
    if shutil.which("lspci"):
        output = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10).stdout
        for line in output.splitlines():
            if "Intel" in line and any(word in line for word in ("VGA", "Display", "3D")):
                print("GPU:", line.split(": ", 1)[-1])
    if distro.get("ID") != "ubuntu":
        print("Ubuntu is the primary target; other systemd Linux distributions are best effort.")


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "uninstall"))
    parser.add_argument("--prefix", type=Path, help="Default: ~/.local; custom prefixes keep config under PREFIX/config.")
    parser.add_argument("--llama-root", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--oneapi-env", type=Path)
    parser.add_argument("--replace", action="store_true", help="Back up and replace conflicting files.")
    return parser.parse_args()


def install(args, prefix, destination, manifest):
    detect()
    safe_parents(destination)
    old = json.loads(manifest.read_text()) if manifest.exists() else {"files": {}}
    if old.get("prefix", str(prefix)) != str(prefix):
        raise ValueError("Installation receipt prefix does not match")
    config_dir = Path(old["config_dir"]) if "config_dir" in old else (
        prefix / "config" / PROJECT if args.prefix else Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / PROJECT)
    config_dir = config_dir.absolute()
    safe_parents(config_dir)
    llama = args.llama_root
    if llama is None:
        found = shutil.which("llama-server")
        llama = Path(found).resolve().parents[2] if found and Path(found).resolve().parent.name == "bin" else Path.home() / "llama.cpp"
    llama = llama.expanduser().absolute()
    build = (args.build_dir or llama / "build-sycl").expanduser().absolute()
    models = (args.model_root or Path.home() / "Models").expanduser().absolute()
    oneapi = (args.oneapi_env or Path("/opt/intel/oneapi/setvars.sh")).expanduser().absolute()
    plan = {}
    for pattern in ("bin/*", "lib/*.py", "lib/*.sh", "tools/manage_install.py", "docs/*.md", "config/*.conf", "README.md", "LICENSE", "THIRD-PARTY-NOTICES.md", "uninstall.sh"):
        for source in SOURCE.glob(pattern):
            if source.is_file() or source.is_symlink():
                plan[destination / source.relative_to(SOURCE)] = source
    config = config_dir / "config.conf"
    config_text = "# User-owned shell configuration. See profiles.example.conf.\n"
    config_text += "".join(key + "=" + shlex.quote(str(value)) + "\n" for key, value in (
        ("LLAMA_ROOT", llama), ("BUILD_DIR", build), ("MODEL_ROOT", models), ("ONEAPI_ENV", oneapi)))
    if not config.exists() and not config.is_symlink():
        plan[config] = config_text.encode()
    for command in COMMANDS:
        wrapper = "#!/usr/bin/env bash\nset -euo pipefail\n"
        wrapper += "if [[ -z \"${AAG_LLAMA_CONFIG:-}\" ]]; then export AAG_LLAMA_CONFIG=" + shlex.quote(str(config)) + "; fi\n"
        wrapper += "exec " + shlex.quote(str(destination / "bin" / command)) + ' "$@"\n'
        plan[prefix / "bin" / command] = wrapper.encode()
    for target in plan:
        safe_parents(target.parent)
        current = fingerprint(target)
        owned = old["files"].get(str(target))
        if target.exists() and current is None:
            raise ValueError(f"Cannot replace a directory: {target}")
        if current and (not owned or current != owned["installed"]) and not args.replace:
            raise ValueError(f"Existing file differs: {target}. Use --replace to back it up first.")
    records = old["files"].copy()
    backup_dir = destination / ".backups" / str(time.time_ns())
    for index, (target, source) in enumerate(plan.items()):
        current = fingerprint(target)
        original = records.get(str(target), {}).get("original")
        if current:
            backup = backup_dir / str(index)
            safe_parents(backup.parent)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                backup.symlink_to(os.readlink(target))
            else:
                shutil.copy2(target, backup)
            if str(target) not in records:
                original = {"path": str(backup), "fingerprint": current}
        if isinstance(source, bytes):
            write(target, source, 0o755 if target.parent == prefix / "bin" else 0o600)
        elif isinstance(source, dict) or source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            if current:
                target.unlink()
            target.symlink_to(source["link"] if isinstance(source, dict) else os.readlink(source))
        else:
            write(target, source.read_bytes(), source.stat().st_mode & 0o777)
        records[str(target)] = {"installed": fingerprint(target), "original": original}
    write(manifest, (json.dumps({"prefix": str(prefix), "config_dir": str(config_dir), "files": records}, indent=2) + "\n").encode(), 0o600)
    print("Installed launcher commands in", prefix / "bin")
    print("Configuration:", config)
    print("Add this directory to PATH if needed:", prefix / "bin")
    print("Next: aag-llama-server-start --check-runtime")
    print("Then: aag-llama-server-start --dry-run")
    if not (build / "bin" / "llama-server").is_file():
        print("llama-server is missing; configure BUILD_DIR after following docs/INTEL-SYCL.md.")
    if not oneapi.is_file() and not shutil.which("sycl-ls"):
        print("oneAPI/SYCL is missing. No packages were installed. See docs/INTEL-SYCL.md.")
    print("Uninstall:", str(destination / "uninstall.sh"), "--prefix", str(prefix))


def uninstall(prefix, destination, manifest):
    safe_parents(destination)
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("No safe installation receipt exists for this prefix")
    receipt = json.loads(manifest.read_text())
    if receipt["prefix"] != str(prefix):
        raise ValueError("Installation receipt prefix does not match")
    config = Path(receipt["config_dir"]) / "config.conf"
    records = receipt["files"]
    for name, record in records.items():
        path = Path(name)
        if not path.is_relative_to(destination) and path != config and path not in [prefix / "bin" / command for command in COMMANDS]:
            raise ValueError("Unsafe receipt path")
        safe_parents(path.parent)
        original = record.get("original")
        if original:
            backup = Path(original["path"])
            if not backup.is_relative_to(destination / ".backups") or fingerprint(backup) != original["fingerprint"]:
                raise ValueError("Backup missing or changed; refusing partial uninstall")
    preserved = []
    for name, record in records.items():
        path = Path(name)
        if fingerprint(path) != record["installed"]:
            if path.exists() or path.is_symlink():
                preserved.append(name)
            continue
        path.unlink()
        original = record.get("original")
        if original:
            backup = Path(original["path"])
            if backup.is_symlink():
                path.symlink_to(os.readlink(backup))
            else:
                shutil.copy2(backup, path)
    manifest.unlink()
    for directory in sorted({Path(name).parent for name in records} | {destination}, key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    print("Uninstalled unchanged project files. Models, llama.cpp, services, and logs were not touched.")
    for path in preserved:
        print("Preserved modified file:", path)
    if (destination / ".backups").exists():
        print("Replacement backups retained in", destination / ".backups")


def main():
    args = parse()
    if sys.platform != "linux":
        raise ValueError("Linux is required")
    prefix = (args.prefix or Path.home() / ".local").expanduser().absolute()
    safe_parents(prefix)
    destination = prefix / "share" / PROJECT
    manifest = destination / ".install-manifest.json"
    if args.action == "install":
        install(args, prefix, destination, manifest)
    else:
        uninstall(prefix, destination, manifest)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit("Installation error: " + str(error))
