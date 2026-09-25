#!/usr/bin/env python3
"""Print a compatibility fingerprint for compiled FoundationPose dependencies."""
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def command_version(command: str, args: list[str]) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        result = subprocess.run([executable, *args], text=True, capture_output=True, timeout=10)
        output = (result.stdout + result.stderr).strip().splitlines()
        return output[0].strip() if output else f"exit={result.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"


def detect() -> dict:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"PyTorch must already be installed in the Colab runtime: {exc}")

    gpu_name = "cpu"
    capability = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        capability = f"sm{major}{minor}"
    try:
        abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except (AttributeError, TypeError):
        abi = None
    lock_path = ROOT / "colab_sources.json"
    sources = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    data = {
        "python": platform.python_version(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "cpu",
        "torch_cxx11_abi": abi,
        "gpu_name": gpu_name,
        "compute_capability": capability,
        "nvcc": command_version("nvcc", ["--version"]),
        "gcc": command_version("gcc", ["--version"]),
        "g++": command_version("g++", ["--version"]),
        "sources": {name: entry["commit"] for name, entry in sources.items()},
    }
    safe = lambda value: re.sub(r"[^A-Za-z0-9]+", "", str(value))
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_id = safe(torch.__version__)
    cuda_id = "cu" + safe(torch.version.cuda) if torch.version.cuda else "cpu"
    data["fingerprint"] = "_".join((py, f"torch{torch_id}", cuda_id, capability,
                                     f"abi{abi if abi is not None else 'unknown'}"))
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprint", action="store_true", help="print only the cache key")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    data = detect()
    if args.fingerprint:
        print(data["fingerprint"])
    elif args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        for key, value in data.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
