#!/usr/bin/env python3
"""Verify the imports and native artifacts used by the Colab CV pipeline."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose", type=Path, default=Path("/content/FoundationPose"))
    parser.add_argument("--cnos", type=Path, default=Path("/content/cnos"))
    args = parser.parse_args()
    fp_root = args.foundationpose.resolve()
    cnos_root = args.cnos.resolve()
    for path in (fp_root, fp_root / "mycpp", fp_root / "mycpp" / "build", cnos_root):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}")
    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; CUDA available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")

    modules = ["cv2", "numpy", "scipy", "trimesh", "open3d", "ultralytics",
               "nvdiffrast.torch", "pytorch3d", "hydra", "pyrender", "pycocotools"]
    for module in modules:
        importlib.import_module(module)
        print(f"OK import {module}")

    import warp as wp
    wp.init()
    if wp.__version__ != "1.17.0":
        raise RuntimeError(f"Expected warp-lang 1.17.0, found {wp.__version__}")
    import mycpp
    print(f"OK Warp {wp.__version__}; mycpp={mycpp.__file__}")
    from estimater import FoundationPose
    import Utils
    if not hasattr(Utils, "erode_depth"):
        raise RuntimeError("FoundationPose Utils.erode_depth is unavailable")
    print(f"OK FoundationPose={FoundationPose.__module__}; Warp erode_depth present")
    if not (cnos_root / "run_inference.py").is_file():
        raise FileNotFoundError(f"CNOS checkout missing run_inference.py: {cnos_root}")

    # Exercise CUDA allocation and synchronization without loading datasets or weights.
    tensor = torch.ones((8,), device="cuda")
    assert float(tensor.sum().item()) == 8.0
    torch.cuda.synchronize()
    print("OK CUDA allocation/synchronization smoke test")


if __name__ == "__main__":
    main()
