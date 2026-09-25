#!/usr/bin/env python3
"""Run the existing YCB-V cells in one Colab process after cached setup."""
from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.environ.setdefault("COLAB_WORK_ROOT", "/content"))
FP_ROOT = Path(os.environ.setdefault("FOUNDATIONPOSE_ROOT", str(WORK_ROOT / "FoundationPose")))
CNOS_ROOT = Path(os.environ.setdefault("CNOS_ROOT", str(WORK_ROOT / "cnos")))
YCB_INPUT = Path(os.environ.setdefault("YCB_INPUT_ROOT", "/content/data"))
ASSET_ROOT = Path(os.environ.setdefault("FOUNDATIONPOSE_ASSETS_ROOT", "/content/drive/MyDrive/foundationpose-assets"))
PIPELINE_ROOT = Path(os.environ.setdefault("YCBV_WORK_ROOT", str(WORK_ROOT / "cnos_ycbv_workspace")))

os.environ["COLAB_CACHED_ENV"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def fail_if_missing(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} không tồn tại: {path}")


def preflight() -> None:
    import torch

    print("[PREFLIGHT]")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}")
    if not torch.cuda.is_available():
        raise RuntimeError("Colab chưa có GPU. Chọn Runtime > Change runtime type > GPU rồi restart.")
    if not (FP_ROOT / ".git").is_dir():
        raise FileNotFoundError(f"FoundationPose checkout chưa có tại {FP_ROOT}; chạy installer trước.")
    if not (CNOS_ROOT / "run_inference.py").is_file():
        raise FileNotFoundError(f"CNOS checkout chưa có tại {CNOS_ROOT}; chạy installer trước.")

    required_inputs = (
        (YCB_INPUT / "ycbv_test_all" / "test", "BOP test RGB-D scenes"),
        (YCB_INPUT / "ycbv_models" / "models", "YCB-V CAD models"),
        (YCB_INPUT / "ycbv_base" / "ycbv" / "test_targets_bop19.json", "BOP target annotations"),
    )
    for path, description in required_inputs:
        fail_if_missing(path, description)

    target_ids = {
        int(value) for value in os.environ.get("YCBV_TARGET_OBJ_IDS", ",".join(map(str, range(1, 22)))).split(",")
        if value.strip()
    }
    if not target_ids:
        raise ValueError("YCBV_TARGET_OBJ_IDS is empty")
    model_root = YCB_INPUT / "ycbv_models" / "models"
    missing_cads = [oid for oid in sorted(target_ids) if not (model_root / f"obj_{oid:06d}.ply").is_file()]
    if missing_cads:
        raise FileNotFoundError(
            f"Thiếu CAD cho object IDs {missing_cads}. Đặt YCBV_TARGET_OBJ_IDS theo CAD có trong dataset."
        )

    weights = ("2024-01-11-20-02-45", "2023-10-28-18-33-37")
    missing_weights = []
    for name in weights:
        if (FP_ROOT / "weights" / name / "config.yml").is_file() and (
            FP_ROOT / "weights" / name / "model_best.pth"
        ).is_file():
            continue
        if not any(
            (candidate / "config.yml").is_file() and (candidate / "model_best.pth").is_file()
            for candidate in ASSET_ROOT.glob(f"**/{name}")
            if candidate.is_dir()
        ):
            missing_weights.append(name)
    if missing_weights:
        raise FileNotFoundError(
            f"Thiếu FoundationPose checkpoints {missing_weights}. Tải scorer/refiner từ link chính thức "
            "được nêu trong NVlabs/FoundationPose README, đặt vào FOUNDATIONPOSE_ASSETS_ROOT."
        )
    print(f"YCB input: {YCB_INPUT}")
    print(f"FoundationPose weights source: {ASSET_ROOT}")
    print(f"Target object IDs: {sorted(target_ids)}")
    print("BOP RGB-D, camera/GT annotations, CAD files, and checkpoints found.")


def main() -> None:
    preflight()
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    cells = (
        "cell2_foundationpose_init.py",
        "cell3_ycbv_full_video.py",
        "cell4.py",
        "cell5_cnos_full_video.py",
        "cell7_tracking_flexible.py",
        "cell8_benchmark_flexible.py",
        "cell9_visualize_full_video.py",
        "cell10.py",
    )
    shared = globals()
    for index, filename in enumerate(cells, 1):
        path = REPO_ROOT / filename
        print(f"\n[{index}/{len(cells)}] {filename}", flush=True)
        source = path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), shared, shared)

    print("\n[OUTPUTS]")
    for relative in (
        "visualizations/fastsam_foundationpose_6d_tracking.mp4",
        "metrics/final_benchmark_summary.csv",
        "metrics/pose_metrics_by_frame.csv",
        "metrics/tracking_event_counts.csv",
        "metrics/detector_events.csv",
        "metrics/ycbv_full_scene_benchmark_charts.png",
        "metrics/ycbv_full_scene_init_vs_tracking_latency.png",
    ):
        path = PIPELINE_ROOT / relative
        print(f"{path}: {'OK' if path.is_file() else 'MISSING'}")

    video_path = PIPELINE_ROOT / "visualizations" / "fastsam_foundationpose_6d_tracking.mp4"
    summary_path = PIPELINE_ROOT / "metrics" / "final_benchmark_summary.csv"
    import pandas as pd

    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError("Video output is missing or empty")
    if not summary_path.is_file() or pd.read_csv(summary_path).empty:
        raise RuntimeError("Benchmark summary has no rows; no scored BOP targets were produced")
    valid_poses = [
        row for row in shared.get("tracking_records", [])
        if row.get("state") == "TRACKING" and row.get("pose") is not None
    ]
    if not valid_poses:
        raise RuntimeError("No valid FoundationPose tracking pose was recorded; do not treat the video as a successful benchmark")
    print(f"Successful pose records: {len(valid_poses)}")


if __name__ == "__main__":
    main()
