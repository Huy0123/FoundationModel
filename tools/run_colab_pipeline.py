#!/usr/bin/env python3
"""Run the existing YCB-V cells in one Colab process after cached setup."""
from __future__ import annotations

import os
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.environ.setdefault("COLAB_WORK_ROOT", "/content"))
FP_ROOT = Path(os.environ.setdefault("FOUNDATIONPOSE_ROOT", str(WORK_ROOT / "FoundationPose")))
CNOS_ROOT = Path(os.environ.setdefault("CNOS_ROOT", str(WORK_ROOT / "cnos")))
YCB_INPUT = Path(os.environ.get("YCB_INPUT_ROOT", "/content/data"))
ASSET_ROOT = Path(os.environ.setdefault("FOUNDATIONPOSE_ASSETS_ROOT", "/content/drive/MyDrive/foundationpose-assets"))
PIPELINE_ROOT = Path(os.environ.setdefault("YCBV_WORK_ROOT", str(WORK_ROOT / "cnos_ycbv_workspace")))

os.environ["COLAB_CACHED_ENV"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def fail_if_missing(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} không tồn tại: {path}")


def configure_dataset() -> tuple[Path, Path, Path, set[int]]:
    roots = [YCB_INPUT]
    if (YCB_INPUT / "ycbv_5cad_test").is_dir():
        roots.insert(0, YCB_INPUT / "ycbv_5cad_test")
    elif YCB_INPUT.name == "ycbv_5cad_test" and YCB_INPUT.parent.is_dir():
        roots.insert(0, YCB_INPUT)

    dataset_root = next((root for root in roots if (root / "test").is_dir()), None)
    if dataset_root is None:
        dataset_root = next((root for root in roots if (root / "ycbv_test_all" / "test").is_dir()), None)
    if dataset_root is None:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục test RGB-D trong {YCB_INPUT}; cần `test/` hoặc `ycbv_test_all/test/`."
        )

    test_root = dataset_root / "test" if (dataset_root / "test").is_dir() else dataset_root / "ycbv_test_all" / "test"
    candidates = (
        dataset_root / "metadata" / "test_targets_bop19.json",
        dataset_root / "ycbv_base" / "ycbv" / "test_targets_bop19.json",
    )
    targets_file = Path(os.environ["YCBV_TARGETS_FILE"]) if os.environ.get("YCBV_TARGETS_FILE") else next(
        (path for path in candidates if path.is_file()), candidates[0]
    )
    raw_models = dataset_root / "models"
    if not raw_models.is_dir():
        raw_models = dataset_root / "ycbv_models" / "models"

    available: set[int] = set()
    for model_dir in raw_models.iterdir() if raw_models.is_dir() else ():
        match = re.match(r"^(\d{3})_", model_dir.name)
        if match and (model_dir / "textured_simple.obj").is_file():
            # BOP YCB-V object ID equals the three-digit YCB object number minus one.
            available.add(int(match.group(1)) - 1)
    if not available:
        for model_path in raw_models.glob("obj_*.ply") if raw_models.is_dir() else ():
            match = re.fullmatch(r"obj_(\d{6})\.ply", model_path.name)
            if match:
                available.add(int(match.group(1)))

    if os.environ.get("YCBV_TARGET_OBJ_IDS"):
        target_ids = {int(value) for value in os.environ["YCBV_TARGET_OBJ_IDS"].split(",") if value.strip()}
    else:
        target_ids = available
        os.environ["YCBV_TARGET_OBJ_IDS"] = ",".join(map(str, sorted(target_ids)))
    if not target_ids:
        raise ValueError(f"Không suy ra được object IDs từ CAD trong {raw_models}")
    missing_ids = sorted(target_ids - available)
    if missing_ids:
        raise FileNotFoundError(f"Không có CAD source cho BOP object IDs {missing_ids} trong {raw_models}")
    os.environ.setdefault("YCBV_PREFERRED_OBJ_IDS", ",".join(map(str, sorted(target_ids))))

    models_source = raw_models
    if not all((raw_models / f"obj_{oid:06d}.ply").is_file() for oid in target_ids):
        models_source = PIPELINE_ROOT / "converted_ycbv_models"
    os.environ["YCB_INPUT_ROOT"] = str(dataset_root)
    os.environ["YCBV_TEST_ROOT"] = str(test_root)
    os.environ["YCBV_TARGETS_FILE"] = str(targets_file)
    os.environ["YCBV_MODELS_SOURCE"] = str(models_source)
    return dataset_root, test_root, targets_file, target_ids


def convert_obj_models(raw_root: Path, output_root: Path, target_ids: set[int]) -> None:
    """Convert the supplied meter-scale OBJ meshes to BOP PLY files in millimeters."""
    import numpy as np
    import trimesh
    from scipy.spatial.distance import pdist

    output_root.mkdir(parents=True, exist_ok=True)
    models_info = {}
    for model_dir in sorted(raw_root.iterdir()):
        match = re.match(r"^(\d{3})_", model_dir.name)
        obj_path = model_dir / "textured_simple.obj"
        if not match or not obj_path.is_file():
            continue
        oid = int(match.group(1)) - 1
        if oid not in target_ids:
            continue
        mesh = trimesh.load(str(obj_path), force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
            raise ValueError(f"Mesh OBJ rỗng hoặc không hợp lệ: {obj_path}")
        extents_m = np.asarray(mesh.extents, dtype=float)
        if not np.isfinite(extents_m).all() or not (0.005 < float(extents_m.max()) < 1.0):
            raise ValueError(f"Kích thước OBJ không giống đơn vị mét, không tự scale: {obj_path}, {extents_m}")
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * 1000.0
        destination = output_root / f"obj_{oid:06d}.ply"
        mesh.export(destination, file_type="ply")
        bounds = np.asarray(mesh.bounds, dtype=float)
        extents = bounds[1] - bounds[0]
        hull_vertices = mesh.convex_hull.vertices
        diameter = float(pdist(hull_vertices).max()) if len(hull_vertices) > 1 else float(np.linalg.norm(extents))
        models_info[str(oid)] = {
            "diameter": diameter,
            "min_x": float(bounds[0, 0]), "min_y": float(bounds[0, 1]), "min_z": float(bounds[0, 2]),
            "size_x": float(extents[0]), "size_y": float(extents[1]), "size_z": float(extents[2]),
        }
        print(f"Converted {model_dir.name} -> {destination.name} (BOP id={oid}, diameter={diameter:.1f} mm)")
    if set(models_info) != {str(oid) for oid in target_ids}:
        raise RuntimeError(f"CAD conversion incomplete: expected {sorted(target_ids)}, got {sorted(models_info)}")
    (output_root / "models_info.json").write_text(json.dumps(models_info, indent=2), encoding="utf-8")


def preflight() -> None:
    import torch

    dataset_root, test_root, targets_file, target_ids = configure_dataset()
    models_source = Path(os.environ["YCBV_MODELS_SOURCE"])

    print("[PREFLIGHT]")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}")
    if not torch.cuda.is_available():
        raise RuntimeError("Colab chưa có GPU. Chọn Runtime > Change runtime type > GPU rồi restart.")
    if not (FP_ROOT / ".git").is_dir():
        raise FileNotFoundError(f"FoundationPose checkout chưa có tại {FP_ROOT}; chạy installer trước.")
    if not (CNOS_ROOT / "run_inference.py").is_file():
        raise FileNotFoundError(f"CNOS checkout chưa có tại {CNOS_ROOT}; chạy installer trước.")

    required_inputs = ((test_root, "BOP test RGB-D scenes"), (models_source, "YCB-V BOP CAD models"),
                       (targets_file, "BOP target annotations"))
    for path, description in required_inputs:
        fail_if_missing(path, description)

    if models_source == PIPELINE_ROOT / "converted_ycbv_models":
        complete_cache = (models_source / "models_info.json").is_file() and all(
            (models_source / f"obj_{oid:06d}.ply").is_file() for oid in target_ids
        )
        if not complete_cache:
            convert_obj_models(dataset_root / "models", models_source, target_ids)

    if not target_ids:
        raise ValueError("YCBV_TARGET_OBJ_IDS is empty")
    missing_cads = [oid for oid in sorted(target_ids) if not (models_source / f"obj_{oid:06d}.ply").is_file()]
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
    print(f"YCB input: {dataset_root}")
    print(f"RGB-D scenes: {test_root}")
    print(f"BOP models: {models_source}")
    print(f"BOP target file: {targets_file}")
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
