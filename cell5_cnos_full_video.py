# CELL 5: FastSAM + CNOS, chỉ chạy khi CELL 7 yêu cầu init/recovery.
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CNOS_ROOT = Path(os.environ.get('CNOS_ROOT', '/content/cnos' if Path('/content/cnos').is_dir() else '/kaggle/working/cnos'))
WORK_ROOT = Path(WORK_ROOT)
DETECTION_DIR = WORK_ROOT / 'predictions' / 'cnos_fast_events'
RESULT_FILE = (WORK_ROOT / 'results' / 'cnos_exps' /
               'FastSAM_template_pyrender0_aggavg_5_ycbv_with_score_distribution.json')
TARGET_FILE = Path(str(WORK_ROOT / 'datasets') + 'ycbv') / 'test_targets_bop19.json'
CNOS_CATEGORY_MODE = 'target_compact_1based'
FORCE_CNOS_INFERENCE = False

target_obj_ids = sorted({int(oid) for oid in TARGET_OBJ_IDS})
assert target_obj_ids and set(target_obj_ids) <= set(map(int, CAD_BY_ID))
assert (CNOS_ROOT / 'run_inference.py').is_file(), 'Chạy CELL 4 trước.'
assert TARGET_FILE.parent.is_dir(), 'Thiếu BOP query alias từ CELL 4.'
assert CNOS_CATEGORY_MODE in {'target_compact_1based', 'bop_obj_id'}
DETECTION_DIR.mkdir(parents=True, exist_ok=True)


def _decode_rle(segmentation, shape):
    if not isinstance(segmentation, dict) or 'counts' not in segmentation:
        raise ValueError('CNOS segmentation phải là COCO RLE.')
    from pycocotools import mask as mask_utils
    rle = dict(segmentation)
    if isinstance(rle['counts'], list):
        rle = mask_utils.frPyObjects(rle, *shape)
    elif isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('ascii')
    mask = np.asarray(mask_utils.decode(rle))
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.shape != tuple(shape):
        raise ValueError(f'CNOS mask shape {mask.shape}, expected {shape}')
    return mask.astype(np.uint8)


def _read_npz(path, shape):
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data['category_id'], dtype=np.int32)
        scores = np.asarray(data['score'], dtype=np.float32)
        masks = np.asarray(data['segmentation'], dtype=np.uint8)
    if masks.shape != (len(ids), *shape) or len(scores) != len(ids):
        raise ValueError(f'CNOS cache không hợp lệ: {path}')
    best = {}
    for oid, score, mask in zip(ids, scores, masks):
        oid = int(oid)
        if oid in target_obj_ids and (oid not in best or score > best[oid][1]):
            best[oid] = (mask.astype(bool), float(score))
    return best


def detect_frame(fid, shape):
    """Return (best masks by BOP ID, detector milliseconds, cache hit)."""
    fid, sid_int = int(fid), int(sid)
    rgb_path = Path(YCB_ROOT) / 'test' / f'{sid_int:06d}' / 'rgb' / f'{fid:06d}.png'
    if not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)
    cache_path = DETECTION_DIR / f'scene{sid_int:06d}_frame{fid:06d}.npz'
    manifest_path = cache_path.with_suffix('.json')
    image_stat = rgb_path.stat()
    signature = {
        'scene_id': sid_int, 'frame_id': fid, 'target_obj_ids': target_obj_ids,
        'category_mode': CNOS_CATEGORY_MODE, 'rgb_size': image_stat.st_size,
        'rgb_mtime_ns': image_stat.st_mtime_ns, 'shape': list(shape),
    }
    if not FORCE_CNOS_INFERENCE and cache_path.is_file() and manifest_path.is_file():
        if json.loads(manifest_path.read_text()) == signature:
            return _read_npz(cache_path, shape), 0.0, True

    # BaseBOPTest đọc target file. Một hàng/frame tránh inference trùng ảnh.
    # obj_id ở đây chỉ là trường bắt buộc, không chọn CAD và không lấy từ GT.
    TARGET_FILE.write_text(json.dumps([{
        'scene_id': sid_int, 'im_id': fid,
        'obj_id': target_obj_ids[0], 'inst_count': 1,
    }]), encoding='utf-8')
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.unlink(missing_ok=True)
    command = [
        sys.executable, 'run_inference.py', 'dataset_name=ycbv',
        'model=cnos_fast', 'model.onboarding_config.rendering_type=pyrender',
        f'machine.root_dir={WORK_ROOT}',
        f'data.query_dataloader.root_dir={WORK_ROOT}/datasets',
        'machine.num_workers=0',
    ]
    env = os.environ.copy()
    env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    env['PYOPENGL_PLATFORM'] = 'egl'
    started = time.perf_counter()
    subprocess.run(command, cwd=CNOS_ROOT, env=env, check=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not RESULT_FILE.is_file():
        raise FileNotFoundError(f'CNOS không ghi output: {RESULT_FILE}')
    rows = json.loads(RESULT_FILE.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError('CNOS output không phải JSON list.')
    foreign = [(r.get('scene_id'), r.get('image_id')) for r in rows
               if int(r['scene_id']) != sid_int or int(r['image_id']) != fid]
    if foreign:
        raise RuntimeError(f'CNOS đã xử lý frame ngoài yêu cầu: {foreign[:3]}')
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != tuple(shape):
        raise ValueError(f'RGB shape không khớp tại frame {fid}')
    ids, scores, masks = [], [], []
    for row in rows:
        category = int(row['category_id'])
        if CNOS_CATEGORY_MODE == 'target_compact_1based':
            if not 1 <= category <= len(target_obj_ids):
                raise ValueError(f'CNOS category_id {category} ngoài {target_obj_ids}')
            oid = target_obj_ids[category - 1]
        else:
            oid = category
        if oid not in target_obj_ids:
            continue
        score = float(row['score'])
        if not np.isfinite(score):
            raise ValueError('CNOS score không hữu hạn.')
        ids.append(oid)
        scores.append(score)
        masks.append(_decode_rle(row['segmentation'], shape))
    masks_array = np.stack(masks) if masks else np.zeros((0, *shape), dtype=np.uint8)
    np.savez_compressed(cache_path, category_id=np.asarray(ids, np.int32),
                        score=np.asarray(scores, np.float32), segmentation=masks_array)
    manifest_path.write_text(json.dumps(signature), encoding='utf-8')
    return _read_npz(cache_path, shape), elapsed_ms, False


print('CNOS event provider sẵn sàng; CELL 7 sẽ gọi detect_frame().')
