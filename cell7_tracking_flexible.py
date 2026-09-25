# ==============================================================================
# CELL 7: FastSAM/CNOS tại init hoặc lost -> FoundationPose tracking
# ==============================================================================
import os
import sys
import json
import time
import cv2
import trimesh
import numpy as np
import torch
from pathlib import Path

# Các biến sau được thiết lập ở cell chuẩn bị dữ liệu trước đó:
# WORK_ROOT, YCB_ROOT, TEST_SOURCE, CAD_BY_ID, TARGET_OBJ_IDS, sid, frame_ids
required = ('WORK_ROOT', 'YCB_ROOT', 'TEST_SOURCE', 'CAD_BY_ID', 'sid', 'frame_ids',
            'detect_frame')
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(f"Thiếu biến từ các cell trước: {', '.join(missing)}")

FP_ROOT = Path(os.environ.get('FOUNDATIONPOSE_ROOT', '/content/FoundationPose' if Path('/content/FoundationPose').is_dir() else '/kaggle/working/FoundationPose'))
if not (FP_ROOT / '.git').is_dir():
    raise FileNotFoundError(f'Không tìm thấy FoundationPose tại {FP_ROOT}; chạy cell setup trước.')
for p in (FP_ROOT, FP_ROOT / 'mycpp', FP_ROOT / 'mycpp' / 'build'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.chdir(FP_ROOT)

import nvdiffrast.torch as dr
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

for name in ('2024-01-11-20-02-45', '2023-10-28-18-33-37'):
    assert (FP_ROOT / 'weights' / name / 'config.yml').is_file(), f'Thiếu checkpoint FoundationPose: {name}'

POSE_OUT = Path(WORK_ROOT) / 'poses'
POSE_OUT.mkdir(parents=True, exist_ok=True)
print('--- Nạp FoundationPose ---')
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
estimators, meshes, box_cache = {}, {}, {}

# Cấu hình có thể chỉnh theo video / detector.
MIN_MASK_PIXELS = 64
MIN_DETECTION_SCORE = 0.15
DETECT_RETRY_INTERVAL = 3       # Cooldown khi mất object / chưa init được.
MAX_TRANSLATION_STEP_M = 0.20  # Heuristic mất track; tuỳ tốc độ scene.
MIN_DEPTH_SUPPORT = 0.02       # Tỉ lệ pixel depth gần pose trong projected box.
DEPTH_TOLERANCE_M = 0.08
TRACK_ITERATIONS = 2
REGISTER_ITERATIONS = 3

available_ids = {int(k) for k in CAD_BY_ID}
if 'TARGET_OBJ_IDS' in globals() and TARGET_OBJ_IDS is not None:
    allowed_ids = {int(x) for x in TARGET_OBJ_IDS} & available_ids
    skipped_ids = {int(x) for x in TARGET_OBJ_IDS} - available_ids
    if skipped_ids:
        print(f'Cảnh báo: bỏ qua ID không có CAD: {sorted(skipped_ids)}')
else:
    # Nếu không cấu hình TARGET_OBJ_IDS, chỉ dùng ID có CAD trong bộ dữ liệu.
    allowed_ids = available_ids
if not allowed_ids:
    raise RuntimeError('Không có obj_id hợp lệ: kiểm tra TARGET_OBJ_IDS và CAD_BY_ID.')

def get_estimator(oid):
    oid = int(oid)
    if oid not in estimators:
        mesh = trimesh.load(str(CAD_BY_ID[oid]), process=False)
        mesh.vertices *= 1e-3  # CAD YCB-V từ mm sang mét
        np.random.seed(1000 + oid)
        model_pts = mesh.sample(5000)
        meshes[oid] = mesh
        estimators[oid] = FoundationPose(
            model_pts=model_pts,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
        )
    return estimators[oid]

def model_corners(oid):
    oid = int(oid)
    if oid not in box_cache:
        lo, hi = meshes[oid].bounds
        box_cache[oid] = np.array([
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ], dtype=np.float32)
    return box_cache[oid]

def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1.0))

def detection_bbox(mask):
    yy, xx = np.where(mask)
    if not len(xx):
        return None
    return int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)

def projected_bbox(oid, pose, K, shape):
    xyz = model_corners(oid) @ pose[:3, :3].T + pose[:3, 3]
    valid = xyz[:, 2] > 1e-4
    if not valid.any():
        return None
    uvw = xyz[valid] @ K.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    h, w = shape
    box = (
        max(0.0, float(uv[:, 0].min())),
        max(0.0, float(uv[:, 1].min())),
        min(float(w), float(uv[:, 0].max())),
        min(float(h), float(uv[:, 1].max())),
    )
    return box if box[2] > box[0] and box[3] > box[1] else None

def pose_is_valid(oid, pose, K, shape, depth, previous_pose=None, mask=None, score=0.0):
    if pose is None:
        return False, 0.0
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return False, 0.0
    if not (0.05 < pose[2, 3] < 5.0):
        return False, 0.0
    box = projected_bbox(oid, pose, K, shape)
    if box is None:
        return False, 0.0
    if previous_pose is not None:
        motion = np.linalg.norm(pose[:3, 3] - np.asarray(previous_pose)[:3, 3])
        if motion > MAX_TRANSLATION_STEP_M:
            return False, 0.0

    # FoundationPose không trả xác suất lost. Kiểm tra thêm depth tại vị trí CAD.
    xyz = model_corners(oid) @ pose[:3, :3].T + pose[:3, 3]
    x1, y1, x2, y2 = box
    crop = depth[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
    valid_depth = crop[np.isfinite(crop) & (crop > 0)]
    if not valid_depth.size:
        return False, 0.0
    support = np.mean((valid_depth >= xyz[:, 2].min() - DEPTH_TOLERANCE_M) &
                      (valid_depth <= xyz[:, 2].max() + DEPTH_TOLERANCE_M))
    if support < MIN_DEPTH_SUPPORT:
        return False, float(support)
    # Không có detection thì chưa có bằng chứng bbox để so khớp; chỉ kiểm tra pose.
    agreement = bbox_iou(box, detection_bbox(mask)) if mask is not None and score >= MIN_DETECTION_SCORE else 1.0
    return agreement >= 0.01, float(min(agreement, support))

def make_record(oid, fid, shape, score, event, state):
    return {
        'scene_id': int(sid),
        'im_id': int(fid),
        'obj_id': int(oid),
        'state': state,
        'event': event,
        'latency_ms': 0.0,
        'track_ms': 0.0,
        'registration_ms': 0.0,
        'detector_ms': 0.0,
        'detector_called': False,
        'detector_cache_hit': False,
        'recovery_latency_frames': None,
        'pose': None,
        'mask_packbits': None,
        'mask_shape': list(shape),
        'det_score': float(score),
        'pose_gate_score': 0.0,  # heuristic hình học/depth, không phải confidence của FP
        'ever_initialized': bool(states.get(oid, {}).get('initialized', False)),
    }

def usable_detection(detection):
    if detection is None:
        return False
    mask, score = detection
    return int(mask.sum()) >= MIN_MASK_PIXELS and float(score) >= MIN_DETECTION_SCORE

def run_registration(oid, fid, frame_index, rgb, depth, K, shape, mask, score, record, state):
    """Đăng ký lần đầu hoặc bắt lại từ mask dự đoán."""
    started = time.perf_counter()
    try:
        pose = get_estimator(oid).register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask.astype(bool),
            ob_id=int(oid),
            iteration=REGISTER_ITERATIONS,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        ok, agreement = pose_is_valid(oid, pose, K, shape, depth, mask=mask, score=score)
        if ok:
            was_initialized = state['initialized']
            lost_at = state['lost_at']
            state.update(
                initialized=True,
                status='TRACKING',
                lost_at=None,
                pose=np.asarray(pose),
            )
            record.update(
                state='TRACKING',
                event='RE_INIT_SUCCESS' if was_initialized else 'REGISTER_SUCCESS',
                pose=np.asarray(pose).tolist(),
                latency_ms=record['latency_ms'] + elapsed,
                registration_ms=elapsed,
                pose_gate_score=agreement,
                ever_initialized=True,
            )
            if was_initialized and lost_at is not None:
                record['recovery_latency_frames'] = int(frame_index - lost_at)
        else:
            state['status'] = 'LOST'
            state['pose'] = None
            if state['lost_at'] is None:
                state['lost_at'] = frame_index
            record.update(state='LOST', event='REGISTER_FAIL',
                          latency_ms=record['latency_ms'] + elapsed,
                          registration_ms=elapsed,
                          pose_gate_score=agreement, ever_initialized=state['initialized'])
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        state['status'] = 'LOST'
        state['pose'] = None
        if state['lost_at'] is None:
            state['lost_at'] = frame_index
        record.update(
            state='LOST',
            event='REGISTER_FAIL',
            latency_ms=record['latency_ms'] + elapsed,
            registration_ms=elapsed,
            error=str(exc)[:500],
            ever_initialized=state['initialized'],
        )

# State chỉ được tạo khi detector thực sự thấy vật, không dùng GT trong vòng lặp.
states = {}
tracking_records = []
detector_events = []
scene_dir = Path(TEST_SOURCE) / f'{int(sid):06d}'
cam_all = json.loads((scene_dir / 'scene_camera.json').read_text())

frame_ids = list(frame_ids)
if not frame_ids:
    raise ValueError('frame_ids rỗng; không có frame để xử lý.')
print(f'--- Tracking {len(frame_ids)} frames; CAD IDs cho phép: {sorted(allowed_ids)} ---')
t0 = time.time()
last_detection_frame = -DETECT_RETRY_INTERVAL

for frame_index, fid in enumerate(frame_ids):
    frame_root = Path(YCB_ROOT) / 'test' / f'{int(sid):06d}'
    rgb_path = frame_root / 'rgb' / f'{int(fid):06d}.png'
    depth_path = frame_root / 'depth' / f'{int(fid):06d}.png'
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or raw_depth is None:
        raise FileNotFoundError(f'Thiếu RGB/depth frame {fid}: {rgb_path} | {depth_path}')

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    shape = rgb.shape[:2]
    cam = cam_all[str(int(fid))]
    K = np.asarray(cam['cam_K'], dtype=np.float64).reshape(3, 3)
    depth = raw_depth.astype(np.float32) * float(cam['depth_scale']) / 1000.0

    # Tracker đang ổn thì chỉ track_one. Không đọc mask của frame hiện tại.
    records_by_oid = {}
    for oid, state in sorted(states.items()):
        if state['status'] != 'TRACKING':
            continue
        record = make_record(oid, fid, shape, 0.0, 'TRACK', 'TRACKING')
        started = time.perf_counter()
        try:
            pose = get_estimator(oid).track_one(
                rgb=rgb, depth=depth, K=K, iteration=TRACK_ITERATIONS
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            ok, confidence = pose_is_valid(
                oid, pose, K, shape, depth, previous_pose=state['pose']
            )
            if ok:
                state['pose'] = np.asarray(pose)
                record.update(pose=np.asarray(pose).tolist(), latency_ms=elapsed,
                              track_ms=elapsed,
                              pose_gate_score=confidence, ever_initialized=True)
            else:
                state['status'] = 'LOST'
                state['pose'] = None
                state['lost_at'] = frame_index
                record.update(state='LOST', event='TRACK_LOST', latency_ms=elapsed,
                              track_ms=elapsed,
                              pose_gate_score=confidence, ever_initialized=True)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            state['status'] = 'LOST'
            state['pose'] = None
            state['lost_at'] = frame_index
            record.update(state='LOST', event='TRACK_FAIL',
                          latency_ms=elapsed, track_ms=elapsed,
                          error=str(exc)[:500], ever_initialized=True)
        records_by_oid[oid] = record

    needs_detection = (frame_index == 0 or not states or
                       any(s['status'] != 'TRACKING' for s in states.values()))
    if needs_detection and frame_index - last_detection_frame >= DETECT_RETRY_INTERVAL:
        reason = 'INIT' if frame_index == 0 else 'LOST'
        # Provider lỗi là lỗi pipeline, không chuyển thành "không phát hiện".
        detections, detector_ms, cache_hit = detect_frame(fid, shape)
        last_detection_frame = frame_index
        detector_events.append({
            'scene_id': int(sid), 'im_id': int(fid), 'reason': reason,
            'latency_ms': float(detector_ms), 'cache_hit': bool(cache_hit),
            'detected_ids': sorted(int(x) for x in detections),
        })
        for oid, (mask, score) in sorted(detections.items()):
            if oid not in allowed_ids or not usable_detection((mask, score)):
                continue
            if oid not in states:
                states[oid] = {
                    'initialized': False, 'status': 'LOST',
                    'lost_at': frame_index, 'pose': None,
                }
            state = states[oid]
            if state['status'] == 'TRACKING':
                continue
            record = records_by_oid.get(oid)
            if record is None:
                record = make_record(oid, fid, shape, score, 'DETECTED', 'LOST')
            record['det_score'] = float(score)
            record['detector_called'] = True
            record['detector_ms'] = float(detector_ms)
            record['detector_cache_hit'] = bool(cache_hit)
            record['mask_packbits'] = np.packbits(mask.reshape(-1)).tobytes().hex()
            run_registration(oid, fid, frame_index, rgb, depth, K, shape,
                             mask, score, record, state)
            records_by_oid[oid] = record
    for oid, state in sorted(states.items()):
        if oid not in records_by_oid:
            records_by_oid[oid] = make_record(
                oid, fid, shape, 0.0, 'WAITING_FOR_DETECTION', state['status']
            )
    tracking_records.extend(records_by_oid.values())

    if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(frame_ids):
        print(
            f"Frame [{frame_index + 1:3d}/{len(frame_ids)}] | "
            f"objects_seen={len(states)} | records={len(tracking_records)}"
        )

total_time = time.time() - t0
initialized_count = sum(bool(s['initialized']) for s in states.values())
print('----------------------------------------------------------------------')
print(f'Hoàn tất trong {total_time:.1f}s ({total_time / len(frame_ids):.2f}s/frame).')
print(f'Object từng có mask hợp lệ: {len(states)} | đăng ký thành công: {initialized_count}')
print(f'Tổng bản ghi tracking: {len(tracking_records)}')
print(f'Số lần gọi FastSAM+CNOS: {len(detector_events)}')
if not states:
    print('Không có detection hợp lệ trong video; đã bỏ qua frame/object vắng mặt, không tạo LOST giả.')
print('----------------------------------------------------------------------')
