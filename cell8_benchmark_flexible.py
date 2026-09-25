# ==============================================================================
# CELL 8: BENCHMARK 6D POSE, TRACKING, LOSS/RECOVERY VÀ HIỆU NĂNG
# Chỉ chấm pose/tracking của FoundationPose. GT chỉ dùng sau inference.
# ==============================================================================
import json
import numpy as np
import pandas as pd
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree

# CELL 7 có thể tạo danh sách rỗng nếu video không có detection hợp lệ.
if 'tracking_records' not in globals():
    raise RuntimeError('Thiếu tracking_records. Hãy chạy CELL 7 trước.')
for name in ('WORK_ROOT', 'TEST_SOURCE', 'frame_ids', 'total_time'):
    if name not in globals():
        raise RuntimeError(f'Thiếu biến {name}; hãy chạy các cell chuẩn bị dữ liệu trước.')

KEYS = ['scene_id', 'im_id', 'obj_id']
POSE_COLUMNS = [
    *KEYS, 'state', 'event', 'latency_ms', 'track_ms', 'registration_ms',
    'has_gt', 'target_present',
    'pose_valid', 'lost', 'trans_err_cm', 'rot_err_deg', 'add_m', 'adds_m',
    'add_pass_0.1d', 'adds_pass_0.1d', 'ever_initialized'
]
SUMMARY_COLUMNS = [
    'Obj_ID', 'ADD_0.1d(%)', 'ADD-S_0.1d(%)', 'ADD_mean(mm)', 'ADD-S_mean(mm)',
    'Trans_Err(cm)', 'Rot_Err(deg)', 'Loss_Rate(%)', 'Track_Success_Rate(%)',
    'ReDetect_Rate(%)', 'FP_Call_FPS', 'Pipeline_FPS', 'Detector_Calls',
    'Detector_Inferences',
    'Detector_Latency(ms)', 'Reg_Latency(ms)', 'Visible_Frames',
    'Absent_Frames', 'GT_Frames', 'Pose_Frames', 'No_Record_Frames', 'Init_Successes'
]
EVENT_COLUMNS = ['Obj_ID', 'Event', 'Count']

records = list(tracking_records or [])
detector_rows = list(globals().get('detector_events', []))
pipeline_fps = float(len(frame_ids) / total_time) if total_time > 0 else 0.0
fresh_detector_rows = [r for r in detector_rows if not r['cache_hit']]
detector_latency = (float(np.mean([r['latency_ms'] for r in fresh_detector_rows]))
                    if fresh_detector_rows else 0.0)
tracking_df = pd.DataFrame(records)
for col, default in {
    'scene_id': np.nan, 'im_id': np.nan, 'obj_id': np.nan, 'state': 'UNKNOWN',
    'event': 'UNKNOWN', 'latency_ms': 0.0, 'track_ms': 0.0,
    'registration_ms': 0.0, 'pose': None, 'ever_initialized': False,
}.items():
    if col not in tracking_df.columns:
        tracking_df[col] = default

for col in KEYS:
    tracking_df[col] = pd.to_numeric(tracking_df[col], errors='coerce')
tracking_df['latency_ms'] = pd.to_numeric(tracking_df['latency_ms'], errors='coerce').fillna(0.0)
for col in ('track_ms', 'registration_ms'):
    tracking_df[col] = pd.to_numeric(tracking_df[col], errors='coerce').fillna(0.0)
tracking_df = tracking_df.dropna(subset=KEYS).copy()
if not tracking_df.empty:
    tracking_df[KEYS] = tracking_df[KEYS].astype(int)

# Object allowlist là tập CAD được cấu hình trước inference.
allowed_ids = None
if 'TARGET_OBJ_IDS' in globals() and TARGET_OBJ_IDS is not None:
    allowed_ids = {int(x) for x in TARGET_OBJ_IDS}

def row_keys(df):
    if df.empty:
        return set()
    return set(map(tuple, df[KEYS].drop_duplicates().to_numpy()))

record_keys = row_keys(tracking_df)
# Lấy đầy đủ frame trong sequence đang chạy để phân biệt frame vắng vật với LOST.
sequence_frames = set()
if 'sid' in globals() and 'frame_ids' in globals():
    sequence_frames.update((int(sid), int(fid)) for fid in frame_ids)
sequence_frames.update((s, f) for s, f, _ in record_keys)

# Nạp GT cho mọi frame đã được chọn; không dùng GT trong detection/tracking.
gt_cache = {}
gt_available_frames = set()
gt_by_key = {}
for scene_id, im_id in sorted(sequence_frames):
    gt_path = Path(TEST_SOURCE) / f'{scene_id:06d}' / 'scene_gt.json'
    if not gt_path.is_file():
        continue
    if scene_id not in gt_cache:
        gt_cache[scene_id] = json.loads(gt_path.read_text())
    gt_map = gt_cache[scene_id]
    frame_gt = gt_map.get(str(im_id), gt_map.get(im_id, None))
    if frame_gt is None:
        continue
    gt_available_frames.add((scene_id, im_id))
    for instance in frame_gt:
        oid = int(instance['obj_id'])
        if allowed_ids is not None and oid not in allowed_ids:
            continue
        gt_by_key.setdefault((scene_id, im_id, oid), []).append(instance)

if len(gt_available_frames) != len(sequence_frames):
    missing_gt = sorted(sequence_frames - gt_available_frames)[:5]
    raise RuntimeError(f'Thiếu GT cho frame đánh giá: {missing_gt}')
visible_keys = set(gt_by_key)
if allowed_ids is not None:
    visible_keys = {k for k in visible_keys if k[2] in allowed_ids}

# Giữ record cả khi dự đoán nhầm vật không có trong GT để chẩn đoán pose, nhưng
# các record đó không làm tăng LOST/Loss_Rate.
if allowed_ids is not None and not tracking_df.empty:
    tracking_df = tracking_df[tracking_df['obj_id'].isin(allowed_ids)].copy()
    record_keys = row_keys(tracking_df)
record_map = {}
if not tracking_df.empty:
    for row in tracking_df.to_dict('records'):
        record_map[(int(row['scene_id']), int(row['im_id']), int(row['obj_id']))] = row

# Cache điểm CAD để tính ADD và ADD-S trên cùng bộ điểm.
point_cache = {}
model_info = {}
if 'MODEL_SOURCE' in globals():
    info_path = Path(MODEL_SOURCE) / 'models_info.json'
    if info_path.is_file():
        model_info = json.loads(info_path.read_text())

def cad_points(oid, count=1000):
    oid = int(oid)
    if oid not in point_cache:
        if 'CAD_BY_ID' not in globals() or oid not in CAD_BY_ID:
            return None
        mesh = trimesh.load(str(CAD_BY_ID[oid]), process=False)
        mesh.vertices *= 1e-3  # CAD YCB-V lưu theo mm; pose translation dùng mét.
        old_state = np.random.get_state()
        try:
            np.random.seed(1000 + oid)
            point_cache[oid] = mesh.sample(count)
        finally:
            np.random.set_state(old_state)
    return point_cache[oid]

def rotation_error_deg(R_pred, R_gt):
    cosine = (np.trace(R_pred.T @ R_gt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def valid_pose_matrix(value):
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    return pose

# Pose metrics cho record có/không có GT; bổ sung row NO_RECORD cho frame GT-visible
# mà CELL 7 không ghi record. Những frame này là miss thật, không phải object vắng mặt.
all_metric_keys = record_keys | visible_keys
pose_rows = []
for key in sorted(all_metric_keys):
    scene_id, im_id, oid = key
    record = record_map.get(key, {})
    gt_candidates = gt_by_key.get(key, [])
    gt = gt_candidates[0] if gt_candidates else None
    target_present = key in visible_keys
    state = str(record.get('state', 'NO_RECORD'))
    event = str(record.get('event', 'NO_RECORD'))
    pose = valid_pose_matrix(record.get('pose'))
    row = {
        'scene_id': scene_id, 'im_id': im_id, 'obj_id': oid,
        'state': state, 'event': event,
        'latency_ms': float(record.get('latency_ms') or 0.0),
        'track_ms': float(record.get('track_ms') or 0.0),
        'registration_ms': float(record.get('registration_ms') or 0.0),
        'has_gt': gt is not None, 'target_present': target_present,
        'pose_valid': pose is not None,
        'lost': bool(target_present and (state != 'TRACKING' or pose is None)),
        'trans_err_cm': np.nan, 'rot_err_deg': np.nan,
        'add_m': np.nan, 'adds_m': np.nan,
        # End-to-end: frame có GT mà pipeline không xuất pose là thất bại.
        'add_pass_0.1d': 0.0 if gt is not None else np.nan,
        'adds_pass_0.1d': 0.0 if gt is not None else np.nan,
        'ever_initialized': bool(record.get('ever_initialized', False)),
    }
    if gt is not None and pose is not None:
        R_gt = np.asarray(gt['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
        t_gt = np.asarray(gt['cam_t_m2c'], dtype=np.float64) * 1e-3
        row['trans_err_cm'] = float(np.linalg.norm(pose[:3, 3] - t_gt) * 100.0)
        row['rot_err_deg'] = rotation_error_deg(pose[:3, :3], R_gt)
        points = cad_points(oid)
        if points is not None and len(points):
            pred_points = points @ pose[:3, :3].T + pose[:3, 3]
            gt_points = points @ R_gt.T + t_gt
            add_m = float(np.linalg.norm(pred_points - gt_points, axis=1).mean())
            adds_m = float(cKDTree(gt_points).query(pred_points, k=1)[0].mean())
            row['add_m'] = add_m
            row['adds_m'] = adds_m
            info = model_info.get(str(oid), {})
            diameter = float(info.get('diameter', 0.0)) * 1e-3
            if diameter > 0:
                row['add_pass_0.1d'] = float(add_m <= 0.1 * diameter)
                row['adds_pass_0.1d'] = float(adds_m <= 0.1 * diameter)
    pose_rows.append(row)

pose_metrics_df = pd.DataFrame(pose_rows, columns=POSE_COLUMNS)
metric_dir = Path(WORK_ROOT) / 'metrics'
metric_dir.mkdir(parents=True, exist_ok=True)
pose_csv_path = metric_dir / 'pose_metrics_by_frame.csv'
pose_metrics_df.to_csv(pose_csv_path, index=False)

# Summary chỉ có object từng hiện diện trong GT.
present_obj_ids = sorted({key[2] for key in visible_keys})
summary_rows = []
event_rows = []
visible_pose = pose_metrics_df[pose_metrics_df['target_present']] if not pose_metrics_df.empty else pose_metrics_df

for oid in present_obj_ids:
    obj_visible_keys = {key for key in visible_keys if key[2] == oid}
    obj_gt = pose_metrics_df[(pose_metrics_df['obj_id'] == oid) & pose_metrics_df['has_gt']]
    obj_present_pose = pose_metrics_df[(pose_metrics_df['obj_id'] == oid) & pose_metrics_df['target_present']]

    success_mask = obj_present_pose['state'].eq('TRACKING') & obj_present_pose['pose_valid']
    visible_count = len(obj_visible_keys)
    success_count = int(success_mask.sum())
    lost_count = max(0, visible_count - success_count)
    loss_rate = float(lost_count / visible_count * 100.0) if visible_count else np.nan
    track_success_rate = float(success_count / visible_count * 100.0) if visible_count else np.nan

    obj_records = tracking_df[tracking_df['obj_id'] == oid] if not tracking_df.empty else tracking_df
    obj_present_records = obj_records[
        obj_records.apply(lambda r: (int(r['scene_id']), int(r['im_id']), int(r['obj_id'])) in obj_visible_keys, axis=1)
    ] if not obj_records.empty else obj_records
    events = obj_present_records['event'].fillna('UNKNOWN').astype(str) if not obj_present_records.empty else pd.Series(dtype=str)
    previously_initialized = (
        obj_present_records['ever_initialized'].fillna(False).astype(bool)
        if not obj_present_records.empty and 'ever_initialized' in obj_present_records.columns
        else pd.Series(False, index=events.index)
    )
    # Cell 7 ghi REGISTER_FAIL cho cả đăng ký lại; phân biệt bằng cờ
    # ever_initialized được chụp trước lần gọi register().
    reinit_attempts = events.str.startswith('RE_INIT') | (events.eq('REGISTER_FAIL') & previously_initialized)
    reinit_successes = int(events.eq('RE_INIT_SUCCESS').sum())
    redetect_rate = float(reinit_successes / reinit_attempts.sum() * 100.0) if reinit_attempts.any() else 0.0

    timed_calls = obj_present_records[obj_present_records['latency_ms'] > 0] if not obj_present_records.empty else obj_present_records
    total_latency_s = float(timed_calls['latency_ms'].sum() / 1000.0) if not timed_calls.empty else 0.0
    call_count = int((timed_calls['track_ms'] > 0).sum() +
                     (timed_calls['registration_ms'] > 0).sum())
    fps = float(call_count / total_latency_s) if total_latency_s > 0 else 0.0
    registration_calls = timed_calls[timed_calls['registration_ms'] > 0]
    reg_latency = float(registration_calls['registration_ms'].mean()) if not registration_calls.empty else 0.0
    init_successes = int(events.isin(['REGISTER_SUCCESS', 'RE_INIT_SUCCESS']).sum())

    # Ưu tiên đếm frame của scene tương ứng; CELL benchmark thông thường chạy một scene.
    scene_ids_for_obj = {k[0] for k in obj_visible_keys}
    obj_sequence_frames = {(s, f) for s, f in sequence_frames if s in scene_ids_for_obj}
    absent_frames = max(0, len(obj_sequence_frames) - visible_count) if obj_sequence_frames else 0
    no_record_frames = sum(1 for key in obj_visible_keys if key not in record_map)

    summary_rows.append({
        'Obj_ID': str(oid),
        'ADD_0.1d(%)': float(obj_gt['add_pass_0.1d'].mean() * 100.0) if obj_gt['add_pass_0.1d'].notna().any() else np.nan,
        'ADD-S_0.1d(%)': float(obj_gt['adds_pass_0.1d'].mean() * 100.0) if obj_gt['adds_pass_0.1d'].notna().any() else np.nan,
        'ADD_mean(mm)': float(obj_gt['add_m'].mean() * 1000.0) if obj_gt['add_m'].notna().any() else np.nan,
        'ADD-S_mean(mm)': float(obj_gt['adds_m'].mean() * 1000.0) if obj_gt['adds_m'].notna().any() else np.nan,
        'Trans_Err(cm)': float(obj_gt['trans_err_cm'].mean()) if obj_gt['trans_err_cm'].notna().any() else np.nan,
        'Rot_Err(deg)': float(obj_gt['rot_err_deg'].mean()) if obj_gt['rot_err_deg'].notna().any() else np.nan,
        'Loss_Rate(%)': loss_rate,
        'Track_Success_Rate(%)': track_success_rate,
        'ReDetect_Rate(%)': redetect_rate,
        'FP_Call_FPS': fps,
        'Pipeline_FPS': pipeline_fps,
        'Detector_Calls': len(detector_rows),
        'Detector_Inferences': len(fresh_detector_rows),
        'Detector_Latency(ms)': detector_latency,
        'Reg_Latency(ms)': reg_latency,
        'Visible_Frames': int(visible_count),
        'Absent_Frames': int(absent_frames),
        'GT_Frames': int(len(obj_gt)),
        'Pose_Frames': int(obj_gt['pose_valid'].sum()) if not obj_gt.empty else 0,
        'No_Record_Frames': int(no_record_frames),
        'Init_Successes': init_successes,
    })

    # Tổng hợp trạng thái trên frame target thật sự hiện diện. Mất record được tính
    # là NO_RECORD; frame vật vắng mặt không được đưa vào event/Loss_Rate.
    for key in sorted(obj_visible_keys):
        record = record_map.get(key)
        event = str(record.get('event', 'NO_RECORD')) if record else 'NO_RECORD'
        if record and event == 'REGISTER_FAIL' and bool(record.get('ever_initialized', False)):
            event = 'RE_INIT_FAIL'
        event_rows.append({'Obj_ID': str(oid), 'Event': event, 'Count': 1})

summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
if summary_rows:
    numeric_cols = [col for col in SUMMARY_COLUMNS if col != 'Obj_ID']
    macro = {'Obj_ID': 'MACRO_AVG'}
    for col in numeric_cols:
        values = pd.to_numeric(summary_df[col], errors='coerce')
        macro[col] = float(values.mean()) if values.notna().any() else np.nan
    summary_df = pd.concat([summary_df, pd.DataFrame([macro])], ignore_index=True)

events_df = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
if not events_df.empty:
    events_df = events_df.groupby(['Obj_ID', 'Event'], as_index=False)['Count'].sum()

summary_path = metric_dir / 'final_benchmark_summary.csv'
events_path = metric_dir / 'tracking_event_counts.csv'
detector_path = metric_dir / 'detector_events.csv'
summary_df.to_csv(summary_path, index=False)
events_df.to_csv(events_path, index=False)
pd.DataFrame(detector_rows, columns=[
    'scene_id', 'im_id', 'reason', 'latency_ms', 'cache_hit', 'detected_ids'
]).to_csv(detector_path, index=False)

print('--- Kết quả benchmark ---')
if summary_rows:
    print(summary_df.to_string(index=False))
else:
    print('Không có object hiện diện trong frame đánh giá; không tính LOST giả.')
print('--- Event trên các frame GT-visible ---')
print(events_df.to_string(index=False) if not events_df.empty else '(không có event)')
print(f'Pose metrics: {pose_csv_path}')
print(f'Benchmark summary: {summary_path}')
print(f'Tracking events: {events_path}')
print(f'Detector events: {detector_path}')
