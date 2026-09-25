# ==============================================================================
# CELL 3: CHỌN FULL YCB-V SEQUENCE ÍT CHE KHUẤT & CHUẨN BỊ DATASET
# Không cắt 150 frame: dùng toàn bộ frame RGB/depth/camera hợp lệ của một scene.
# ==============================================================================
import os
import json
import shutil
from pathlib import Path

# 1. Đường dẫn YCB-V từ Kaggle Input
YCB_INPUT = Path(os.environ.get('YCB_INPUT_ROOT', '/kaggle/input/datasets/truonglamnhut/ycb-video-v'))
TEST_SOURCE = Path(os.environ.get('YCBV_TEST_ROOT', YCB_INPUT / 'ycbv_test_all' / 'test'))
MODEL_ARCHIVE = YCB_INPUT / 'ycbv_models'
MODEL_SOURCE = Path(os.environ.get('YCBV_MODELS_SOURCE', MODEL_ARCHIVE / 'models'))
BASE_META = YCB_INPUT / 'ycbv_base' / 'ycbv'
TARGETS_PATH = Path(os.environ.get('YCBV_TARGETS_FILE', BASE_META / 'test_targets_bop19.json'))

for source in (TEST_SOURCE, MODEL_SOURCE, TARGETS_PATH):
    if not source.exists():
        raise FileNotFoundError(f'Thiếu đường dẫn dữ liệu YCB-V: {source}')

# Danh sách class cần benchmark. Không yêu cầu tất cả class phải xuất hiện trong scene.
TARGET_OBJ_IDS = {
    int(value) for value in os.environ.get('YCBV_TARGET_OBJ_IDS', ','.join(map(str, range(1, 22)))).split(',')
    if value.strip()
}
PREFERRED_OBJ_IDS = {
    int(value) for value in os.environ.get('YCBV_PREFERRED_OBJ_IDS', '2,3,4,5,6').split(',')
    if value.strip()
} & TARGET_OBJ_IDS
if not PREFERRED_OBJ_IDS:
    PREFERRED_OBJ_IDS = set(TARGET_OBJ_IDS)
MIN_UNOCCLUDED_FRACTION = 0.80
MIN_VISIBLE_PIXELS = 1

WORK_ROOT = Path(os.environ.get('YCBV_WORK_ROOT', '/kaggle/working/cnos_ycbv_workspace'))
DATA_ROOT = WORK_ROOT / 'datasets'
YCB_ROOT = DATA_ROOT / 'ycbv'
WORK_ROOT.mkdir(parents=True, exist_ok=True)
YCB_ROOT.mkdir(parents=True, exist_ok=True)

def replace_link(dst: Path, src: Path):
    """Tạo symlink, gỡ chỉ đích đã được tạo trong workspace này."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    dst.symlink_to(src, target_is_directory=src.is_dir())

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def scene_frame_ids(scene_dir):
    rgb_dir = scene_dir / 'rgb'
    depth_dir = scene_dir / 'depth'
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        return [], {'rgb_only': 0, 'depth_only': 0, 'missing_camera': 0}
    rgb_ids = {int(p.stem) for p in rgb_dir.glob('*.png') if p.stem.isdigit()}
    depth_ids = {int(p.stem) for p in depth_dir.glob('*.png') if p.stem.isdigit()}
    camera_path = scene_dir / 'scene_camera.json'
    if not camera_path.is_file():
        return [], {'rgb_only': len(rgb_ids), 'depth_only': len(depth_ids), 'missing_camera': len(rgb_ids)}
    camera_ids = {int(k) for k in read_json(camera_path)}
    complete_ids = sorted(rgb_ids & depth_ids & camera_ids)
    missing = {
        'rgb_only': len(rgb_ids - depth_ids),
        'depth_only': len(depth_ids - rgb_ids),
        'missing_camera': len((rgb_ids & depth_ids) - camera_ids),
    }
    return complete_ids, missing

# 2. Tìm scene có nhiều frame mục tiêu rõ, ít che khuất.
# scene_gt_info.json (visib_fract/px_count_visib) là nguồn chấm mức che khuất.
targets_path = TARGETS_PATH
if not targets_path.is_file():
    raise FileNotFoundError(f'Thiếu BOP target list: {targets_path}')
all_targets = read_json(targets_path)
target_scene_ids = sorted({
    int(row['scene_id']) for row in all_targets
    if int(row['obj_id']) in PREFERRED_OBJ_IDS
})
if not target_scene_ids:
    raise RuntimeError(f'Không có scene target cho object IDs {sorted(PREFERRED_OBJ_IDS)}')

candidates = []
skipped_scenes = []
for scene_id in target_scene_ids:
    scene_dir = TEST_SOURCE / f'{scene_id:06d}'
    frame_ids, missing = scene_frame_ids(scene_dir)
    if not frame_ids:
        skipped_scenes.append((scene_id, 'không có frame RGB/depth/camera đồng bộ'))
        continue
    if any(missing.values()):
        skipped_scenes.append((scene_id, f'RGB/depth/camera không đầy đủ: {missing}'))
        continue

    gt_path = scene_dir / 'scene_gt.json'
    info_path = scene_dir / 'scene_gt_info.json'
    if not gt_path.is_file():
        skipped_scenes.append((scene_id, 'thiếu scene_gt.json'))
        continue
    gt_map = read_json(gt_path)
    info_map = read_json(info_path) if info_path.is_file() else {}
    visibility_available = info_path.is_file()

    instance_visibility = []
    visible_frames = set()
    clear_frames = set()
    objects_seen = set()
    instances_without_info = 0
    for fid in frame_ids:
        gt_instances = gt_map.get(str(fid), gt_map.get(fid, []))
        frame_info = info_map.get(str(fid), info_map.get(fid, []))
        for index, instance in enumerate(gt_instances):
            oid = int(instance['obj_id'])
            if oid not in PREFERRED_OBJ_IDS:
                continue
            objects_seen.add(oid)
            info = frame_info[index] if index < len(frame_info) else {}
            if 'visib_fract' in info:
                visible_fraction = float(info['visib_fract'])
            else:
                # Thiếu visibility không được xem là vật rõ; xếp scene có annotation
                # visibility đầy đủ lên trước.
                visible_fraction = 0.0
                instances_without_info += 1
            visible_fraction = min(1.0, max(0.0, visible_fraction))
            visible_pixels = int(info.get('px_count_visib', 1 if visible_fraction > 0 else 0))
            instance_visibility.append(visible_fraction)
            if visible_pixels >= MIN_VISIBLE_PIXELS and visible_fraction > 0:
                visible_frames.add(fid)
                if visible_fraction >= MIN_UNOCCLUDED_FRACTION:
                    clear_frames.add(fid)

    if not instance_visibility:
        skipped_scenes.append((scene_id, 'không có object mục tiêu trong annotation'))
        continue

    frame_count = len(frame_ids)
    mean_visibility = float(sum(instance_visibility) / len(instance_visibility))
    unoccluded_frame_ratio = len(clear_frames) / frame_count
    candidate = {
        'scene_id': scene_id,
        'frame_ids': frame_ids,
        'frame_count': frame_count,
        'mean_visibility': mean_visibility,
        'visible_frames': len(visible_frames),
        'unoccluded_frames': len(clear_frames),
        'unoccluded_frame_ratio': unoccluded_frame_ratio,
        'object_ids_seen': sorted(objects_seen),
        'visibility_available': visibility_available,
        'instances_without_info': instances_without_info,
        'missing': missing,
    }
    candidates.append(candidate)

if not candidates:
    raise RuntimeError('Không tìm được scene đầy đủ frame và có object mục tiêu để chạy benchmark.')
if skipped_scenes:
    print(f'Bỏ qua {len(skipped_scenes)} scene không đủ file/frame/annotation.')

# Ưu tiên số frame có vật rõ nhất, sau đó tỷ lệ/độ rõ trung bình và độ dài video.
# scene_id nhỏ hơn dùng làm tie-break ổn định.
candidates.sort(
    key=lambda c: (
        c['visibility_available'],
        c['unoccluded_frames'],
        c['unoccluded_frame_ratio'],
        c['mean_visibility'],
        c['visible_frames'],
        c['frame_count'],
        -c['scene_id'],
    ),
    reverse=True,
)
selected = candidates[0]
sid = int(selected['scene_id'])
frame_ids = list(selected['frame_ids'])  # Tất cả frame; không dùng slicing/truncation.
# Toàn bộ target BOP của scene trong full video, không giới hạn số frame.
filtered_targets = [
    row for row in all_targets
    if int(row['scene_id']) == sid
    and int(row['obj_id']) in TARGET_OBJ_IDS
    and int(row['im_id']) in set(frame_ids)
]
EVAL_TARGETS = filtered_targets
EVAL_TARGETS_PATH = WORK_ROOT / 'eval_targets.json'
EVAL_TARGETS_PATH.write_text(json.dumps(EVAL_TARGETS), encoding='utf-8')

score_rows = [
    {k: v for k, v in c.items() if k not in ('frame_ids', 'missing')} | c['missing']
    for c in candidates
]
scene_scores_path = WORK_ROOT / 'scene_visibility_scores.json'
scene_scores_path.write_text(json.dumps(score_rows, indent=2), encoding='utf-8')

# 3. Chuẩn bị một scene đầy đủ tại workspace; liên kết toàn bộ frame/annotation.
test_dir = YCB_ROOT / 'test'
if test_dir.is_symlink() or test_dir.is_file():
    test_dir.unlink()
elif test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(parents=True, exist_ok=True)

scene_name = f'{sid:06d}'
src_scene = TEST_SOURCE / scene_name
dst_scene = test_dir / scene_name
dst_scene.mkdir()

camera_all = read_json(src_scene / 'scene_camera.json')
camera_full = {
    str(fid): camera_all[str(fid)] if str(fid) in camera_all else camera_all[fid]
    for fid in frame_ids
}
(dst_scene / 'scene_camera.json').write_text(json.dumps(camera_full), encoding='utf-8')
for annotation_name in ('scene_gt.json', 'scene_gt_info.json'):
    src = src_scene / annotation_name
    if src.is_file():
        replace_link(dst_scene / annotation_name, src)

for subdir in ('rgb', 'depth'):
    source_dir = src_scene / subdir
    output_dir = dst_scene / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    for fid in frame_ids:
        src = source_dir / f'{fid:06d}.png'
        if not src.is_file():
            raise FileNotFoundError(f'Thiếu {subdir} frame {fid}: {src}')
        replace_link(output_dir / src.name, src)

# 4. Chuẩn bị CAD cho các class cần đánh giá.
models_dir = YCB_ROOT / 'models'
models_dir.mkdir(exist_ok=True)
models_subset = models_dir / 'models'
if models_subset.is_symlink():
    models_subset.unlink()
elif models_subset.exists():
    shutil.rmtree(models_subset)
models_subset.mkdir()

CAD_BY_ID = {}
for oid in sorted(TARGET_OBJ_IDS):
    src = MODEL_SOURCE / f'obj_{oid:06d}.ply'
    if not src.is_file():
        raise FileNotFoundError(f'Thiếu CAD model: {src}')
    dst = models_subset / src.name
    shutil.copy2(src, dst)

    # Bỏ comment texturefile lỗi thời, chỉ sửa header và giữ nguyên binary payload.
    ply_data = dst.read_bytes()
    header_end = ply_data.find(b'end_header')
    if header_end < 0:
        raise ValueError(f'Header PLY không hợp lệ: {dst}')
    header = ply_data[:header_end]
    payload = ply_data[header_end:]
    header_lines = [
        line for line in header.splitlines()
        if not line.lower().startswith(b'comment texturefile')
    ]
    dst.write_bytes(b'\n'.join(header_lines) + b'\n' + payload)
    CAD_BY_ID[oid] = dst

full_info = read_json(MODEL_SOURCE / 'models_info.json')
(models_dir / 'models_info.json').write_text(
    json.dumps({str(oid): full_info[str(oid)] for oid in sorted(TARGET_OBJ_IDS)}),
    encoding='utf-8',
)

# BOP environment
os.environ['BOP_DIR'] = str(DATA_ROOT)
os.environ['YCB_VIDEO_DIR'] = str(YCB_ROOT)

# 5. Xác nhận video được giữ trọn vẹn.
assert len(frame_ids) == selected['frame_count']
assert len(list((dst_scene / 'rgb').glob('*.png'))) == len(frame_ids)
assert len(list((dst_scene / 'depth').glob('*.png'))) == len(frame_ids)
assert all(Path(CAD_BY_ID[oid]).is_file() for oid in TARGET_OBJ_IDS)

print('-' * 78)
print(f'Selected scene: {sid:06d}')
print(f'Full sequence frames: {len(frame_ids)} ({frame_ids[0]}..{frame_ids[-1]})')
print(f'Frame range contains gaps: {any(b != a + 1 for a, b in zip(frame_ids, frame_ids[1:]))}')
print(f'Target CAD IDs configured: {sorted(TARGET_OBJ_IDS)}')
print(f"Target IDs present in selected scene: {selected['object_ids_seen']}")
print(f"Frames with target visible: {selected['visible_frames']}/{len(frame_ids)}")
print(f"Frames with target visibility >= {MIN_UNOCCLUDED_FRACTION:.0%}: {selected['unoccluded_frames']}")
print(f"Mean target visibility: {selected['mean_visibility']:.3f}")
print(f"Visibility annotations available: {selected['visibility_available']}")
print(f'All scene scores: {scene_scores_path}')
print(f'Full video dataset: {dst_scene}')
print(f'Eval targets: {EVAL_TARGETS_PATH} ({len(EVAL_TARGETS)} rows)')
print('-' * 78)
