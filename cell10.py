# ==============================================================================
# CELL 10: RENDER VIDEO OVERLAY 6D BOUNDING BOX, HỆ TRỤC & TRẠNG THÁI TRACKING
# ==============================================================================
import cv2
import json
import trimesh
import numpy as np
from pathlib import Path

outdir = WORK_ROOT / 'visualizations'
outdir.mkdir(exist_ok=True, parents=True)
assert 'tracking_records' in globals(), "Thiếu tracking_records! Hãy chạy Cell 7 trước."

# 1. Gom nhóm bản ghi tracking theo (scene_id, im_id)
records_by_frame = {}
for r in tracking_records:
    records_by_frame.setdefault((int(r['scene_id']), int(r['im_id'])), []).append(r)

keys = [(int(sid), int(fid)) for fid in frame_ids]
assert keys, "Không có frame nào để render video."

sid0, fid0 = keys[0]
first_frame = cv2.imread(str(TEST_SOURCE / f'{sid0:06d}' / 'rgb' / f'{fid0:06d}.png'))
assert first_frame is not None, "Không đọc được frame đầu tiên."
h, w = first_frame.shape[:2]

video_path = outdir / 'fastsam_foundationpose_6d_tracking.mp4'
writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
assert writer.isOpened(), f"Không thể khởi tạo VideoWriter tại: {video_path}"

# 12 cạnh của hộp 3D bounding box
edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
box_cache = {}

def corners(oid):
    if oid not in box_cache:
        mesh = trimesh.load(str(CAD_BY_ID[oid]), process=False)
        mesh.vertices *= 1e-3  # Đổi sang mét
        lo, hi = mesh.bounds
        box_cache[oid] = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return box_cache[oid]

def project(points, K):
    q = points @ K.T
    z = q[:, 2]
    uv = np.zeros((len(q), 2), int)
    ok = z > 1e-6
    uv[ok] = np.round(q[ok, :2] / z[ok, None]).astype(int)
    return uv, ok

def draw_pose(frame, pose, oid, K, color):
    T = np.asarray(pose, float)
    pts = corners(oid) @ T[:3, :3].T + T[:3, 3]
    uv, ok = project(pts, K)

    # Vẽ 12 cạnh 3D Bounding Box
    for a, b in edges:
        if ok[a] and ok[b]:
            cv2.line(frame, tuple(uv[a]), tuple(uv[b]), color, 2, cv2.LINE_AA)

    # Vẽ hệ trục tọa độ 3D (X: Đỏ, Y: Xanh lá, Z: Xanh dương)
    origin = T[:3, 3]
    axes = np.vstack([
        origin,
        origin + T[:3, :3] @ np.array([0.05, 0, 0]),
        origin + T[:3, :3] @ np.array([0, 0.05, 0]),
        origin + T[:3, :3] @ np.array([0, 0, 0.05])
    ])
    q, valid = project(axes, K)
    axis_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # BGR
    for j, c in enumerate(axis_colors, 1):
        if valid[0] and valid[j]:
            cv2.arrowedLine(frame, tuple(q[0]), tuple(q[j]), c, 2, tipLength=0.2)

print(f"--- Bắt đầu render video ({len(keys)} frames) ---")
for key in keys:
    s_id, f_id = key
    scene = TEST_SOURCE / f'{s_id:06d}'
    frame = cv2.imread(str(scene / 'rgb' / f'{f_id:06d}.png'))
    if frame is None:
        continue

    cam = json.loads((scene / 'scene_camera.json').read_text())[str(f_id)]
    K = np.asarray(cam['cam_K'], np.float32).reshape(3, 3)
    overlay = frame.copy()

    # Chỉ hiển thị dự đoán; mask xuất hiện trên frame init/recovery.
    for r in records_by_frame.get(key, []):
        oid = int(r['obj_id'])
        state = r['state']
        event = r['event']

        # Màu theo trạng thái: LOST (Đỏ), RE-INIT (Vàng), TRACKING (Xanh lá)
        color = ((0, 0, 255) if state == 'LOST' else
                 (0, 255, 255) if state != 'TRACKING' or event == 'RE_INIT_SUCCESS'
                 else (0, 220, 0))

        # Vẽ 3D Bounding Box & Trục tọa độ
        if r.get('pose') is not None:
            draw_pose(overlay, r['pose'], oid, K, color)

        # Chú thích nhãn trạng thái vật thể
        label = f'ID {oid}: {state}' + (' / RE-INIT' if event == 'RE_INIT_SUCCESS' else '')
        y_pos = 26 + 26 * sorted(TARGET_OBJ_IDS).index(oid)
        cv2.putText(overlay, label, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Hòa trộn overlay vào frame
    frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)
    cv2.putText(frame, f'Scene {s_id:06d} | Frame {f_id:06d} | FoundationPose output',
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    writer.write(frame)

writer.release()
assert video_path.is_file() and video_path.stat().st_size > 0, f"Lỗi: Không tạo được video tại {video_path}"

print("----------------------------------------------------------------------")
print(f"🎉 RENDER HOÀN TẤT!")
print(f"✅ Video kết quả (MP4): {video_path}")
print(f"✅ Kích thước video: {video_path.stat().st_size / (1024 * 1024):.2f} MB")
print("----------------------------------------------------------------------")
