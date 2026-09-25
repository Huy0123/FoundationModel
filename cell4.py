# ==============================================================================
# CELL 4: THIẾT LẬP CNOS & RENDER 3D TEMPLATES (PYRENDER)
# (Tối ưu từ Cell 13 và hấp thụ Cell 16 cũ)
# ==============================================================================
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# Đảm bảo các cờ môi trường bắt buộc cho EGL & PyTorch
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

CNOS_ROOT = Path(os.environ.get('CNOS_ROOT', '/content/cnos' if Path('/content/cnos').is_dir() else '/kaggle/working/cnos'))
WORK_ROOT = Path(os.environ.get('YCBV_WORK_ROOT', '/kaggle/working/cnos_ycbv_workspace'))
DATASETS_DIR = WORK_ROOT / 'datasets'

# 1. Clone repo CNOS
if not (CNOS_ROOT / 'run_inference.py').is_file():
    if CNOS_ROOT.exists():
        shutil.rmtree(CNOS_ROOT)
    print("--- 1. Đang clone CNOS ---")
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/nv-nguyen/cnos.git', str(CNOS_ROOT)], check=True)

# 2. Cài đặt các thư viện cần thiết cho CNOS & PyRender
print("--- 2. Cài đặt thư viện bổ trợ cho CNOS & PyRender ---")
constraint_file = Path(os.environ.get('COLAB_WORK_ROOT', '/content')) / 'cnos_ycbv_constraints.txt'
constraint_file.write_text('numpy==1.26.4\n')  # <-- THÊM DÒNG NÀY ĐỂ TẠO FILE
os.environ['PIP_CONSTRAINT'] = str(constraint_file)

if os.environ.get('COLAB_CACHED_ENV') != '1':
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'hydra-core==1.3.2', 'hydra-colorlog', 'pyrender', 'gdown',
                    'ruamel.yaml', 'distinctipy', 'fvcore', 'iopath', 'onnx',
                    'onnxruntime', 'pycocotools', 'ultralytics==8.0.135'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/thodan/bop_toolkit.git'], check=True)
    # Cài đặt PyOpenGL tương thích EGL (loại bỏ PyOpenGL-accelerate dễ gây lỗi C-extension)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', 'PyOpenGL==3.1.10', 'setuptools==80.9.0'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'PyOpenGL-accelerate'], check=False)
else:
    print('Dùng lại dependencies từ tools/install_colab_cached.sh; bỏ qua cài package lần hai.')

# 3. Tạo cấu trúc thư mục alias cho CNOS
print("--- 3. Thiết lập liên kết dữ liệu cho CNOS ---")
models_dir = DATASETS_DIR / 'ycbv/models/models'
models_dir.mkdir(parents=True, exist_ok=True)
src_models = Path(os.environ.get(
    'YCBV_MODELS_SOURCE',
    Path(os.environ.get('YCB_INPUT_ROOT', '/kaggle/input/datasets/truonglamnhut/ycb-video-v')) / 'ycbv_models' / 'models'
))

for oid in sorted(TARGET_OBJ_IDS):
    src = src_models / f'obj_{oid:06d}.ply'
    dst = models_dir / src.name
    assert src.is_file(), f"Thiếu CAD: {src}"
    if not dst.exists():
        dst.symlink_to(src)

info_src = src_models / 'models_info.json'
info_dst = models_dir.parent / 'models_info.json'
full_info = json.loads(info_src.read_text())
if info_dst.is_symlink() or info_dst.exists():
    info_dst.unlink()
info_dst.write_text(json.dumps({str(int(oid)): full_info[str(int(oid))] for oid in TARGET_OBJ_IDS}))

# CNOS tự nối 'ycbv' vào root query nên cần alias 'datasetsycbv'
dataset_alias = Path(str(DATASETS_DIR) + 'ycbv')
dataset_alias.mkdir(parents=True, exist_ok=True)
for name in ('test', 'models'):
    dst = dataset_alias / name
    src = DATASETS_DIR / 'ycbv' / name
    if dst.is_symlink() or dst.is_file(): dst.unlink()
    elif dst.exists(): shutil.rmtree(dst)
    dst.symlink_to(src, target_is_directory=True)

# BaseBOPTest dùng mỗi hàng để chọn ảnh. Một hàng/ảnh tránh lặp inference.
# CELL 5 sẽ thu hẹp file này về đúng frame khi phát hiện.
query_targets = [{'scene_id': int(sid), 'im_id': int(fid),
                  'obj_id': min(TARGET_OBJ_IDS), 'inst_count': 1}
                 for fid in frame_ids]
(dataset_alias / 'test_targets_bop19.json').write_text(json.dumps(query_targets))

# Dataloader của CNOS đọc trực tiếp từ workspace/ycbv/test
ycbv_alias = WORK_ROOT / 'ycbv'
if ycbv_alias.is_symlink() or ycbv_alias.is_file(): ycbv_alias.unlink()
elif ycbv_alias.exists(): shutil.rmtree(ycbv_alias)
ycbv_alias.symlink_to(DATASETS_DIR / 'ycbv', target_is_directory=True)

# 4. Patch renderer pyrender.py của CNOS (tránh lỗi kích thước bounding sphere)
renderer_file = CNOS_ROOT / 'src/poses/pyrender.py'
rtext = renderer_file.read_text()
if 'diameter = float(np.linalg.norm(mesh.extents * 2))' not in rtext:
    rtext = rtext.replace('diameter = get_obj_diameter(mesh)',
                          'diameter = float(np.linalg.norm(mesh.extents * 2))')
    renderer_file.write_text(rtext)
    print("✅ Đã patch pyrender.py thành công.")

# 5. Render 642 templates cho từng vật thể mục tiêu
print("--- 4. Kiểm tra và Render Templates 3D ---")
template_root = DATASETS_DIR / 'templates_pyrender/ycbv'
counts = {oid: len([p for p in (template_root / f'obj_{oid:06d}').rglob('*') if p.is_file()])
          for oid in sorted(TARGET_OBJ_IDS)}

if not all(counts.get(oid, 0) >= 642 for oid in TARGET_OBJ_IDS):
    print("Đang tiến hành render 642 templates/vật thể (quá trình này mất khoảng 1-2 phút)...")
    cmd_render = [
        sys.executable, '-m', 'src.scripts.render_template_with_pyrender',
        'dataset_name=ycbv', f'machine.root_dir={WORK_ROOT}', '+gpus_devices=0'
    ]
    subprocess.run(cmd_render, cwd=CNOS_ROOT, check=True)
    counts = {oid: len([p for p in (template_root / f'obj_{oid:06d}').rglob('*') if p.is_file()])
              for oid in sorted(TARGET_OBJ_IDS)}

assert all(counts.get(oid, 0) >= 642 for oid in TARGET_OBJ_IDS), f"Lỗi: Không đủ 642 templates: {counts}"
print("----------------------------------------------------------------------")
print(f"✅ Templates đã sẵn sàng: {counts}")
print(f"✅ Query targets: {len(query_targets)} frame; CELL 7 chỉ gọi CNOS khi init/lost.")
print("----------------------------------------------------------------------")
