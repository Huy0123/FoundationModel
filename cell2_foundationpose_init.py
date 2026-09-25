# ==============================================================================
# CELL 2 — KIỂM TRA ABI, BUILD mycpp, NẠP FOUNDATIONPOSE
# Chạy sau khi đã Restart Session/Kernel từ cuối Cell 1.
# ==============================================================================
import importlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

WORKING_DIR = Path('/kaggle/working')
FP_ROOT = WORKING_DIR / 'FoundationPose'
if not (FP_ROOT / '.git').is_dir():
    raise FileNotFoundError('Thiếu FoundationPose; hãy chạy hoàn tất Cell 1 trước khi restart.')

os_path = __import__('os').environ
os_path['CUDA_HOME'] = '/usr/local/cuda'
os_path['PATH'] = f"{os_path['CUDA_HOME']}/bin:" + os_path.get('PATH', '')
os_path['PYOPENGL_PLATFORM'] = 'egl'
os_path['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import numpy as np
import scipy
import torch
import cv2

print(f'Python: {sys.version.split()[0]}')
print(f'NumPy: {np.__version__} | SciPy: {scipy.__version__}')
print(f'Torch: {torch.__version__} | Torch CUDA: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if np.__version__ != '1.26.4':
    raise RuntimeError(f'NumPy ABI không đúng: cần 1.26.4, đang có {np.__version__}. Restart session và chạy lại Cell 1.')
if not torch.cuda.is_available():
    raise RuntimeError('Kaggle chưa bật GPU. Chọn Accelerator=GPU rồi restart session.')

# Fail sớm ở đúng module đang lỗi, thay vì để lỗi xuất hiện sâu trong estimator.
import open3d
import trimesh
import nvdiffrast.torch
import pytorch3d
import warp as wp
wp.init()
print(f'OpenCV: {cv2.__version__} | Open3D: {open3d.__version__} | Warp: {wp.__version__}')
print(f'PyTorch3D: {pytorch3d.__file__}')

for path in (FP_ROOT, FP_ROOT / 'mycpp', FP_ROOT / 'mycpp' / 'build'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

print('--- Build C++ extension mycpp ---')
build_dir = FP_ROOT / 'mycpp' / 'build'
build_dir.mkdir(parents=True, exist_ok=True)
subprocess.run([
    'cmake', '..',
    f'-DPYTHON_EXECUTABLE={sys.executable}',
    f'-Dpybind11_DIR={subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"], text=True).strip()}',
    '-DCMAKE_BUILD_TYPE=Release',
], cwd=build_dir, check=True)
subprocess.run(['cmake', '--build', '.', '--parallel', '2'], cwd=build_dir, check=True)

extensions = list(build_dir.glob('*.so'))
if not extensions:
    raise RuntimeError(f'Build xong nhưng không thấy file .so trong {build_dir}')
for extension in extensions:
    shutil.copy2(extension, FP_ROOT / 'mycpp' / extension.name)
importlib.invalidate_caches()
import mycpp
print(f'mycpp OK: {mycpp.__file__}')

# Chép weights theo cách idempotent: giữ folder đích đã có đủ config.yml.
DST = FP_ROOT / 'weights'
DST.mkdir(exist_ok=True)
ASSET_ROOT = Path('/kaggle/input/datasets/aiopen1/foundationpose-assets')
for run_name in ('2024-01-11-20-02-45', '2023-10-28-18-33-37'):
    target = DST / run_name
    if (target / 'config.yml').exists():
        print(f'Weights đã có: {run_name}')
        continue
    found = next((p for p in ASSET_ROOT.glob(f'**/{run_name}')
                  if p.is_dir() and (p / 'config.yml').exists()), None)
    if found is None:
        print(f'CẢNH BÁO: Không tìm thấy weights {run_name} trong {ASSET_ROOT}')
        continue
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(found, target)
    print(f'Đã chép weights: {run_name}')

# Giữ workaround Kaggle hiện có, nhưng kiểm tra chính xác dòng trước khi sửa.
cluster_file = FP_ROOT / 'estimater.py'
cluster_src = cluster_file.read_text()
cluster_call = 'rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())'
cluster_marker = '# Kaggle-safe fallback: skip the native pose clustering call'
if cluster_call in cluster_src:
    cluster_file.write_text(cluster_src.replace(cluster_call, cluster_marker + '\n# ' + cluster_call, 1))
    print('Đã tắt native pose clustering workaround cho Kaggle.')
elif cluster_marker in cluster_src:
    print('Native pose clustering workaround đã áp dụng.')
else:
    raise RuntimeError('Không tìm thấy dòng mycpp.cluster_poses dự kiến; không tự ý sửa estimater.py.')

from estimater import FoundationPose
import Utils
assert hasattr(Utils, 'erode_depth'), 'FoundationPose Utils chưa nạp được Warp depth kernels.'
print('FoundationPose import OK; Warp depth kernels sẵn sàng.')
