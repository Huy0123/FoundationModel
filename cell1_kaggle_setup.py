# ==============================================================================
# CELL 1 — CÀI MÔI TRƯỜNG MỘT LẦN, GIỮ NGUYÊN PYTORCH/CUDA CỦA KAGGLE
# Chạy Cell 1 xong: Restart Session/Kernel, rồi chạy Cell 2.
# ==============================================================================
import os
import re
import sys
import subprocess
from pathlib import Path

WORKING_DIR = Path('/kaggle/working')
FP_ROOT = WORKING_DIR / 'FoundationPose'
CONSTRAINT_FILE = WORKING_DIR / 'foundationpose-constraints.txt'
FILTERED_REQ = WORKING_DIR / 'foundationpose-requirements-kaggle.txt'

os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get('PATH', '')
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

def run(cmd, *, cwd=None):
    print('+', ' '.join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)

if not (FP_ROOT / '.git').is_dir():
    run(['git', 'clone', '--recursive', 'https://github.com/NVlabs/FoundationPose.git', FP_ROOT])

# Cài công cụ build hệ thống trước khi bắt đầu build extension.
run(['bash', '-lc', 'apt-get update -qq && apt-get install -y -qq build-essential cmake libeigen3-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libboost-program-options-dev'])
run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'pip', 'setuptools', 'wheel', 'ninja', 'pybind11'])

# Pin duy nhất trước mọi import native. Cell 2 tuyệt đối không thay đổi NumPy.
CONSTRAINT_FILE.write_text('numpy==1.26.4\n')
req_file = FP_ROOT / 'requirements.txt'
if not req_file.exists():
    raise FileNotFoundError(f'Không tìm thấy {req_file}')

# requirements upstream có thể yêu cầu NumPy 2.x và cài lại torch/torchvision.
# Bỏ các gói đó để giữ NumPy 1.26.4 và PyTorch/CUDA do Kaggle cung cấp.
blocked = {'numpy', 'torch', 'torchvision', 'torchaudio', 'opencv-python',
           'opencv-python-headless', 'opencv-contrib-python'}
filtered = []
for line in req_file.read_text().splitlines():
    match = re.match(r'^\s*([A-Za-z0-9_.-]+)', line)
    name = match.group(1).lower().replace('_', '-') if match else ''
    if name in blocked or not line.strip() or line.lstrip().startswith('#'):
        continue
    filtered.append(line)
FILTERED_REQ.write_text('\n'.join(filtered) + '\n')

# Một lượt giải dependency có constraint để pip không tự nâng NumPy lên 2.x.
run([
    sys.executable, '-m', 'pip', 'install', '-q', '--constraint', CONSTRAINT_FILE,
    'numpy==1.26.4', 'scipy==1.13.1', 'trimesh==4.4.1',
    'opencv-python==4.10.0.84', 'ultralytics', 'timm', 'open3d',
    '-r', FILTERED_REQ,
])

# Build extension bằng đúng torch đã có trong Kaggle; không để pip tự thay torch.
run([
    sys.executable, '-m', 'pip', 'install', '-q', '--no-build-isolation',
    '--constraint', CONSTRAINT_FILE,
    'git+https://github.com/facebookresearch/pytorch3d.git',
    'git+https://github.com/NVlabs/nvdiffrast.git',
])

print('\nCài đặt xong. Hãy Restart Session/Kernel ngay bây giờ để nạp lại NumPy/native libraries.')
print('Sau khi restart, chạy CELL 2. Không chạy lại lệnh pip cài NumPy ở các cell sau.')
