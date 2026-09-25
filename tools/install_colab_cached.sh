#!/usr/bin/env bash
set -Eeuo pipefail

START_TIME="$(date +%s)"
LOG_FILE="${COLAB_SETUP_LOG:-/content/colab_setup.log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WORK_ROOT="${COLAB_WORK_ROOT:-/content}"
FP_ROOT="$WORK_ROOT/FoundationPose"
CNOS_ROOT="$WORK_ROOT/cnos"
CACHE_ROOT="${COLAB_ENV_CACHE_ROOT:-/content/drive/MyDrive/colab_env_cache/foundationpose}"
FINGERPRINT="$("$PYTHON" "$SCRIPT_DIR/detect_colab_env.py" --fingerprint)"
CACHE_DIR="$CACHE_ROOT/$FINGERPRINT"
SOURCE_LOCK="$SCRIPT_DIR/colab_sources.json"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if command -v apt-get >/dev/null && command -v dpkg-query >/dev/null; then
  APT_PACKAGES=(build-essential cmake libeigen3-dev libboost-program-options-dev libboost-system-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev)
  MISSING_APT=()
  for package in "${APT_PACKAGES[@]}"; do
    dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed' || MISSING_APT+=("$package")
  done
  if (( ${#MISSING_APT[@]} > 0 )); then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${MISSING_APT[@]}"
  fi
fi

echo "[ENV]"
"$PYTHON" "$SCRIPT_DIR/detect_colab_env.py" --json
if ! "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU runtime is required"'; then
  echo "Enable a Colab GPU runtime; PyTorch/CUDA will not be upgraded by this installer." >&2
  exit 2
fi
if [[ ! -d /content/drive/MyDrive ]]; then
  echo "Mount Google Drive first: from google.colab import drive; drive.mount('/content/drive')" >&2
  exit 2
fi

read_source() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' "$SOURCE_LOCK" "$1" "$2"; }
checkout_locked() {
  local name="$1" destination="$2" url="$3" commit="$4"
  if [[ ! -d "$destination/.git" ]]; then git clone --recursive "$url" "$destination"; fi
  git -C "$destination" fetch --tags origin "$commit"
  git -C "$destination" checkout --detach "$commit"
  git -C "$destination" submodule update --init --recursive
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]]
}
checkout_locked foundationpose "$FP_ROOT" "$(read_source foundationpose url)" "$(read_source foundationpose commit)"
checkout_locked cnos "$CNOS_ROOT" "$(read_source cnos url)" "$(read_source cnos commit)"

CACHE_STATUS=MISS
FORCE_REBUILD="${FORCE_REBUILD_CACHE:-0}"
if [[ "$FORCE_REBUILD" != 1 && -f "$CACHE_DIR/metadata.json" && -d "$CACHE_DIR/wheelhouse" && -d "$CACHE_DIR/mycpp" ]]; then
  if "$PYTHON" - "$CACHE_DIR/metadata.json" "$SCRIPT_DIR/detect_colab_env.py" <<'PY'
import json, runpy, sys
old = json.load(open(sys.argv[1], encoding='utf-8'))
now = runpy.run_path(sys.argv[2])['detect']()
keys = ('python_major_minor', 'torch', 'torch_cuda', 'torch_cxx11_abi', 'compute_capability', 'nvcc', 'gcc', 'g++', 'sources')
raise SystemExit(0 if old.get('artifact_format') == 1 and all(old.get(k) == now.get(k) for k in keys) else 1)
PY
  then CACHE_STATUS=HIT; fi
fi

if [[ "$CACHE_STATUS" == HIT ]]; then
  echo "[CACHE HIT] $CACHE_DIR"
  LOCAL_ARTIFACTS="$WORK_ROOT/colab_env_cache_local/$FINGERPRINT"
  rm -rf "$LOCAL_ARTIFACTS"
  mkdir -p "$LOCAL_ARTIFACTS"
  cp -a "$CACHE_DIR/wheelhouse" "$LOCAL_ARTIFACTS/"
  cp -a "$CACHE_DIR/mycpp" "$LOCAL_ARTIFACTS/"
else
  echo "[CACHE MISS] $CACHE_DIR"
  bash "$SCRIPT_DIR/build_colab_cache.sh"
  CACHE_STATUS=MISS
  LOCAL_ARTIFACTS="$WORK_ROOT/colab_env_cache_local/$FINGERPRINT"
  rm -rf "$LOCAL_ARTIFACTS"
  mkdir -p "$LOCAL_ARTIFACTS"
  cp -a "$CACHE_DIR/wheelhouse" "$LOCAL_ARTIFACTS/"
  cp -a "$CACHE_DIR/mycpp" "$LOCAL_ARTIFACTS/"
fi

echo "[INSTALL] Install cached compiled wheels"
"$PYTHON" -m pip install --no-build-isolation --no-deps --no-index --find-links "$LOCAL_ARTIFACTS/wheelhouse" pytorch3d nvdiffrast
mkdir -p "$FP_ROOT/mycpp"
cp -a "$LOCAL_ARTIFACTS/mycpp/"*.so "$FP_ROOT/mycpp/"

echo "[INSTALL] Install missing Python dependencies without changing PyTorch/CUDA"
CONSTRAINT_FILE="$WORK_ROOT/foundationpose-constraints.txt"
FILTERED_REQ="$WORK_ROOT/foundationpose-requirements-colab.txt"
printf 'numpy==1.26.4\n' > "$CONSTRAINT_FILE"
"$PYTHON" - "$FP_ROOT/requirements.txt" "$FILTERED_REQ" <<'PY'
import re, sys
blocked = {'numpy','torch','torchvision','torchaudio','opencv-python','opencv-python-headless','opencv-contrib-python','pyopengl-accelerate','warp-lang'}
lines=[]
for line in open(sys.argv[1], encoding='utf-8'):
    match=re.match(r'^\s*([A-Za-z0-9_.-]+)', line)
    name=match.group(1).lower().replace('_','-') if match else ''
    if line.strip() and not line.lstrip().startswith('#') and name not in blocked:
        lines.append(line.rstrip())
open(sys.argv[2], 'w', encoding='utf-8').write('\n'.join(lines)+'\n')
PY
"$PYTHON" -m pip install --constraint "$CONSTRAINT_FILE" numpy==1.26.4 scipy==1.13.1 trimesh==4.4.1 \
  opencv-python==4.10.0.84 ultralytics==8.0.135 timm open3d warp-lang==1.17.0 \
  -r "$FILTERED_REQ"
"$PYTHON" -m pip install --constraint "$CONSTRAINT_FILE" hydra-core==1.3.2 hydra-colorlog pyrender gdown \
  ruamel.yaml distinctipy fvcore iopath onnx onnxruntime pycocotools
BOP_WHEEL_DIR="$WORK_ROOT/bop_toolkit_wheel"
mkdir -p "$BOP_WHEEL_DIR"
BOP_URL="$(read_source bop_toolkit url)"
BOP_COMMIT="$(read_source bop_toolkit commit)"
"$PYTHON" -m pip install --constraint "$CONSTRAINT_FILE" "git+$BOP_URL@$BOP_COMMIT"
"$PYTHON" -m pip install --no-deps --force-reinstall PyOpenGL==3.1.10 setuptools==80.9.0
"$PYTHON" -m pip uninstall -y PyOpenGL-accelerate || true

echo "[VERIFY]"
"$PYTHON" "$SCRIPT_DIR/verify_colab_env.py" --foundationpose "$FP_ROOT" --cnos "$CNOS_ROOT"
ELAPSED=$(( $(date +%s) - START_TIME ))
echo "Total setup time: $ELAPSED seconds"
echo "Cache status: $CACHE_STATUS"
