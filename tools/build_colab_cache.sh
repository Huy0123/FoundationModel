#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WORK_ROOT="${COLAB_WORK_ROOT:-/content}"
FP_ROOT="$WORK_ROOT/FoundationPose"
CNOS_ROOT="$WORK_ROOT/cnos"
CACHE_ROOT="${COLAB_ENV_CACHE_ROOT:-/content/drive/MyDrive/colab_env_cache/foundationpose}"
SOURCE_LOCK="$SCRIPT_DIR/colab_sources.json"
FINGERPRINT="$("$PYTHON" "$SCRIPT_DIR/detect_colab_env.py" --fingerprint)"
CACHE_DIR="$CACHE_ROOT/$FINGERPRINT"
BUILD_DIR="$WORK_ROOT/colab_env_build/$FINGERPRINT"
WHEELHOUSE="$BUILD_DIR/wheelhouse"
PARTIAL_DIR="$CACHE_ROOT/$FINGERPRINT.partial"
MAX_JOBS="${MAX_JOBS:-2}"
PYTORCH3D_INSTALL_MODE="${PYTORCH3D_INSTALL_MODE:-source}"
PYTORCH3D_WHEEL_INDEX="${PYTORCH3D_WHEEL_INDEX:-https://miropsota.github.io/torch_packages_builder}"

if [[ ! "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got: $MAX_JOBS" >&2
  exit 2
fi
if [[ "$PYTORCH3D_INSTALL_MODE" != source && "$PYTORCH3D_INSTALL_MODE" != auto && "$PYTORCH3D_INSTALL_MODE" != prebuilt ]]; then
  echo "PYTORCH3D_INSTALL_MODE must be source, auto, or prebuilt" >&2
  exit 2
fi
export MAX_JOBS

[[ -d /content/drive/MyDrive ]] || { echo "Google Drive is not mounted at /content/drive" >&2; exit 2; }
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
command -v cmake >/dev/null || { echo "cmake missing after installing Colab build dependencies" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc missing; enable a CUDA GPU runtime" >&2; exit 2; }
"$PYTHON" -m pip install ninja pybind11 wheel
if [[ "${FORCE_REBUILD_CACHE:-0}" == 1 ]]; then
  rm -rf "$PARTIAL_DIR" "$BUILD_DIR"
fi
if [[ -d "$PARTIAL_DIR" ]]; then
  if [[ ! -f "$PARTIAL_DIR/metadata.json" ]] || ! "$PYTHON" - "$PARTIAL_DIR/metadata.json" "$SCRIPT_DIR/detect_colab_env.py" <<'PY'
import json, runpy, sys
old = json.load(open(sys.argv[1], encoding='utf-8'))
now = runpy.run_path(sys.argv[2])['detect']()
keys = ('python_major_minor', 'torch', 'torch_cuda', 'torch_cxx11_abi', 'compute_capability', 'nvcc', 'gcc', 'g++', 'sources')
raise SystemExit(0 if all(old.get(key) == now.get(key) for key in keys) else 1)
PY
  then
    echo "[CACHE] Partial artifacts match this runtime and source lock"
  else
    echo "[CACHE] Discarding incompatible partial artifacts"
    rm -rf "$PARTIAL_DIR"
  fi
fi
mkdir -p "$WHEELHOUSE" "$PARTIAL_DIR/wheelhouse" "$PARTIAL_DIR/mycpp"

if [[ ! -f "$PARTIAL_DIR/metadata.json" ]]; then
  "$PYTHON" - "$PARTIAL_DIR/metadata.json" "$SCRIPT_DIR/detect_colab_env.py" <<'PY'
import json, runpy, sys
metadata = runpy.run_path(sys.argv[2])['detect']()
with open(sys.argv[1], 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
    f.write('\n')
PY
fi

# Restore completed build stages after a Colab runtime interruption.
find "$PARTIAL_DIR/wheelhouse" -maxdepth 1 -type f -name '*.whl' -exec cp -f {} "$WHEELHOUSE/" \;

persist_artifact() {
  local source="$1" destination_dir="$2" name temporary
  name="$(basename "$source")"
  temporary="$destination_dir/.${name}.partial.$$"
  cp -f "$source" "$temporary"
  mv -f "$temporary" "$destination_dir/$name"
}

has_wheel() {
  compgen -G "$WHEELHOUSE/$1-*.whl" >/dev/null
}

checkout_locked() {
  local name="$1" destination="$2" url="$3" commit="$4"
  if [[ ! -d "$destination/.git" ]]; then
    git clone --recursive "$url" "$destination"
  fi
  git -C "$destination" fetch --tags origin "$commit"
  git -C "$destination" checkout --detach "$commit"
  git -C "$destination" submodule update --init --recursive
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]]
}

read_source() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' "$SOURCE_LOCK" "$1" "$2"; }
checkout_locked foundationpose "$FP_ROOT" "$(read_source foundationpose url)" "$(read_source foundationpose commit)"
checkout_locked cnos "$CNOS_ROOT" "$(read_source cnos url)" "$(read_source cnos commit)"

echo "[BUILD] MAX_JOBS=$MAX_JOBS (limits CUDA compiler parallelism to reduce Colab OOM risk)"
P3D_URL="$(read_source pytorch3d url)"
P3D_COMMIT="$(read_source pytorch3d commit)"
NVD_URL="$(read_source nvdiffrast url)"
NVD_COMMIT="$(read_source nvdiffrast commit)"
if has_wheel pytorch3d; then
  echo "[RESUME] Reusing completed PyTorch3D wheel from Drive partial cache"
else
  PREBUILT_SPEC="$("$PYTHON" - <<'PY'
import sys
import torch
if sys.platform == "linux" and torch.__version__.split("+")[0] == "2.11.0" and torch.version.cuda == "12.8":
    print("pytorch3d==0.7.9+d9839a9pt2.11.0cu128")
PY
)"
  USE_PREBUILT=0
  if [[ "$PYTORCH3D_INSTALL_MODE" == prebuilt && -z "$PREBUILT_SPEC" ]]; then
    echo "Prebuilt PyTorch3D wheel currently requires Linux, PyTorch 2.11.0, and CUDA 12.8; got:" >&2
    "$PYTHON" "$SCRIPT_DIR/detect_colab_env.py" --json >&2
    exit 2
  fi
  if [[ "$PYTORCH3D_INSTALL_MODE" != source && -n "$PREBUILT_SPEC" ]]; then
    echo "[DOWNLOAD] Community prebuilt PyTorch3D wheel: $PREBUILT_SPEC"
    if "$PYTHON" -m pip download --only-binary=:all: --no-deps \
      --index-url "$PYTORCH3D_WHEEL_INDEX" --dest "$WHEELHOUSE" "$PREBUILT_SPEC"; then
      USE_PREBUILT=1
    elif [[ "$PYTORCH3D_INSTALL_MODE" == prebuilt ]]; then
      echo "Could not download the matching prebuilt PyTorch3D wheel; stopping without a source build." >&2
      exit 1
    else
      echo "[WARN] Prebuilt wheel unavailable; falling back to the pinned source build"
    fi
  fi
  if (( USE_PREBUILT == 0 )); then
    echo "[BUILD] Pinned PyTorch3D source wheel"
    "$PYTHON" -m pip wheel --no-build-isolation --no-deps -w "$WHEELHOUSE" "git+$P3D_URL@$P3D_COMMIT"
  fi
  mapfile -t P3D_WHEELS < <(find "$WHEELHOUSE" -maxdepth 1 -type f -name 'pytorch3d-*.whl' -print)
  (( ${#P3D_WHEELS[@]} > 0 )) || { echo "PyTorch3D wheel download/build produced no wheel" >&2; exit 1; }
  for wheel in "${P3D_WHEELS[@]}"; do persist_artifact "$wheel" "$PARTIAL_DIR/wheelhouse"; done
fi

if has_wheel nvdiffrast; then
  echo "[RESUME] Reusing completed nvdiffrast wheel from Drive partial cache"
else
  echo "[BUILD] nvdiffrast wheel"
  "$PYTHON" -m pip wheel --no-build-isolation --no-deps -w "$WHEELHOUSE" "git+$NVD_URL@$NVD_COMMIT"
  mapfile -t NVD_WHEELS < <(find "$WHEELHOUSE" -maxdepth 1 -type f -name 'nvdiffrast-*.whl' -print)
  (( ${#NVD_WHEELS[@]} > 0 )) || { echo "nvdiffrast build produced no wheel" >&2; exit 1; }
  for wheel in "${NVD_WHEELS[@]}"; do persist_artifact "$wheel" "$PARTIAL_DIR/wheelhouse"; done
fi

echo "[BUILD] Build FoundationPose mycpp C++ extension"
MYCPP_BUILD="$FP_ROOT/mycpp/build"
mkdir -p "$MYCPP_BUILD"
mapfile -t EXTENSIONS < <(find "$PARTIAL_DIR/mycpp" -maxdepth 1 -type f -name '*.so' -print)
if (( ${#EXTENSIONS[@]} == 0 )); then
  PYBIND11_DIR="$("$PYTHON" -m pybind11 --cmakedir)"
  cmake -S "$FP_ROOT/mycpp" -B "$MYCPP_BUILD" \
    -DPYTHON_EXECUTABLE="$(command -v "$PYTHON")" \
    -Dpybind11_DIR="$PYBIND11_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$MYCPP_BUILD" --parallel "${BUILD_JOBS:-2}"
  mapfile -t EXTENSIONS < <(find "$MYCPP_BUILD" -maxdepth 2 -type f -name '*.so' -print)
  for extension in "${EXTENSIONS[@]}"; do persist_artifact "$extension" "$PARTIAL_DIR/mycpp"; done
  mapfile -t EXTENSIONS < <(find "$PARTIAL_DIR/mycpp" -maxdepth 1 -type f -name '*.so' -print)
fi
(( ${#EXTENSIONS[@]} > 0 )) || { echo "mycpp build produced no .so" >&2; exit 1; }

TMP_CACHE="$CACHE_DIR.tmp.$$"
rm -rf "$TMP_CACHE"
mkdir -p "$TMP_CACHE/wheelhouse" "$TMP_CACHE/mycpp"
cp -a "$WHEELHOUSE"/. "$TMP_CACHE/wheelhouse/"
for extension in "${EXTENSIONS[@]}"; do cp -a "$extension" "$TMP_CACHE/mycpp/"; done
"$PYTHON" - "$TMP_CACHE/metadata.json" "$SCRIPT_DIR/detect_colab_env.py" <<'PY'
import json, runpy, sys
metadata = runpy.run_path(sys.argv[2])['detect']()
metadata['artifact_format'] = 1
with open(sys.argv[1], 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
    f.write('\n')
PY
rm -rf "$CACHE_DIR"
mv "$TMP_CACHE" "$CACHE_DIR"
rm -rf "$PARTIAL_DIR"
echo "[BUILD] Cache saved: $CACHE_DIR"
