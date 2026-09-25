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
mkdir -p "$WHEELHOUSE" "$CACHE_DIR"

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

echo "[BUILD] Build source-pinned PyTorch3D and nvdiffrast wheels (no build isolation)"
P3D_URL="$(read_source pytorch3d url)"
P3D_COMMIT="$(read_source pytorch3d commit)"
NVD_URL="$(read_source nvdiffrast url)"
NVD_COMMIT="$(read_source nvdiffrast commit)"
"$PYTHON" -m pip wheel --no-build-isolation --no-deps -w "$WHEELHOUSE" "git+$P3D_URL@$P3D_COMMIT"
"$PYTHON" -m pip wheel --no-build-isolation --no-deps -w "$WHEELHOUSE" "git+$NVD_URL@$NVD_COMMIT"

echo "[BUILD] Build FoundationPose mycpp C++ extension"
MYCPP_BUILD="$FP_ROOT/mycpp/build"
mkdir -p "$MYCPP_BUILD"
PYBIND11_DIR="$("$PYTHON" -m pybind11 --cmakedir)"
cmake -S "$FP_ROOT/mycpp" -B "$MYCPP_BUILD" \
  -DPYTHON_EXECUTABLE="$(command -v "$PYTHON")" \
  -Dpybind11_DIR="$PYBIND11_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$MYCPP_BUILD" --parallel "${BUILD_JOBS:-2}"
mapfile -t EXTENSIONS < <(find "$MYCPP_BUILD" -maxdepth 2 -type f -name '*.so' -print)
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
echo "[BUILD] Cache saved: $CACHE_DIR"
