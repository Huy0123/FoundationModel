# Google Colab dependency cache

This setup caches only compiled artifacts for the current YCB-V pipeline. It does not cache a virtual environment, datasets, checkpoints, or model weights. Source revisions are pinned in `tools/colab_sources.json` and included in cache metadata.

## First run

In a Colab cell:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then clone your GitHub repository (replace `<repo>` with its URL) and run:

```bash
!git clone https://github.com/Huy0123/FoundationModel.git /content/FoundationModel
%cd /content/FoundationModel
!bash tools/install_colab_cached.sh
```

Select a Colab GPU runtime before installing. The installer uses the runtime's existing PyTorch and CUDA; it does not upgrade PyTorch, install conda, or change CUDA. It installs missing compiler, Boost, Eigen, and EGL development packages with `apt-get install` (no `apt upgrade`) and installs the Python build helpers `ninja` and `pybind11` before a cache-miss build.

## Later runs

Mount Drive, clone or update the repository under `/content`, then run `bash tools/install_colab_cached.sh` again. A matching cache is copied from Drive to local `/content` before pip installs its wheels. On a cache hit the installer does not compile PyTorch3D, nvdiffrast, or FoundationPose `mycpp` again.

The Drive cache path is:

```text
/content/drive/MyDrive/colab_env_cache/foundationpose/<fingerprint>/
```

## Inspect the environment and cache key

```bash
!python tools/detect_colab_env.py
!python tools/detect_colab_env.py --json
!python tools/detect_colab_env.py --fingerprint
```

The fingerprint includes Python major/minor, PyTorch, torch CUDA, GPU compute capability, and PyTorch C++ ABI. Metadata additionally compares GPU compiler versions and all pinned dependency commits before accepting a hit.

## Force rebuild

```bash
!FORCE_REBUILD_CACHE=1 bash tools/install_colab_cached.sh
```

This rebuilds the matching fingerprint directory. It does not change the source pins. To build into a different Drive directory, set `COLAB_ENV_CACHE_ROOT`.

## Timing and logs

The installer appends output to `/content/colab_setup.log`, including environment, cache status, and total elapsed seconds. Measure wall-clock time with:

```bash
!time bash tools/install_colab_cached.sh
```

## Run the full RGB-D tracking and benchmark pipeline

The pipeline runner executes the existing cells in one Python process and skips the old Kaggle installer. It accepts a standard BOP YCB-V layout or the supplied `ycbv_5cad_test_colab.zip`. The supplied archive contains RGB-D scenes, camera/GT annotations, BOP target metadata, and five OBJ CADs. Unzip it as shown earlier to `/content/data`; the runner detects `/content/data/ycbv_5cad_test` automatically. It converts those five meter-scale OBJ meshes to BOP PLY meshes in millimeters and caches them under `/content/cnos_ycbv_workspace/converted_ycbv_models/`.

The archive's five CADs map to BOP object IDs `2,4,5,6,9`:

```text
003_cracker_box       -> 2
005_tomato_soup_can   -> 4
006_mustard_bottle    -> 5
007_tuna_fish_can     -> 6
010_potted_meat_can   -> 9
```

For another standard BOP YCB-V extraction, the expected layout is:

```text
<YCB_INPUT_ROOT>/
  ycbv_test_all/test/<scene>/{rgb,depth,scene_camera.json,scene_gt.json}
  ycbv_models/models/{obj_XXXXXX.ply,models_info.json}
  ycbv_base/ycbv/test_targets_bop19.json
```

The supplied archive includes the annotations needed for pose metrics. The pipeline also needs both FoundationPose checkpoint folders: `2024-01-11-20-02-45` (scorer) and `2023-10-28-18-33-37` (refiner), each with `config.yml` and `model_best.pth`. The official FoundationPose README links to the [pretrained weights](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing); copy/extract them under `/content/drive/MyDrive/foundationpose-assets/`.

After the installer finishes building all wheels and prints its final verification success, update the cloned repository and set the asset path. The runner auto-detects the supplied archive and these five object IDs:

```python
import os
os.environ["YCB_INPUT_ROOT"] = "/content/data"
os.environ["FOUNDATIONPOSE_ASSETS_ROOT"] = "/content/drive/MyDrive/foundationpose-assets"
os.environ["YCBV_TARGET_OBJ_IDS"] = "2,4,5,6,9"
os.environ["YCBV_PREFERRED_OBJ_IDS"] = "2,4,5,6,9"
```

Run the end-to-end cells:

```python
%run /content/FoundationModel/tools/run_colab_pipeline.py
```

If the notebook already cloned the repo before this runner update, run `!git -C /content/FoundationModel pull` first. Do not start the runner while the PyTorch3D wheel is still building. After the installer reports success, restart the Colab session once so the active kernel reloads the compiled packages; then mount Drive again, pull the repo update, and run the commands above. Keep the runtime session alive until the runner writes its final success line; then copy the MP4 and metrics from the workspace to Drive.

The runner checks GPU access, BOP directory layout, CADs, and weights before starting. It then selects a scene, prepares CNOS templates, runs detector-assisted FoundationPose tracking, computes benchmark CSVs/charts, and renders the final MP4. The runner reuses the cached `mycpp` extension and skips duplicate pip installs. Results are written under `/content/cnos_ycbv_workspace/visualizations/` and `/content/cnos_ycbv_workspace/metrics/`; copy them to Drive before the Colab runtime ends.

## Cache invalidation and native import errors

The cache is rejected if Python, PyTorch/CUDA, compute capability, compiler versions, C++ ABI, or any pinned source commit differs. Edit the source lock only after deliberately choosing and reviewing new upstream revisions.

If a CUDA extension import fails, check the environment JSON and `metadata.json`, confirm that the runtime has a GPU and `nvcc`, compare C++ ABI/compiler details, then force a rebuild. Do not copy `.so` files between different fingerprints. A cache miss performs the first source build, which may take several minutes.

## Dependencies identified in the checked-in setup cells

* **Compiled and cached:** PyTorch3D and nvdiffrast are installed from pinned source with `--no-build-isolation`; FoundationPose `mycpp` is built with CMake and cached as a fingerprint-bound `.so` artifact.
* **Installed normally:** FoundationPose requirements (excluding runtime-owned PyTorch, OpenCV, NumPy, and Warp), `warp-lang==1.17.0`, CNOS/PyRender support packages, and BOP toolkit. These use pip's compatible wheels where available.
* **No separate FastSAM extension found:** FastSAM is invoked through Ultralytics; its model weights are not stored in this cache.

The repository directory must be on local `/content`; only compiled wheel and `.so` artifacts go to Drive.
