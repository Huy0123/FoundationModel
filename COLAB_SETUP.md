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
```

After extracting the dataset and weights and setting their paths below, use the one-cell install-and-run command. On its first use it builds the compiled dependencies; subsequent uses load the Drive cache.

Select a Colab GPU runtime before installing. The installer uses the runtime's existing PyTorch and CUDA; it does not upgrade PyTorch, install conda, or change CUDA. It installs missing compiler, Boost, Eigen, and EGL development packages with `apt-get install` (no `apt upgrade`) and installs the Python build helpers `ninja` and `pybind11` before a cache-miss build. CUDA compiler parallelism defaults to two jobs to reduce Colab RAM exhaustion; set `MAX_JOBS=1` for a smaller-memory runtime, or raise it if the runtime has enough RAM.

For the reported Colab environment (`torch 2.11.0+cu128`), PyTorch3D can use a community-built binary wheel instead of compiling its CUDA/C++ extensions. This wheel is not published or tested by the PyTorch3D maintainers; the installer checks the matching Python/OS wheel and the final verification imports it. Set `PYTORCH3D_INSTALL_MODE=prebuilt` to use it and fail fast rather than silently spending time on a source build. Use `source` for the pinned upstream build, or `auto` to try the wheel and fall back to source if unavailable.

## Later runs

Mount Drive, clone or update the repository under `/content`, then run `bash tools/install_colab_cached.sh` again. A matching cache is copied from Drive to local `/content` before pip installs its wheels. On a cache hit the installer does not compile PyTorch3D, nvdiffrast, or FoundationPose `mycpp` again. During a first setup, completed wheels and the `mycpp` extension are saved separately to Drive. If Colab disconnects after a stage completes, rerunning the command resumes from those saved artifacts. In `source` mode, an interruption during PyTorch3D compilation requires that wheel build to restart; in `prebuilt` mode it downloads the matching wheel instead.

The Drive cache path is:

```text
/content/drive/MyDrive/colab_env_cache/foundationpose/<fingerprint>/
```

In-progress stages use the sibling folder `<fingerprint>.partial/`.

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

After mounting Drive, extracting the YCB-V archive, and unzipping the FoundationPose weights, set the paths. The runner auto-detects object IDs from the CAD models present: it finds `2,4,5,6,9` in the supplied five-CAD archive, or all 21 IDs when the full model library is present.

Run installation and the entire pipeline from one Colab Python cell:

```python
import os
os.environ["YCB_INPUT_ROOT"] = "/content/data"
os.environ["FOUNDATIONPOSE_ASSETS_ROOT"] = "/content/drive/MyDrive/foundationpose-assets"
os.environ["PYTORCH3D_INSTALL_MODE"] = "prebuilt"
os.environ.pop("YCBV_TARGET_OBJ_IDS", None)
os.environ.pop("YCBV_PREFERRED_OBJ_IDS", None)
!git -C /content/FoundationModel pull && bash /content/FoundationModel/tools/install_colab_cached.sh && python3 /content/FoundationModel/tools/run_colab_pipeline.py
```

With `PYTORCH3D_INSTALL_MODE=prebuilt` and the reported Torch/CUDA combination, the first run downloads PyTorch3D instead of compiling it; nvdiffrast and `mycpp` still need their first build. Later runs with the same environment fingerprint use the Drive cache. The runner starts in a fresh Python process after installation, so a kernel restart is not needed. If Colab disconnects during a source-mode build, reconnect, mount Drive, update the repository, and rerun the same cell to resume completed stages. Keep the runtime alive until the runner writes its final success line; then copy the MP4 and metrics from the workspace to Drive.

The runner checks GPU access, BOP directory layout, CADs, and weights before starting. It then selects a scene, prepares CNOS templates, runs detector-assisted FoundationPose tracking, computes benchmark CSVs/charts, and renders the final MP4. The runner reuses the cached `mycpp` extension and skips duplicate pip installs. Results are written under `/content/cnos_ycbv_workspace/visualizations/` and `/content/cnos_ycbv_workspace/metrics/`; copy them to Drive before the Colab runtime ends.

## Cache invalidation and native import errors

The cache is rejected if Python, PyTorch/CUDA, compute capability, compiler versions, C++ ABI, or any pinned source commit differs. Edit the source lock only after deliberately choosing and reviewing new upstream revisions.

If a CUDA extension import fails, check the environment JSON and `metadata.json`, confirm that the runtime has a GPU and `nvcc`, compare C++ ABI/compiler details, then force a rebuild. Do not copy `.so` files between different fingerprints. A cache miss performs the first source build, which may take several minutes.

## Dependencies identified in the checked-in setup cells

* **Compiled and cached:** PyTorch3D and nvdiffrast are installed from pinned source with `--no-build-isolation`; FoundationPose `mycpp` is built with CMake and cached as a fingerprint-bound `.so` artifact.
* **Installed normally:** FoundationPose requirements (excluding runtime-owned PyTorch, OpenCV, NumPy, and Warp), `warp-lang==1.17.0`, CNOS/PyRender support packages, and BOP toolkit. These use pip's compatible wheels where available.
* **No separate FastSAM extension found:** FastSAM is invoked through Ultralytics; its model weights are not stored in this cache.

The repository directory must be on local `/content`; only compiled wheel and `.so` artifacts go to Drive.
