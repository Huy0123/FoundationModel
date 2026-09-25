# YCB-V: FastSAM + CNOS + FoundationPose

Chạy từng file như **một cell Kaggle** theo thứ tự:

1. `cell1_kaggle_setup.py`; restart kernel.
2. `cell2_foundationpose_init.py`.
3. `cell3_ycbv_full_video.py` (chọn nguyên scene, nạp 21 CAD).
4. `cell4.py` (CNOS và CAD templates).
5. `cell5_cnos_full_video.py` (chỉ định nghĩa `detect_frame`, chưa inference).
6. `cell7_tracking_flexible.py` (FastSAM/CNOS tại frame đầu và khi lost; FoundationPose `track_one` trên frame thường).
7. `cell8_benchmark_flexible.py`, rồi `cell9_visualize_full_video.py` và `cell10.py`.

Cell 6 và cell đánh giá mask cũ đã bỏ. GT chỉ được đọc để chọn scene và tính lỗi pose sau inference. Mask của detector chỉ dùng cho FoundationPose `register`; không chấm IoU/Dice. Frame có GT nhưng không có pose được tính là thất bại ở ADD/ADD-S 0.1d và tracking success.

Ngưỡng phát hiện/lost ở đầu Cell 7: `MIN_DETECTION_SCORE`, `DETECT_RETRY_INTERVAL`, `MAX_TRANSLATION_STEP_M`, `MIN_DEPTH_SUPPORT`, `DEPTH_TOLERANCE_M`. FoundationPose không trả confidence lost; kiểm tra hình học/depth là heuristic cần hiệu chỉnh theo video. Khi một vật đang track tốt, detector sẽ không tìm vật mới xuất hiện cho đến khi có lost; đó là chính sách chỉ detect ở init/recovery.

`metrics/pose_metrics_by_frame.csv`, `final_benchmark_summary.csv`, `tracking_event_counts.csv`, `detector_events.csv` là báo cáo pipeline. `Pipeline_FPS` tính cả detector, đăng ký lại, tracking và I/O; cache hit được ghi riêng. Để đo cold run, đặt `FORCE_CNOS_INFERENCE=True` ở Cell 5.

Trước khi chạy Kaggle, kiểm tra đường dẫn dataset/assets và JSON output CNOS trong Cell 3/5. Repo upstream có thể đổi schema hoặc tên file; Cell 5 sẽ báo lỗi khi output sai frame/category thay vì dùng nhầm cache.

## Google Colab setup

See [COLAB_SETUP.md](COLAB_SETUP.md) for the Drive-backed compiled dependency cache and setup commands. The installer uses pinned upstream sources and the Colab runtime's existing PyTorch/CUDA.
