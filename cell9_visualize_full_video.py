# ==============================================================================
# CELL 9: BIỂU ĐỒ BENCHMARK FULL VIDEO — KHỚP CELL 8 LINH HOẠT
# ==============================================================================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if 'WORK_ROOT' not in globals():
    raise RuntimeError('Thiếu WORK_ROOT; hãy chạy Cell 3 trước.')

out_dir = Path(WORK_ROOT) / 'metrics'
summary_path = out_dir / 'final_benchmark_summary.csv'
pose_path = out_dir / 'pose_metrics_by_frame.csv'
if not summary_path.is_file():
    raise FileNotFoundError(f'Thiếu benchmark summary: {summary_path}. Hãy chạy Cell 8 trước.')

summary_df = pd.read_csv(summary_path)
if 'Obj_ID' not in summary_df.columns:
    raise ValueError('Summary thiếu cột Obj_ID; kiểm tra output của Cell 8.')

# Bỏ macro row và chỉ vẽ object có ID số. Không giả định video có đủ 5 object.
summary_df['_obj_id'] = pd.to_numeric(summary_df['Obj_ID'], errors='coerce')
obj_df = summary_df[summary_df['_obj_id'].notna()].copy()
obj_df['_obj_id'] = obj_df['_obj_id'].astype(int)
obj_df = obj_df.sort_values('_obj_id').reset_index(drop=True)

if obj_df.empty:
    print('Cell 8 không ghi nhận object hiện diện; bỏ qua biểu đồ để tránh chart rỗng.')
else:
    ycb_names = {
        1: '001_chips_can',
        2: '002_master_chef_can',
        3: '003_cracker_box',
        4: '004_sugar_box',
        5: '005_tomato_soup_can',
        6: '006_mustard_bottle',
        7: '007_tuna_fish_can',
        8: '008_pudding_box',
        9: '009_gelatin_box',
        10: '010_potted_meat_can',
        11: '011_banana',
        12: '019_pitcher_base',
        13: '021_bleach_cleanser',
        14: '024_bowl',
        15: '025_mug',
        16: '035_power_drill',
        17: '036_wood_block',
        18: '037_scissors',
        19: '040_large_marker',
        20: '051_large_clamp',
        21: '052_extra_large_clamp',
    }
    labels = [ycb_names.get(oid, f'Obj_{oid}') for oid in obj_df['_obj_id']]
    x = np.arange(len(labels))
    width = 0.36

    def numeric_column(frame, name):
        if name not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[name], errors='coerce')

    # Cell 8 hiện tách đúng ADD và ADD-S. Hỗ trợ CSV cũ bằng cách báo rõ fallback.
    adds_col = 'ADD-S_0.1d(%)'
    if adds_col not in obj_df.columns:
        if 'ADD_0.1d(%)' in obj_df.columns:
            print('Cảnh báo: summary cũ chỉ có ADD_0.1d(%); biểu đồ ADD-S sẽ dùng cột cũ.')
            adds_col = 'ADD_0.1d(%)'
        else:
            obj_df[adds_col] = np.nan

    # Cell 8 lưu latency từng frame, không lưu Track_Latency(ms) theo object.
    # Tính lại từ record có GT và loại register/re-init khỏi tracking latency.
    track_latency_by_obj = {}
    if pose_path.is_file():
        pose_df = pd.read_csv(pose_path)
        required_pose_cols = {'obj_id', 'track_ms', 'target_present'}
        if required_pose_cols.issubset(pose_df.columns):
            pose_df['obj_id'] = pd.to_numeric(pose_df['obj_id'], errors='coerce')
            pose_df['track_ms'] = pd.to_numeric(pose_df['track_ms'], errors='coerce')
            present = pose_df['target_present'].astype(str).str.lower().isin({'true', '1'})
            track_calls = pose_df[
                present & pose_df['track_ms'].gt(0)
            ]
            track_latency_by_obj = (
                track_calls.groupby('obj_id')['track_ms'].mean().to_dict()
            )
    obj_df['Track_Latency(ms)'] = [
        track_latency_by_obj.get(int(oid), np.nan) for oid in obj_df['_obj_id']
    ]

    # --------------------------------------------------------------------------
    # FIGURE 1: pose accuracy, errors, loss/recovery, tracking speed
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(max(13, len(labels) * 1.7), 8))
    fig.suptitle(
        'ĐÁNH GIÁ 6D POSE & TRACKING — YCB-V full scene '
        '(CNOS + FoundationPose)',
        fontsize=13,
        fontweight='bold',
    )

    adds_values = numeric_column(obj_df, adds_col)
    ax[0, 0].bar(x, adds_values, color='#2a9d8f', width=0.58)
    ax[0, 0].set_title('ADD-S đạt ngưỡng 0.1d', fontweight='bold')
    ax[0, 0].set_ylabel('Tỉ lệ đạt (%)')
    ax[0, 0].set_ylim(0, 105)
    ax[0, 0].set_xticks(x, labels, rotation=20, ha='right')
    ax[0, 0].grid(axis='y', alpha=0.3)

    trans = numeric_column(obj_df, 'Trans_Err(cm)')
    rot = numeric_column(obj_df, 'Rot_Err(deg)')
    ax[0, 1].bar(x - width / 2, trans, width=width, label='Translation (cm)', color='#457b9d')
    ax_rot = ax[0, 1].twinx()
    ax_rot.bar(x + width / 2, rot, width=width, label='Rotation (deg)', color='#e76f51')
    ax[0, 1].set_title('Sai số tư thế', fontweight='bold')
    ax[0, 1].set_ylabel('Translation error (cm)', color='#457b9d')
    ax_rot.set_ylabel('Rotation error (deg)', color='#e76f51')
    ax[0, 1].set_xticks(x, labels, rotation=20, ha='right')
    h1, l1 = ax[0, 1].get_legend_handles_labels()
    h2, l2 = ax_rot.get_legend_handles_labels()
    ax[0, 1].legend(h1 + h2, l1 + l2, loc='upper left')
    ax[0, 1].grid(axis='y', alpha=0.3)

    loss = numeric_column(obj_df, 'Loss_Rate(%)')
    redetect = numeric_column(obj_df, 'ReDetect_Rate(%)')
    ax[1, 0].bar(x - width / 2, loss, width=width, label='Tracking miss/loss (%)', color='#e63946')
    ax[1, 0].bar(x + width / 2, redetect, width=width, label='Re-init success (%)', color='#2b9348')
    ax[1, 0].set_title('Tracking loss và khả năng bắt lại', fontweight='bold')
    ax[1, 0].set_ylabel('Tỉ lệ (%)')
    ax[1, 0].set_ylim(0, 105)
    ax[1, 0].set_xticks(x, labels, rotation=20, ha='right')
    ax[1, 0].legend()
    ax[1, 0].grid(axis='y', alpha=0.3)

    track_latency = numeric_column(obj_df, 'Track_Latency(ms)')
    track_fps = 1000.0 / track_latency.replace(0, np.nan)
    ax[1, 1].bar(x, track_fps, color='#588157', width=0.58)
    ax[1, 1].set_title('Tốc độ riêng của track_one', fontweight='bold')
    ax[1, 1].set_ylabel('Tracking FPS')
    ax[1, 1].set_xticks(x, labels, rotation=20, ha='right')
    ax[1, 1].grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig_out1 = out_dir / 'ycbv_full_scene_benchmark_charts.png'
    fig.savefig(fig_out1, dpi=160, bbox_inches='tight')
    plt.show()
    print(f'Đã lưu biểu đồ benchmark: {fig_out1}')

    # --------------------------------------------------------------------------
    # FIGURE 2: registration vs tracking latency
    # --------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(max(9, len(labels) * 1.35), 4.8))
    reg_latency = numeric_column(obj_df, 'Reg_Latency(ms)')
    ax2.bar(x - width / 2, reg_latency, width=width,
            label='Register / re-init (ms)', color='#d62828')
    ax2.set_ylabel('Registration latency (ms)', color='#d62828')
    ax2.set_xticks(x, labels, rotation=20, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    ax_track = ax2.twinx()
    ax_track.bar(x + width / 2, track_latency, width=width,
                 label='Track latency (ms/frame)', color='#457b9d')
    ax_track.set_ylabel('Tracking latency (ms/frame)', color='#457b9d')
    ax2.set_title('Độ trễ register/re-init và tracking theo object', fontweight='bold')
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax_track.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper right')

    fig2.tight_layout()
    fig_out2 = out_dir / 'ycbv_full_scene_init_vs_tracking_latency.png'
    fig2.savefig(fig_out2, dpi=160, bbox_inches='tight')
    plt.show()
    print(f'Đã lưu biểu đồ latency: {fig_out2}')
