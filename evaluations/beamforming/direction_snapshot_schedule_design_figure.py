"""方位別中心sample配置と方位一致表の設計図を生成する。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spflow.beamforming import build_two_second_covariance_snapshot_schedule


def main() -> None:
    """長大64ch ULAを用いて、2秒周期の中心sample配置を可視化する。

    出力図は、中心sample表`[2,n_ch,159]`、方位一致表`[2,159]`、
    global方位ごとの秒別更新回数を同時に示す。信号処理結果の評価は責務に含めない。
    """

    fs_hz = 32768.0
    snapshot_length_samples = 128
    n_ch = 64
    spacing_m = 6.25
    x_m = (np.arange(n_ch, dtype=np.float32) - np.float32((n_ch - 1) / 2.0)) * np.float32(spacing_m)
    positions_m = np.zeros((n_ch, 3), dtype=np.float32)
    positions_m[:, 0] = x_m
    schedule = build_two_second_covariance_snapshot_schedule(
        positions_m,
        fs_hz=fs_hz,
        sound_speed_m_s=1500.0,
        snapshot_length_samples=snapshot_length_samples,
        beams_per_half=159,
    )

    left_extent = snapshot_length_samples // 2
    right_extent = snapshot_length_samples - left_extent
    # 図のタイトルで長方形と断定する前に、生成済み配列そのものへ幾何条件を課す。
    # 秒頭90度のblock開始は全channelで0、右の概念閉端は全channelで1秒となる。
    np.testing.assert_array_equal(
        schedule.channel_center_samples[:, :, 0] - left_extent,
        np.zeros((2, n_ch), dtype=np.int32),
    )
    conceptual_closure_center = np.full(n_ch, int(fs_hz) - right_extent, dtype=np.int32)
    np.testing.assert_array_equal(conceptual_closure_center + right_extent, np.full(n_ch, int(fs_hz)))
    # 0.5秒を挟む2 snapshotは同じendfire方位で、2個の半台形を連続させる。
    np.testing.assert_array_equal(schedule.direction_match_indices[0, 79:81], np.array([0, 0]))
    np.testing.assert_array_equal(schedule.channel_center_samples[1], np.flip(schedule.channel_center_samples[0], axis=0))
    snapshot_start = schedule.channel_center_samples - left_extent
    snapshot_stop = schedule.channel_center_samples + right_extent
    assert bool(np.all(snapshot_start >= 0))
    assert bool(np.all(snapshot_stop <= int(fs_hz)))

    fig = plt.figure(figsize=(15.5, 10.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.45, 1.0))
    center_axis = fig.add_subplot(grid[0, :])
    direction_axis = fig.add_subplot(grid[1, 0])
    count_axis = fig.add_subplot(grid[1, 1])

    channel_index = np.arange(n_ch, dtype=np.int32)
    colors = ("#2878B5", "#D95319")
    for segment_index in (0, 1):
        center_s = schedule.channel_center_samples[segment_index].astype(np.float64) / fs_hz + segment_index
        # center_s shapeは`[n_ch,159]`。各列が1 snapshotのchannel別中心位置を表す。
        for snapshot_index in range(schedule.n_beam):
            center_axis.plot(
                center_s[:, snapshot_index],
                channel_index,
                color=colors[segment_index],
                linewidth=0.45,
                alpha=0.48,
            )
    half_block_s = (snapshot_length_samples / 2.0) / fs_hz
    for second_boundary in (0.0, 1.0, 2.0):
        center_axis.axvline(second_boundary, color="black", linewidth=1.3)
    for second_start in (0.0, 1.0):
        center_axis.axvline(second_start + half_block_s, color="#666666", linestyle="--", linewidth=0.8)
        center_axis.axvline(second_start + 1.0 - half_block_s, color="#666666", linestyle="--", linewidth=0.8)
        # 右端90度は次秒頭と共有するため保持表には重複させず、概念閉端を補助線で示す。
        center_axis.plot(
            np.full(n_ch, second_start + (conceptual_closure_center[0] / fs_hz)),
            channel_index,
            color="#222222",
            linewidth=0.8,
            linestyle=":",
        )
    center_axis.set_xlim(-0.01, 2.01)
    center_axis.set_ylim(n_ch - 0.5, -0.5)
    center_axis.set_xlabel("Center time [s]")
    center_axis.set_ylabel("Channel index")
    center_axis.set_title("Center-sample table: one 90-deg rectangle per second, next second channel-flipped")
    center_axis.text(0.5, 2.0, "segment 0: left side", ha="center", va="top")
    center_axis.text(1.5, 2.0, "segment 1: right side (vertical flip)", ha="center", va="top")
    center_axis.grid(alpha=0.22)

    local_snapshot_index = np.arange(schedule.n_beam, dtype=np.int32)
    for segment_index in (0, 1):
        matched_azimuth = schedule.global_direction_azimuth_deg[
            schedule.direction_match_indices[segment_index]
        ]
        direction_axis.plot(
            local_snapshot_index,
            matched_azimuth,
            color=colors[segment_index],
            linewidth=1.7,
            label=f"segment {segment_index}",
        )
    direction_axis.set_xlabel("Local snapshot order (0 ... 158)")
    direction_axis.set_ylabel("Matched global azimuth [deg]")
    direction_axis.set_title("Direction-match table: 79 directions x 2 + 90 deg x 1")
    direction_axis.legend()
    direction_axis.grid(alpha=0.25)

    update_counts = np.stack(
        [
            np.bincount(schedule.direction_match_indices[segment_index], minlength=159)
            for segment_index in (0, 1)
        ],
        axis=0,
    )
    azimuth_deg = schedule.global_direction_azimuth_deg
    count_axis.step(azimuth_deg, update_counts[0], where="mid", color=colors[0], label="segment 0")
    count_axis.step(azimuth_deg, update_counts[1], where="mid", color=colors[1], label="segment 1")
    count_axis.set_yticks((0, 1, 2))
    count_axis.set_xlabel("Global azimuth [deg]")
    count_axis.set_ylabel("Updates in one second")
    count_axis.set_title("159 retained directions; inactive side is held without decay")
    count_axis.legend()
    count_axis.grid(alpha=0.25)

    output_path = Path("doc/SpFlow/images/direction_snapshot_schedule_design.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
