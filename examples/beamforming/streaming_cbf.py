"""FlowとFrameBufferで固定CBFを逐次実行する最小例。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow import Flow, FrameBuffer
from spflow.beamforming import CBFBeamformer


def process_chunk(
    chunk: NDArray[np.floating[Any]],
    *,
    frame_buffer: FrameBuffer,
    beamformer: CBFBeamformer,
) -> list[NDArray[Any]]:
    """到着した1 chunkから、完成した固定長ビーム出力だけを返す。

    Args:
        chunk: 多チャネル入力。shapeは`[n_channel, n_sample]`、値の単位は任意の振幅単位。
        frame_buffer: axis=1を時間軸とするフレームバッファ。
        beamformer: channel軸へ固定CBF係数を適用するビームフォーマ。

    Returns:
        完成したビーム出力のリスト。各要素のshapeは`[n_beam, frame_size]`。
        frame未完成時は空リスト、1 chunkから複数frameが完成すれば複数要素を返す。

    Raises:
        ValueError: chunkの非時間軸shapeが過去入力と異なる場合、または
            channel数がビームフォーマ係数と一致しない場合。

    Notes:
        Flowは0個・1個・複数個の受け渡しだけを補助する。入力終端の判断や
        FrameBufferのflush時期は通常のPython制御として呼び出し側が管理する。
    """
    # FrameBuffer.processはlistを返すため、Flowは完成frameだけを1段展開する。
    # CBFBeamformer.processは各frameを単一のビーム出力へ写像し、処理レートは決めない。
    return Flow.from_value(chunk).map(frame_buffer.process).map(beamformer.process).to_list()


def flush_stream(
    *,
    frame_buffer: FrameBuffer,
    beamformer: CBFBeamformer,
) -> list[NDArray[Any]]:
    """入力終端の端数をゼロ詰めし、最後のビーム出力を回収する。

    Args:
        frame_buffer: 終端前の入力を保持するフレームバッファ。
        beamformer: flushで完成したframeへ適用する固定ビームフォーマ。

    Returns:
        最終ビーム出力のリスト。各要素のshapeは`[n_beam, frame_size]`。
        端数がなければ空リストを返す。値の単位は入力振幅単位と同じ。

    Raises:
        ValueError: 保持frameのchannel数がビームフォーマ係数と一致しない場合。

    Notes:
        終端端数を暗黙に捨てると入力系列の末尾が失われるため、この例では
        `pad=True`を明示し、完成状態へしてから後段へ公開する。
    """
    # flushの結果も通常のlistなので、Flow.manyから同じビーム形成処理へ合流できる。
    final_frames = frame_buffer.flush(pad=True, fill_value=0.0)
    return Flow.many(final_frames).map(beamformer.process).to_list()


def main() -> None:
    """2チャネル同相信号を二つのchunkとして固定CBFへ入力する。

    入出力と境界条件:
        入力shapeは`[n_channel=2, n_sample]`、出力shapeは
        `[n_beam=1, frame_size=4]`。最初の3 sampleでは出力せず、次の3 sampleで
        完成frameを1個公開し、残り2 sampleはflush時にゼロ詰めする。
    """
    # steering shape: [n_channel=2, n_beam=1]。
    # 同相方向のCBF係数は各channel 1/2となり、同一入力の振幅を保存する。
    steering = np.ones((2, 1), dtype=np.complex64)
    beamformer = CBFBeamformer(steering)
    frame_buffer = FrameBuffer(frame_size=4, hop_size=4, axis=1)

    first_chunk = np.repeat(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), 2, axis=0)
    second_chunk = np.repeat(np.array([[4.0, 5.0, 6.0]], dtype=np.float32), 2, axis=0)

    first_outputs = process_chunk(
        first_chunk,
        frame_buffer=frame_buffer,
        beamformer=beamformer,
    )
    completed_outputs = process_chunk(
        second_chunk,
        frame_buffer=frame_buffer,
        beamformer=beamformer,
    )
    final_outputs = flush_stream(frame_buffer=frame_buffer, beamformer=beamformer)

    if first_outputs:
        raise RuntimeError("frame未完成時にビーム出力が公開されました。")
    np.testing.assert_allclose(completed_outputs[0], [[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_allclose(final_outputs[0], [[5.0, 6.0, 0.0, 0.0]])

    print("completed:", completed_outputs[0])
    print("flushed:", final_outputs[0])


if __name__ == "__main__":
    main()
