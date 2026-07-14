"""flow に関する回帰試験。"""

# 値伝搬部品は境界条件の取り扱いが不具合源になりやすいため、
# バッファ残量やスケジューリング順序が崩れないことを小さな入力で固定する。

import numpy as np

from spflow import Flow


def test_flow_from_value_keeps_none_as_value():
    """Flowで None を値として保持することを確認する。"""
    assert Flow.from_value(None).to_list() == [None]


def test_flow_none_input_calls_current_stage_and_none_output_stops_next_stage():
    """None入力は現在段へ通知し、None出力は後段を呼ばない境界契約を確認する。"""
    current_stage_inputs: list[None] = []
    downstream_inputs: list[str] = []

    def process_current_stage(value: None) -> None:
        # 入力値がない周期でも、現在段は状態更新やtimeout判定を実行できなければならない。
        current_stage_inputs.append(value)
        return None

    def process_downstream(value: str) -> str:
        downstream_inputs.append(value)
        return value

    outputs = Flow.from_value(None).map(process_current_stage).map(process_downstream).to_list()

    assert current_stage_inputs == [None]
    assert downstream_inputs == []
    assert outputs == []


def test_flow_many_consumes_generator():
    """Flowで generator を最後まで消費することを確認する。"""
    flow = Flow.many(x * 2 for x in range(3))

    assert flow.to_list() == [0, 2, 4]


def test_flow_map_expands_list_and_flow_and_drops_none():
    """Flowで list と Flow を展開しつつ None を落とすことを確認する。"""
    flow = Flow.many([1, 2, 3]).map(
        lambda x: None if x == 1 else [x, x + 10] if x == 2 else Flow.from_value(x + 20)
    )

    assert flow.to_list() == [2, 12, 23]


def test_flow_map_treats_tuple_and_ndarray_as_single_value():
    """Flowで tuple と ndarray を単一値として扱うことを確認する。"""
    array = np.array([1, 2])
    result = Flow.from_value("x").map(lambda _: (1, 2)).map(lambda x: x).to_list()
    result_array = Flow.from_value("x").map(lambda _: array).to_list()

    assert result == [(1, 2)]
    assert len(result_array) == 1
    assert result_array[0] is array


def test_flow_map_returns_new_instance():
    """Flowで新しい Flow インスタンスを返すことを確認する。"""
    flow1 = Flow.many([1, 2])
    flow2 = flow1.map(lambda x: x + 1)

    assert flow1.to_list() == [1, 2]
    assert flow2.to_list() == [2, 3]
