"""spflow.flow を実装するモジュール。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Concatenate, Generic, NoReturn, ParamSpec, TypeVar, final, overload

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ValueT = TypeVar("ValueT")
ArgsT = ParamSpec("ArgsT")


@final
class Flow(Generic[InputT]):
    """0 個・1 個・複数個の値を同一インターフェースで扱う軽量コンテナ。

    パイプライン中で「単一値も複数値も同じ map で流したい」用途を想定する。
    逐次計算グラフの制御補助であり、遅延評価や並列実行は責務に含めない。

    型引数`InputT`は保持する項目の型だけを表し、項目数や処理レートを表さない。
    したがって`Flow[Frame]`も0個・1個・複数個のFrameを保持できる。

    Flowは継承による拡張を公開契約に含めない。処理追加はFlowのsubclassではなく、
    `map`へ渡す通常の関数または状態を持つ独立クラスとして実装する。
    """

    def __init__(self, items: list[InputT] | None = None) -> None:
        self._items = [] if items is None else list(items)

    @classmethod
    def empty(cls) -> Flow[InputT]:
        """型引数を維持した空のFlowを返す。"""
        return cls([])

    @staticmethod
    def from_value(value: ValueT) -> Flow[ValueT]:
        """単一値を1要素Flowとして包む。

        `value=None`も1項目として保持し、次の`map`へ渡す。これにより、値がない周期でも
        現在段の状態更新を実行できる。callbackが返した`None`は次段出力なしとして扱うため、
        入力地点の`None`と処理結果の`None`では境界上の意味が異なる。

        このfactoryは常に`Flow[ValueT]`を返す。Flowのsubclass生成は契約に含めない。
        """
        return Flow[ValueT]([value])

    @staticmethod
    def many(items: Iterable[ValueT]) -> Flow[ValueT]:
        """反復可能オブジェクトから項目型を維持したFlowを作る。

        このfactoryは常に`Flow[ValueT]`を返す。Flowのsubclass生成は契約に含めない。
        """
        return Flow[ValueT](list(items))

    @overload
    def map(
        self,
        func: Callable[Concatenate[InputT, ArgsT], None],
        *args: ArgsT.args,
        **kwargs: ArgsT.kwargs,
    ) -> Flow[NoReturn]: ...

    @overload
    def map(
        self,
        func: Callable[
            Concatenate[InputT, ArgsT],
            OutputT | None | list[OutputT] | Flow[OutputT],
        ],
        *args: ArgsT.args,
        **kwargs: ArgsT.kwargs,
    ) -> Flow[OutputT]: ...

    def map(
        self,
        func: Callable[
            Concatenate[InputT, ArgsT],
            OutputT | None | list[OutputT] | Flow[OutputT],
        ],
        *args: ArgsT.args,
        **kwargs: ArgsT.kwargs,
    ) -> Flow[OutputT]:
        """各要素へ関数を適用し、結果を平坦化して新しい Flow を返す。

        `None` は破棄し、`Flow` と `list` は 1 段だけ展開する。これは
        パイプラインの節点が「値なし」「複数値」を返してもそのまま連結できるようにするためである。

        `InputT`はcallbackの第1引数型、`OutputT`は戻り値の項目型として伝播する。
        型付けは実行時の項目数や`None`の境界規約を変更しない。
        """
        outputs: list[OutputT] = []
        for item in self._items:
            # None は drop、Flow/list は 1 段展開することで、節点ごとの分岐結果を自然に連結する。
            result = func(item, *args, **kwargs)
            if result is None:
                continue
            if isinstance(result, Flow):
                outputs.extend(result._items)
                continue
            if isinstance(result, list):
                outputs.extend(result)
                continue
            outputs.append(result)
        return Flow[OutputT](outputs)

    def to_list(self) -> list[InputT]:
        """項目型を維持した通常のlistとして保持要素を返す。"""
        return list(self._items)

    def __repr__(self) -> str:
        return f"Flow({self._items!r})"
