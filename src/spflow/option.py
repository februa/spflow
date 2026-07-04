"""spflow.option を実装するモジュール。"""

from __future__ import annotations

from collections.abc import Iterator, ItemsView, KeysView, Mapping, ValuesView
from typing import Any


class Option:
    """入れ子辞書へドットアクセスを与える設定ラッパー。

    `dict` ベースの設定を `opt.foo.bar` や `opt["foo.bar"]` で読めるようにし、
    エラー時には欠落パスを明示する。設定解決補助が責務であり、型変換や検証ロジックの
    全面代替は責務に含めない。
    """

    def __init__(self, data: Mapping[str, Any], path: str = "opt") -> None:
        if not isinstance(data, Mapping):
            raise TypeError("Option requires a mapping input.")
        self._data = dict(data)
        self._path = path

    def __getattr__(self, name: str) -> Any:
        try:
            return self._resolve_key(name, attr=True)
        except KeyError as exc:
            raise AttributeError(str(exc)) from None

    def __getitem__(self, key: str) -> Any:
        return self._resolve_path(key)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"Option({self._data!r})"

    def keys(self) -> KeysView[str]:
        """直下キー集合を返す。"""
        return self._data.keys()

    def items(self) -> ItemsView[str, Any]:
        """直下の `(key, value)` ビューを返す。"""
        return self._data.items()

    def values(self) -> ValuesView[Any]:
        """直下値ビューを返す。"""
        return self._data.values()

    def get(self, key: str, default: Any = None) -> Any:
        """ドット区切りパスを解決し、未定義時は既定値を返す。"""
        try:
            return self._resolve_path(key)
        except KeyError:
            return default

    def require(self, key: str) -> Any:
        """ドット区切りパスを必須として解決する。未定義時は KeyError を送出する。"""
        return self._resolve_path(key)

    def _resolve_path(self, key: str) -> Any:
        if "." not in key:
            return self._resolve_key(key, attr=False)

        current: Any = self
        current_path = self._path
        for part in key.split("."):
            # ドット区切りパスを 1 段ずつ辿り、欠落位置をその場で具体的に報告する。
            if not isinstance(current, Option):
                raise KeyError(f"{current_path}.{part} is not a mapping.")
            current_path = f"{current_path}.{part}"
            if part not in current._data:
                raise KeyError(f"{current_path} の定義がありません。")
            current = current._wrap_if_mapping(current._data[part], current_path)
        return current

    def _resolve_key(self, key: str, attr: bool) -> Any:
        if key not in self._data:
            raise KeyError(f"{self._path}.{key} の定義がありません。")
        return self._wrap_if_mapping(self._data[key], f"{self._path}.{key}")

    @staticmethod
    def _wrap_if_mapping(value: Any, path: str) -> Any:
        if isinstance(value, Mapping):
            return Option(value, path=path)
        return value
