"""内部用の簡易 validation helper 群をまとめる。"""

from __future__ import annotations


def require(condition: bool, message: str) -> None:
    """条件を満たさない場合に `ValueError` を送出する。"""
    if not condition:
        raise ValueError(message)


def require_positive_int(name: str, value: int) -> None:
    """正の整数パラメータを検証する。"""
    require(value > 0, f"{name} must be positive.")


def require_non_negative_int(name: str, value: int) -> None:
    """0 以上の整数パラメータを検証する。"""
    require(value >= 0, f"{name} must be non-negative.")


def require_positive_float(name: str, value: float) -> None:
    """正の実数パラメータを検証する。"""
    require(value > 0.0, f"{name} must be positive.")


def require_non_negative_float(name: str, value: float) -> None:
    """0 以上の実数パラメータを検証する。"""
    require(value >= 0.0, f"{name} must be non-negative.")


def require_index_in_range(name: str, index: int, size: int) -> None:
    """添字が `[0, size)` に入ることを検証する。"""
    require(0 <= index < size, f"{name} is out of range.")
