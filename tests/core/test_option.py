"""option に関する回帰試験。"""

# コア部品は境界条件の取り扱いが不具合源になりやすいため、
# バッファ残量やスケジューリング順序が崩れないことを小さな入力で固定する。

import pytest

from spflow import Option


def test_option_supports_dot_and_item_access():
    """Optionがドットアクセスと添字アクセスの両方をサポートすることを確認する。"""
    opt = Option({"env": {"fs": 16000}, "name": "demo"})

    assert opt.env.fs == 16000
    assert opt["env"]["fs"] == 16000
    assert opt["env.fs"] == 16000
    assert opt.name == "demo"


def test_option_get_and_require_support_dot_paths():
    """Optionについて `get` と `require` がドット区切りパスをサポートする を確認する。"""
    opt = Option({"stft": {"nfft": 1024}})

    assert opt.get("stft.nfft") == 1024
    assert opt.get("stft.window", "hann") == "hann"
    assert opt.require("stft").nfft == 1024


def test_option_dict_like_minimal_api():
    """Optionが最小限の dict 風 API を提供することを確認する。"""
    opt = Option({"a": 1, "b": 2})

    assert "a" in opt
    assert len(opt) == 2
    assert set(opt.keys()) == {"a", "b"}
    assert dict(opt.items()) == {"a": 1, "b": 2}
    assert list(opt.values()) == [1, 2]


def test_option_keeps_dicts_inside_lists_plain():
    """Optionがlist 内の dict をそのまま保持することを確認する。"""
    opt = Option({"sources": [{"f": 1000}]})

    assert opt.sources[0]["f"] == 1000
    with pytest.raises(AttributeError):
        _ = opt.sources[0].f


def test_option_missing_attribute_has_clear_message():
    """Optionで欠損属性時のメッセージが明確であることを確認する。"""
    opt = Option({"stft": {}})

    with pytest.raises(AttributeError, match=r"opt\.stft\.nfft の定義がありません。"):
        _ = opt.stft.nfft


def test_option_missing_require_raises_key_error():
    """Optionで require 時に KeyError を送出することを確認する。"""
    opt = Option({"stft": {}})

    with pytest.raises(KeyError, match=r"opt\.stft\.nfft の定義がありません。"):
        opt.require("stft.nfft")
