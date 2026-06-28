import pytest

from spflow import Option


def test_option_supports_dot_and_item_access():
    opt = Option({"env": {"fs": 16000}, "name": "demo"})

    assert opt.env.fs == 16000
    assert opt["env"]["fs"] == 16000
    assert opt["env.fs"] == 16000
    assert opt.name == "demo"


def test_option_get_and_require_support_dot_paths():
    opt = Option({"stft": {"nfft": 1024}})

    assert opt.get("stft.nfft") == 1024
    assert opt.get("stft.window", "hann") == "hann"
    assert opt.require("stft").nfft == 1024


def test_option_dict_like_minimal_api():
    opt = Option({"a": 1, "b": 2})

    assert "a" in opt
    assert len(opt) == 2
    assert set(opt.keys()) == {"a", "b"}
    assert dict(opt.items()) == {"a": 1, "b": 2}
    assert list(opt.values()) == [1, 2]


def test_option_keeps_dicts_inside_lists_plain():
    opt = Option({"sources": [{"f": 1000}]})

    assert opt.sources[0]["f"] == 1000
    with pytest.raises(AttributeError):
        _ = opt.sources[0].f


def test_option_missing_attribute_has_clear_message():
    opt = Option({"stft": {}})

    with pytest.raises(AttributeError, match=r"opt\.stft\.nfft の定義がありません。"):
        _ = opt.stft.nfft


def test_option_missing_require_raises_key_error():
    opt = Option({"stft": {}})

    with pytest.raises(KeyError, match=r"opt\.stft\.nfft の定義がありません。"):
        opt.require("stft.nfft")
