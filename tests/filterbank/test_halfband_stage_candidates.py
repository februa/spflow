"""halfband stage candidates に関する回帰試験。"""

import numpy as np

from spflow.filterbank.halfband_stage_candidates import get_known_qmf_candidate, make_known_qmf_candidates


def test_known_qmf_candidates_reconstruct_complex_signal():
    """既知 QMF 候補群について 複素信号を再構成する を確認する。"""
    rng = np.random.default_rng(60)
    x = rng.standard_normal(256) + 1j * rng.standard_normal(256)

    for candidate in make_known_qmf_candidates().values():
        stage = candidate.make_stage()
        low, high = stage.analysis(x)
        reconstructed = stage.synthesis(low, high, length=x.shape[-1])
        np.testing.assert_allclose(reconstructed, x, atol=1e-5)


def test_longer_qmf_candidates_improve_stopband_over_haar_baseline():
    """対象機能について より長い QMF 候補が Haar 基準より stopband を改善する を確認する。"""
    candidates = make_known_qmf_candidates()
    haar = candidates["haar_qmf_taps2"].response_metrics()
    db2 = candidates["daubechies_qmf_order2_taps4"].response_metrics()
    db3 = candidates["daubechies_qmf_order3_taps6"].response_metrics()
    db4 = candidates["daubechies_qmf_order4_taps8"].response_metrics()

    assert db2["low_stopband_attenuation_db"] > haar["low_stopband_attenuation_db"]
    assert db3["low_stopband_attenuation_db"] > db2["low_stopband_attenuation_db"]
    assert db4["low_stopband_attenuation_db"] > db3["low_stopband_attenuation_db"]


def test_order4_candidate_is_still_below_formal_80db_requirement():
    """4 次候補について 正式な 80 dB 条件にはまだ届かない を確認する。"""
    db4 = make_known_qmf_candidates()["daubechies_qmf_order4_taps8"].response_metrics()

    assert db4["low_stopband_attenuation_db"] < 80.0
    assert db4["high_stopband_attenuation_db"] < 80.0


def test_legacy_candidate_aliases_resolve_to_canonical_names():
    """旧候補 aliasについて 正規候補名へ解決される を確認する。"""
    assert get_known_qmf_candidate("haar2").name == "haar_qmf_taps2"
    assert get_known_qmf_candidate("db2_len4").name == "daubechies_qmf_order2_taps4"
    assert get_known_qmf_candidate("db3_len6").name == "daubechies_qmf_order3_taps6"
    assert get_known_qmf_candidate("db4_len8").name == "daubechies_qmf_order4_taps8"
