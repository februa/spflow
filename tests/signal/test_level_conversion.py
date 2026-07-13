"""LevelConverterの数式、reference、スペクトル表現契約を検証する。"""

from __future__ import annotations

import numpy as np
import pytest

from spflow.level_conversion import (
    LevelConverter,
    level_10log10_conjpair_power,
    level_10log10_onesided_psd,
    level_10log10_power,
    level_10log10_twosided_psd,
    level_20log10_onesided_asd,
    level_20log10_rms,
    level_20log10_twosided_asd,
)
from spflow.spectral_level import (
    noise_asd_level_db_to_band_rms,
    one_sided_rfft_bin_rms_power,
)


def _rms_converter(reference_rms: float = 1.0) -> LevelConverter:
    """testで共通利用するRMS入出力converterを生成する。"""

    definition = level_20log10_rms(
        reference_rms=reference_rms,
        reference_label="input RMS",
    )
    return LevelConverter(input_definition=definition, output_definition=definition)


def test_rms_converter_round_trip_preserves_scalar_array_and_reference() -> None:
    """scalar、NumPy scalar、配列のRMS levelが同じreferenceで往復することを確認する。"""

    converter = _rms_converter(reference_rms=2.0)
    levels_db = np.array([-12.0, -6.0, 0.0, 3.0], dtype=np.float64)

    rms = converter.input_to_rms(levels_db)
    round_trip_db = converter.output_to_level(rms)
    numpy_scalar_rms = converter.input_to_rms(np.float64(0.0))

    np.testing.assert_allclose(round_trip_db, levels_db, atol=1.0e-12)
    assert converter.input_to_rms(0.0) == pytest.approx(2.0)
    assert numpy_scalar_rms == pytest.approx(2.0)
    assert isinstance(numpy_scalar_rms, float)
    assert converter.input_level_label == "dB re input RMS"


def test_real_cosine_peak_uses_sqrt_two_times_rms() -> None:
    """0 dB RMS入力がreference RMS、実cosine peakがsqrt(2)倍になることを確認する。"""

    converter = _rms_converter(reference_rms=3.0)

    assert converter.input_to_rms(0.0) == pytest.approx(3.0)
    assert converter.input_to_real_cosine_peak(0.0) == pytest.approx(3.0 * np.sqrt(2.0))


def test_conjpair_output_matches_normalized_positive_frequency_formula() -> None:
    """正周波数係数zの出力が10log10(2|z|²/A_ref²)と一致することを確認する。"""

    input_definition = level_20log10_rms(
        reference_rms=2.0,
        reference_label="input RMS",
    )
    output_definition = level_10log10_conjpair_power(
        reference_rms=2.0,
        reference_label="input RMS",
    )
    converter = LevelConverter(
        input_definition=input_definition,
        output_definition=output_definition,
    )
    reverse_converter = LevelConverter(
        input_definition=output_definition,
        output_definition=output_definition,
    )
    input_level_db = -6.0
    rms = converter.input_to_rms(input_level_db)

    # 実cosineの正規化済み内部正周波数係数はz=A_peak/2=A_rms/sqrt(2)。
    normalized_positive_coefficient = rms / np.sqrt(2.0) * np.exp(1j * 0.37)
    observed_level_db = converter.output_to_level(normalized_positive_coefficient)
    direct_formula_db = 10.0 * np.log10(2.0 * np.abs(normalized_positive_coefficient) ** 2 / 2.0**2)

    assert observed_level_db == pytest.approx(input_level_db)
    assert observed_level_db == pytest.approx(direct_formula_db)
    assert reverse_converter.input_to_linear(input_level_db) == pytest.approx(
        np.abs(normalized_positive_coefficient)
    )


def test_converter_rejects_incompatible_reference_quantity_and_measure() -> None:
    """reference、物理量、密度/積分量の不整合を生成時に拒否することを確認する。"""

    rms_input = level_20log10_rms(reference_rms=1.0, reference_label="input RMS")

    with pytest.raises(ValueError, match="reference_label"):
        LevelConverter(
            input_definition=rms_input,
            output_definition=level_10log10_power(
                reference_power=1.0,
                reference_label="different RMS",
            ),
        )

    with pytest.raises(ValueError, match="mean-square reference"):
        LevelConverter(
            input_definition=rms_input,
            output_definition=level_10log10_power(
                reference_power=4.0,
                reference_label="input RMS",
            ),
        )

    with pytest.raises(ValueError, match="physical_quantity"):
        LevelConverter(
            input_definition=rms_input,
            output_definition=level_10log10_power(
                reference_power=1.0,
                reference_label="input RMS",
                physical_quantity="voltage",
            ),
        )

    with pytest.raises(ValueError, match="explicit integration"):
        LevelConverter(
            input_definition=rms_input,
            output_definition=level_20log10_onesided_asd(
                reference_asd=1.0,
                reference_label="input RMS",
            ),
        )


def test_asd_and_psd_definitions_connect_through_mean_square_density() -> None:
    """同じsidednessのASDとPSDが共通mean-square density ratioへ接続することを確認する。"""

    asd_definition = level_20log10_onesided_asd(
        reference_asd=2.0,
        reference_label="spectral reference",
    )
    psd_definition = level_10log10_onesided_psd(
        reference_psd=4.0,
        reference_label="spectral reference",
    )
    converter = LevelConverter(
        input_definition=asd_definition,
        output_definition=psd_definition,
    )
    levels_db = np.array([-10.0, 0.0, 6.0], dtype=np.float64)
    asd = converter.input_to_linear(levels_db)

    np.testing.assert_allclose(converter.output_to_level(asd**2), levels_db, atol=1.0e-12)


def test_onesided_and_twosided_density_integrals_have_equal_power() -> None:
    """内部周波数のone-sided=2*two-sided PSDが全帯域積分powerを保存することを確認する。"""

    one_psd_definition = level_10log10_onesided_psd(
        reference_psd=1.0,
        reference_label="power/Hz",
    )
    two_psd_definition = level_10log10_twosided_psd(
        reference_psd=1.0,
        reference_label="power/Hz",
    )
    one_converter = LevelConverter(one_psd_definition, one_psd_definition)
    two_converter = LevelConverter(two_psd_definition, two_psd_definition)
    positive_bandwidth_hz = 300.0
    two_sided_psd = two_converter.input_to_linear(0.0)

    # 実信号の内部周波数ではone-sided PSD=2*two-sided PSD、
    # one-sided ASD=sqrt(2)*two-sided ASDであり、正負帯域のpower和を正側へ集約する。
    one_sided_psd = 2.0 * two_sided_psd
    one_sided_power = one_sided_psd * positive_bandwidth_hz
    two_sided_power = two_sided_psd * (2.0 * positive_bandwidth_hz)

    assert one_sided_power == pytest.approx(two_sided_power)
    assert one_converter.output_to_level(one_sided_psd) == pytest.approx(10.0 * np.log10(2.0))

    one_asd_definition = level_20log10_onesided_asd(
        reference_asd=1.0,
        reference_label="amplitude/sqrt(Hz)",
    )
    two_asd_definition = level_20log10_twosided_asd(
        reference_asd=1.0,
        reference_label="amplitude/sqrt(Hz)",
    )
    one_asd_converter = LevelConverter(one_asd_definition, one_asd_definition)
    two_asd_converter = LevelConverter(two_asd_definition, two_asd_definition)
    two_sided_asd = two_asd_converter.input_to_linear(0.0)
    one_sided_asd = np.sqrt(2.0) * two_sided_asd

    assert one_asd_converter.output_to_level(one_sided_asd) == pytest.approx(
        20.0 * np.log10(np.sqrt(2.0))
    )
    assert one_sided_asd**2 * positive_bandwidth_hz == pytest.approx(
        two_sided_asd**2 * (2.0 * positive_bandwidth_hz)
    )


def test_rfft_dc_nyquist_and_internal_bin_use_correct_conjpair_factors() -> None:
    """rFFTのDC/Nyquistは係数1、内部binだけconjpair係数2になることを確認する。"""

    sample_count = 8
    spectrum = np.ones(sample_count // 2 + 1, dtype=np.complex128)

    bin_power = one_sided_rfft_bin_rms_power(spectrum, sample_count=sample_count)

    expected = np.array([1.0, 2.0, 2.0, 2.0, 1.0], dtype=np.float64) / sample_count**2
    np.testing.assert_allclose(bin_power, expected)


def test_asd_bandwidth_and_fft_bin_width_are_explicitly_distinct() -> None:
    """ASDの任意帯域Bと1 FFT binのΔfを別の積分幅として扱うことを確認する。"""

    sampling_frequency_hz = 8000.0
    sample_count = 2000
    frequency_resolution_hz = sampling_frequency_hz / sample_count
    level_db = -30.0

    one_bin_rms = noise_asd_level_db_to_band_rms(
        level_db,
        bandwidth_hz=frequency_resolution_hz,
    )
    band_rms = noise_asd_level_db_to_band_rms(level_db, bandwidth_hz=1000.0)

    assert one_bin_rms == pytest.approx(10.0 ** (level_db / 20.0) * np.sqrt(4.0))
    assert band_rms == pytest.approx(10.0 ** (level_db / 20.0) * np.sqrt(1000.0))
    assert band_rms / one_bin_rms == pytest.approx(np.sqrt(1000.0 / 4.0))


def test_converter_rejects_implicit_sidedness_conversion() -> None:
    """one-sidedとtwo-sidedの密度を係数指定なしに接続しないことを確認する。"""

    with pytest.raises(ValueError, match="sidedness conversion"):
        LevelConverter(
            input_definition=level_20log10_onesided_asd(
                reference_asd=1.0,
                reference_label="amplitude/sqrt(Hz)",
            ),
            output_definition=level_20log10_twosided_asd(
                reference_asd=1.0,
                reference_label="amplitude/sqrt(Hz)",
            ),
        )
