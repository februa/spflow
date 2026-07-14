"""入力levelと出力levelの数式・基準・スペクトル規約を接続する。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from ._validation import require, require_positive_float

RealScalar = float | int | np.floating[Any] | np.integer[Any]
RealValue = RealScalar | NDArray[Any]
NumericValue = RealScalar | complex | np.complexfloating[Any, Any] | NDArray[Any]


class _LinearQuantity(Enum):
    """level式へ直接渡す線形量の定義を識別する。"""

    RMS = "rms"
    POWER = "power"
    CONJPAIR_POWER = "conjpair_power"
    ASD = "asd"
    PSD = "psd"


class _MeasureKind(Enum):
    """mean-squareが積分済みか単位Hz当たりかを識別する。"""

    INTEGRATED = "integrated_mean_square"
    DENSITY_PER_HZ = "mean_square_density_per_hz"


class _SpectrumSidedness(Enum):
    """スペクトル密度のone-sided/two-sided表現規約を識別する。"""

    ONE_SIDED = "one-sided"
    TWO_SIDED = "two-sided"


@dataclass(frozen=True)
class LevelDefinition:
    """一つのdB式と線形量、基準、スペクトル表現規約を保持する。

    利用者はこのクラスを直接構築せず、`level_20log10_rms`などの検証済みfactoryを使う。
    LevelConverterは入力definitionと出力definitionをnormalized mean-square ratioへ接続する。

    FFT、RMS測定、帯域積分、信号生成は責務に含めない。信号処理上は、測定済み線形量と
    dB levelの境界で、数式とreferenceを失わないためのimmutableな値に位置づく。
    """

    _definition_name: str
    _formula: str
    _linear_quantity: _LinearQuantity
    _measure_kind: _MeasureKind
    _sidedness: _SpectrumSidedness | None
    _reference_linear_value: float
    _reference_mean_square: float
    _reference_label: str
    _physical_quantity: str

    def __post_init__(self) -> None:
        """factory内部値にも固定されたlevel定義の不変条件を適用する。"""

        require(len(self._definition_name) > 0, "definition_name must not be empty.")
        require(len(self._formula) > 0, "formula must not be empty.")
        require_positive_float("reference_linear_value", self._reference_linear_value)
        require_positive_float("reference_mean_square", self._reference_mean_square)
        require(len(self._reference_label) > 0, "reference_label must not be empty.")
        require(len(self._physical_quantity) > 0, "physical_quantity must not be empty.")
        if self._measure_kind is _MeasureKind.INTEGRATED:
            require(self._sidedness is None, "integrated definitions must not have sidedness.")
        else:
            require(self._sidedness is not None, "spectral density definitions require sidedness.")

    @property
    def definition_name(self) -> str:
        """ドキュメントへ登録するlinear quantity definition名を返す。"""

        return self._definition_name

    @property
    def formula(self) -> str:
        """このdefinitionが採用するdB変換式を返す。"""

        return self._formula

    @property
    def level_label(self) -> str:
        """図、CSV、JSONへ記録できる明示的なdB reference labelを返す。"""

        return f"dB re {self._reference_label}"


def _make_definition(
    *,
    definition_name: str,
    formula: str,
    linear_quantity: _LinearQuantity,
    measure_kind: _MeasureKind,
    sidedness: _SpectrumSidedness | None,
    reference_linear_value: float,
    reference_mean_square: float,
    reference_label: str,
    physical_quantity: str,
) -> LevelDefinition:
    """公開factoryが検証した値からimmutable definitionを構築する。"""

    return LevelDefinition(
        _definition_name=definition_name,
        _formula=formula,
        _linear_quantity=linear_quantity,
        _measure_kind=measure_kind,
        _sidedness=sidedness,
        _reference_linear_value=float(reference_linear_value),
        _reference_mean_square=float(reference_mean_square),
        _reference_label=str(reference_label),
        _physical_quantity=str(physical_quantity),
    )


def level_20log10_rms(
    *,
    reference_rms: float,
    reference_label: str,
    physical_quantity: str = "signal",
) -> LevelDefinition:
    """RMS amplitudeのdB definitionを生成する。

    Args:
        reference_rms: 0 dBに対応するRMS amplitude。単位は対象信号の線形振幅単位。
        reference_label: 図や保存値に使うreference名。例は`input RMS`。
        physical_quantity: 接続互換性を判定する物理量名。例は`acoustic pressure`。

    Returns:
        `L = 20 log10(A_rms / A_ref)`を表すimmutable definition。

    Raises:
        ValueError: referenceが正でない、または文字列が空の場合。

    境界条件:
        入力線形量はRMSであり、実cosineのpeak amplitudeや瞬時値ではない。
    """

    reference = float(reference_rms)
    require_positive_float("reference_rms", reference)
    return _make_definition(
        definition_name="level_20log10_rms",
        formula="L = 20 log10(A_rms / A_ref)",
        linear_quantity=_LinearQuantity.RMS,
        measure_kind=_MeasureKind.INTEGRATED,
        sidedness=None,
        reference_linear_value=reference,
        reference_mean_square=reference**2,
        reference_label=reference_label,
        physical_quantity=physical_quantity,
    )


def level_10log10_power(
    *,
    reference_power: float,
    reference_label: str,
    physical_quantity: str = "signal",
) -> LevelDefinition:
    """積分済みpowerのdB definitionを生成する。

    Args:
        reference_power: 0 dBに対応する平均二乗値。単位は対象振幅単位の二乗。
        reference_label: 図や保存値に使うreference名。
        physical_quantity: 接続互換性を判定する物理量名。

    Returns:
        `L = 10 log10(P / P_ref)`を表すimmutable definition。

    Raises:
        ValueError: referenceが正でない、または文字列が空の場合。

    境界条件:
        入力は帯域積分済みpowerであり、PSDやFFT係数を直接渡してはいけない。
    """

    reference = float(reference_power)
    require_positive_float("reference_power", reference)
    return _make_definition(
        definition_name="level_10log10_power",
        formula="L = 10 log10(P / P_ref)",
        linear_quantity=_LinearQuantity.POWER,
        measure_kind=_MeasureKind.INTEGRATED,
        sidedness=None,
        reference_linear_value=reference,
        reference_mean_square=reference,
        reference_label=reference_label,
        physical_quantity=physical_quantity,
    )


def level_10log10_conjpair_power(
    *,
    reference_rms: float,
    reference_label: str,
    physical_quantity: str = "signal",
) -> LevelDefinition:
    """正周波数の正規化済み複素係数から実信号RMS levelを得るdefinitionを生成する。

    Args:
        reference_rms: 0 dBに対応する実信号RMS amplitude。
        reference_label: 図や保存値に使うreference名。
        physical_quantity: 接続互換性を判定する物理量名。

    Returns:
        `L = 10 log10(2 |z|^2 / A_ref^2)`を表すimmutable definition。

    Raises:
        ValueError: referenceが正でない、または文字列が空の場合。

    境界条件:
        zはFFT長で正規化済みの内部正周波数係数である。conjpairを持たないDC、Nyquist、
        complex baseband、解析信号には適用せず、それらにはpower definitionを使う。
    """

    reference = float(reference_rms)
    require_positive_float("reference_rms", reference)
    return _make_definition(
        definition_name="level_10log10_conjpair_power",
        formula="L = 10 log10(2 |z|^2 / A_ref^2)",
        linear_quantity=_LinearQuantity.CONJPAIR_POWER,
        measure_kind=_MeasureKind.INTEGRATED,
        sidedness=None,
        reference_linear_value=reference / np.sqrt(2.0),
        reference_mean_square=reference**2,
        reference_label=reference_label,
        physical_quantity=physical_quantity,
    )


def _level_20log10_asd(
    *,
    reference_asd: float,
    reference_label: str,
    physical_quantity: str,
    sidedness: _SpectrumSidedness,
) -> LevelDefinition:
    """明示されたsidednessのASD definitionを生成する。"""

    reference = float(reference_asd)
    require_positive_float("reference_asd", reference)
    side_name = sidedness.value.replace("-", "")
    return _make_definition(
        definition_name=f"level_20log10_{side_name}_asd",
        formula=f"L = 20 log10(ASD_{sidedness.value} / ASD_ref)",
        linear_quantity=_LinearQuantity.ASD,
        measure_kind=_MeasureKind.DENSITY_PER_HZ,
        sidedness=sidedness,
        reference_linear_value=reference,
        reference_mean_square=reference**2,
        reference_label=reference_label,
        physical_quantity=physical_quantity,
    )


def level_20log10_onesided_asd(
    *,
    reference_asd: float,
    reference_label: str,
    physical_quantity: str = "signal",
) -> LevelDefinition:
    """one-sided amplitude spectral densityのdB definitionを生成する。

    Args:
        reference_asd: 0 dBに対応するone-sided ASD。単位は振幅単位/√Hz。
        reference_label: reference名。`input RMS/sqrt(Hz)`のように密度単位を含める。
        physical_quantity: 接続互換性を判定する物理量名。

    Returns:
        `L = 20 log10(ASD_one-sided / ASD_ref)`を表すdefinition。

    Raises:
        ValueError: referenceが正でない、または文字列が空の場合。

    境界条件:
        線形ASDをRMSへ変える帯域積分は含まない。積分幅Bまたは周波数軸を別途明示する。
    """

    return _level_20log10_asd(
        reference_asd=reference_asd,
        reference_label=reference_label,
        physical_quantity=physical_quantity,
        sidedness=_SpectrumSidedness.ONE_SIDED,
    )


def level_20log10_twosided_asd(
    *,
    reference_asd: float,
    reference_label: str,
    physical_quantity: str = "signal",
) -> LevelDefinition:
    """two-sided amplitude spectral densityのdB definitionを生成する。

    Args:
        reference_asd: 0 dBに対応するtwo-sided ASD。単位は振幅単位/√Hz。
        reference_label: reference名。密度単位を省略しない。
        physical_quantity: 接続互換性を判定する物理量名。

    Returns:
        `L = 20 log10(ASD_two-sided / ASD_ref)`を表すdefinition。

    Raises:
        ValueError: referenceが正でない、または文字列が空の場合。

    境界条件:
        実信号の内部周波数ではone-sided ASDがtwo-sided ASDの√2倍になる。
    """

    return _level_20log10_asd(
        reference_asd=reference_asd,
        reference_label=reference_label,
        physical_quantity=physical_quantity,
        sidedness=_SpectrumSidedness.TWO_SIDED,
    )


def _level_10log10_psd(
    *,
    reference_psd: float,
    reference_label: str,
    physical_quantity: str,
    sidedness: _SpectrumSidedness,
) -> LevelDefinition:
    """明示されたsidednessのPSD definitionを生成する。"""

    reference = float(reference_psd)
    require_positive_float("reference_psd", reference)
    side_name = sidedness.value.replace("-", "")
    return _make_definition(
        definition_name=f"level_10log10_{side_name}_psd",
        formula=f"L = 10 log10(PSD_{sidedness.value} / PSD_ref)",
        linear_quantity=_LinearQuantity.PSD,
        measure_kind=_MeasureKind.DENSITY_PER_HZ,
        sidedness=sidedness,
        reference_linear_value=reference,
        reference_mean_square=reference,
        reference_label=reference_label,
        physical_quantity=physical_quantity,
    )


def level_10log10_onesided_psd(
    *,
    reference_psd: float,
    reference_label: str,
    physical_quantity: str = "signal",
) -> LevelDefinition:
    """one-sided power spectral densityのdB definitionを生成する。

    Args:
        reference_psd: 0 dBに対応するone-sided PSD。単位は振幅単位²/Hz。
        reference_label: reference名。密度単位を省略しない。
        physical_quantity: 接続互換性を判定する物理量名。

    Returns:
        `L = 10 log10(PSD_one-sided / PSD_ref)`を表すdefinition。

    Raises:
        ValueError: referenceが正でない、または文字列が空の場合。

    境界条件:
        線形PSDをpowerへ変える帯域積分は含まない。
    """

    return _level_10log10_psd(
        reference_psd=reference_psd,
        reference_label=reference_label,
        physical_quantity=physical_quantity,
        sidedness=_SpectrumSidedness.ONE_SIDED,
    )


def level_10log10_twosided_psd(
    *,
    reference_psd: float,
    reference_label: str,
    physical_quantity: str = "signal",
) -> LevelDefinition:
    """two-sided power spectral densityのdB definitionを生成する。

    Args:
        reference_psd: 0 dBに対応するtwo-sided PSD。単位は振幅単位²/Hz。
        reference_label: reference名。密度単位を省略しない。
        physical_quantity: 接続互換性を判定する物理量名。

    Returns:
        `L = 10 log10(PSD_two-sided / PSD_ref)`を表すdefinition。

    Raises:
        ValueError: referenceが正でない、または文字列が空の場合。

    境界条件:
        実信号の内部周波数ではone-sided PSDがtwo-sided PSDの2倍になる。
    """

    return _level_10log10_psd(
        reference_psd=reference_psd,
        reference_label=reference_label,
        physical_quantity=physical_quantity,
        sidedness=_SpectrumSidedness.TWO_SIDED,
    )


def _return_scalar_or_array(
    source: object, values: NDArray[np.float64]
) -> float | NDArray[np.float64]:
    """scalar入力ではPython float、配列入力では固定float64配列を返す。"""

    if np.asarray(source).ndim == 0:
        return float(values)
    return values


@dataclass(frozen=True)
class LevelConverter:
    """入力dB定義と出力dB定義を同じnormalized mean-square ratioへ接続する。

    入力は`input_definition`に従うdB値、出力は`output_definition`が要求する測定済み
    線形量である。入力側では線形RMS、power、ASD、PSD等を返し、出力側では観測値を
    明示されたreferenceのdBへ変換する。

    FFT、RMS測定、帯域積分、信号生成は責務に含めない。信号処理上は、離れた入力地点と
    出力評価地点が同じ物理量・reference・スペクトル表現へ接続できることを保証する。
    """

    input_definition: LevelDefinition
    output_definition: LevelDefinition

    @classmethod
    def for_definition(cls, definition: LevelDefinition) -> LevelConverter:
        """同じdefinitionを入力と出力に使うConverterを生成する。

        Args:
            definition: 入力dBと出力dBで共有するimmutable level definition。

        Returns:
            `input_definition is output_definition`となるLevelConverter。

        Raises:
            ValueError: definition内部のreferenceやスペクトル規約が不正な場合。

        境界条件:
            RMSからRMS、ASDからASDのように表面式も線形量も同じ場合だけ使う。
            RMS入力とconjpair power出力のような非対称接続には通常constructorを使う。
        """

        return cls(input_definition=definition, output_definition=definition)

    def __post_init__(self) -> None:
        """入出力definitionが同じnormalized mean-squareへ接続可能か検証する。"""

        input_definition = self.input_definition
        output_definition = self.output_definition
        require(
            input_definition._physical_quantity == output_definition._physical_quantity,
            "input and output definitions must describe the same physical_quantity.",
        )
        require(
            input_definition._reference_label == output_definition._reference_label,
            "input and output definitions must use the same reference_label.",
        )
        require(
            bool(
                np.isclose(
                    input_definition._reference_mean_square,
                    output_definition._reference_mean_square,
                    rtol=1.0e-12,
                    atol=0.0,
                )
            ),
            "input and output definitions must use the same mean-square reference.",
        )
        require(
            input_definition._measure_kind is output_definition._measure_kind,
            "spectral-density and band-integrated definitions require explicit integration.",
        )
        require(
            input_definition._sidedness is output_definition._sidedness,
            "one-sided and two-sided definitions require explicit sidedness conversion.",
        )

    @property
    def input_level_label(self) -> str:
        """入力dB値に付けるreference labelを返す。"""

        return self.input_definition.level_label

    @property
    def output_level_label(self) -> str:
        """出力dB値に付けるreference labelを返す。"""

        return self.output_definition.level_label

    @property
    def input_formula(self) -> str:
        """入力definitionが採用する数式を返す。"""

        return self.input_definition.formula

    @property
    def output_formula(self) -> str:
        """出力definitionが採用する数式を返す。"""

        return self.output_definition.formula

    @property
    def float64_tiny_level_db(self) -> float:
        """float64最小正規化線形値に対応する有限dB floorを返す。

        Returns:
            output definitionの線形量をreferenceで正規化した値が`float64.tiny`のときの
            dB level。RMS/ASD/conjpair係数は20log10、power/PSDは10log10で計算する。

        境界条件:
            これはJSON等へ有限値を保存するための明示的なfloor候補であり、
            `output_to_level`の既定動作を変更しない。floor未指定のゼロ観測値は`-inf`となる。
        """

        definition = self.output_definition
        multiplier = (
            20.0
            if definition._linear_quantity
            in {_LinearQuantity.RMS, _LinearQuantity.CONJPAIR_POWER, _LinearQuantity.ASD}
            else 10.0
        )
        return float(multiplier * np.log10(np.finfo(np.float64).tiny))

    @overload
    def input_to_linear(self, level_db: RealScalar) -> float: ...

    @overload
    def input_to_linear(self, level_db: NDArray[Any]) -> NDArray[np.float64]: ...

    def input_to_linear(self, level_db: RealValue) -> float | NDArray[np.float64]:
        """入力dB値をinput definitionが定義する線形量へ変換する。

        Args:
            level_db: 入力level。scalarまたは任意shapeの実数配列。単位はinput_level_label。

        Returns:
            input definitionの線形量。shapeは入力と同じ。RMS/ASDは振幅、power/PSDは
            平均二乗量、conjpair powerは正規化済み正周波数係数の絶対値を返す。

        Raises:
            ValueError: level_dbに非有限値が含まれる場合。

        境界条件:
            complex係数の位相はdBから決まらないため、conjpair definitionでは非負の絶対値を返す。
        """

        levels = np.asarray(level_db, dtype=np.float64)
        require(bool(np.all(np.isfinite(levels))), "level_db must contain only finite values.")
        definition = self.input_definition
        # conjpair definitionの表面式は10log10(2|z|²/A_ref²)だが、返す線形量は
        # powerではなく係数振幅|z|なので、逆変換の指数分母は20となる。
        divisor = (
            20.0
            if definition._linear_quantity
            in {_LinearQuantity.RMS, _LinearQuantity.CONJPAIR_POWER, _LinearQuantity.ASD}
            else 10.0
        )
        linear = definition._reference_linear_value * np.power(10.0, levels / divisor)
        return _return_scalar_or_array(level_db, np.asarray(linear, dtype=np.float64))

    @overload
    def input_to_rms(self, level_db: RealScalar) -> float: ...

    @overload
    def input_to_rms(self, level_db: NDArray[Any]) -> NDArray[np.float64]: ...

    def input_to_rms(self, level_db: RealValue) -> float | NDArray[np.float64]:
        """RMS input levelを線形RMS amplitudeへ変換する。

        Args:
            level_db: RMS level。scalarまたは任意shapeの実数配列。単位はinput_level_label。

        Returns:
            線形RMS amplitude。shapeは入力と同じ、単位はinput definitionの基準振幅と同じ。

        Raises:
            ValueError: input definitionがRMSでない、またはlevelが非有限の場合。

        境界条件:
            ASD、power、PSDをRMSへ暗黙変換しない。密度には明示的な帯域積分が必要である。
        """

        require(
            self.input_definition._linear_quantity is _LinearQuantity.RMS,
            "input_to_rms requires level_20log10_rms input definition.",
        )
        return self.input_to_linear(level_db)

    @overload
    def input_to_real_cosine_peak(self, level_db: RealScalar) -> float: ...

    @overload
    def input_to_real_cosine_peak(self, level_db: NDArray[Any]) -> NDArray[np.float64]: ...

    def input_to_real_cosine_peak(self, level_db: RealValue) -> float | NDArray[np.float64]:
        """RMS input levelを実cosineのpeak amplitudeへ変換する。

        Args:
            level_db: RMS level。scalarまたは任意shape。単位はinput_level_label。

        Returns:
            `sqrt(2) * A_rms`で得るpeak amplitude。shapeは入力と同じ。

        Raises:
            ValueError: input definitionがRMSでない、またはlevelが非有限の場合。

        境界条件:
            実cosine専用であり、複素指数信号やASDからpeakを暗黙生成しない。
        """

        require(
            self.input_definition._linear_quantity is _LinearQuantity.RMS,
            "input_to_real_cosine_peak requires level_20log10_rms input definition.",
        )
        rms = self.input_to_rms(level_db)
        peak = np.asarray(rms, dtype=np.float64) * np.sqrt(2.0)
        return _return_scalar_or_array(level_db, peak)

    @overload
    def output_to_level(
        self,
        observation: complex | RealScalar,
        *,
        floor_db: float | None = None,
    ) -> float: ...

    @overload
    def output_to_level(
        self,
        observation: NDArray[Any],
        *,
        floor_db: float | None = None,
    ) -> NDArray[np.float64]: ...

    def output_to_level(
        self,
        observation: NumericValue,
        *,
        floor_db: float | None = None,
    ) -> float | NDArray[np.float64]:
        """測定済み出力線形量をoutput definitionのdB levelへ変換する。

        Args:
            observation: output definitionが要求する線形量。scalarまたは任意shape配列。
                conjpair definitionだけ複素係数を受け、その他は非負実数を受ける。
            floor_db: 0観測値を含む表示・保存時の明示的なlevel下限。未指定なら`-inf`。

        Returns:
            output definitionに従うdB level。shapeは入力と同じ、単位はoutput_level_label。

        Raises:
            ValueError: 線形量が負、非有限、またはfloorが非有限の場合。

        境界条件:
            floorは信号処理値を変更せず、対数写像時の表示契約としてだけ適用する。
        """

        definition = self.output_definition
        raw_values = np.asarray(observation)
        require(
            bool(np.all(np.isfinite(raw_values))), "observation must contain only finite values."
        )
        if definition._linear_quantity is _LinearQuantity.CONJPAIR_POWER:
            # zは正周波数の正規化済み係数。reference係数A_ref/sqrt(2)との振幅比を
            # 20log10で写像すれば、10log10(2|z|²/A_ref²)と同じ式になる。
            linear_ratio = (
                np.abs(raw_values.astype(np.complex128)) / definition._reference_linear_value
            )
            multiplier = 20.0
        else:
            require(
                not np.iscomplexobj(raw_values),
                "only conjpair_power accepts complex observations.",
            )
            values = np.asarray(raw_values, dtype=np.float64)
            require(bool(np.all(values >= 0.0)), "observation must be non-negative.")
            if definition._linear_quantity in {_LinearQuantity.RMS, _LinearQuantity.ASD}:
                linear_ratio = values / definition._reference_linear_value
                multiplier = 20.0
            else:
                linear_ratio = values / definition._reference_mean_square
                multiplier = 10.0

        ratio = np.asarray(linear_ratio, dtype=np.float64)
        if floor_db is None:
            with np.errstate(divide="ignore"):
                levels = multiplier * np.log10(ratio)
        else:
            floor = float(floor_db)
            require(bool(np.isfinite(floor)), "floor_db must be finite when specified.")
            floor_ratio = 10.0 ** (floor / multiplier)
            levels = multiplier * np.log10(np.maximum(ratio, floor_ratio))
        return _return_scalar_or_array(observation, np.asarray(levels, dtype=np.float64))

    @overload
    def output_rms_to_level(
        self,
        rms_amplitude: RealScalar,
        *,
        floor_db: float | None = None,
    ) -> float: ...

    @overload
    def output_rms_to_level(
        self,
        rms_amplitude: NDArray[Any],
        *,
        floor_db: float | None = None,
    ) -> NDArray[np.float64]: ...

    def output_rms_to_level(
        self,
        rms_amplitude: RealValue,
        *,
        floor_db: float | None = None,
    ) -> float | NDArray[np.float64]:
        """測定済みRMS amplitudeをRMS output definitionのdBへ変換する。

        Args:
            rms_amplitude: 非負RMS amplitude。scalarまたは任意shapeの実数配列。
            floor_db: 0 RMSを含む表示・保存時の明示的なlevel下限。未指定なら`-inf`。

        Returns:
            RMS level。shapeは入力と同じ、単位はoutput_level_label。

        Raises:
            ValueError: output definitionがRMSでない、または値が不正な場合。

        境界条件:
            波形からRMSを測定する処理は含まず、測定済み線形RMSだけを受け付ける。
        """

        require(
            self.output_definition._linear_quantity is _LinearQuantity.RMS,
            "output_rms_to_level requires level_20log10_rms output definition.",
        )
        return self.output_to_level(rms_amplitude, floor_db=floor_db)


__all__ = [
    "LevelConverter",
    "LevelDefinition",
    "level_10log10_conjpair_power",
    "level_10log10_onesided_psd",
    "level_10log10_power",
    "level_10log10_twosided_psd",
    "level_20log10_onesided_asd",
    "level_20log10_rms",
    "level_20log10_twosided_asd",
]
