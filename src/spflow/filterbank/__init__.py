"""src/spflow/filterbank パッケージの公開 API をまとめるモジュール。"""

# 解析器・合成器・完全再構成検証器は組み合わせて使うことが前提なので、
# 利用者がサブモジュール名ではなく処理系の概念名で import できるよう再公開する。
from .causal_analytic_frontend import (
    CausalAnalyticFrontend,
    CausalAnalyticFrontendStreamer,
    CausalAnalyticResult,
    design_hilbert_fir,
)
from .formal_complex_pr_stage import FormalBandPacket, FormalComplexPRHalfbandStage
from .formal_nonuniform_streaming import (
    FormalNonuniformTreeStreamingAnalyzer,
    FormalNonuniformTreeStreamingSynthesizer,
    FormalPacketBlock,
)
from .formal_nonuniform_tree import FormalNonuniformAnalysisResult, FormalNonuniformTreeFilterBank
from .nonuniform_streaming import NonuniformPacketBlock, NonuniformTreeStreamingAnalyzer, NonuniformTreeStreamingSynthesizer
from .nonuniform_tree import NonuniformAnalysisResult, NonuniformBandPacket, NonuniformBandSpec, NonuniformTreeFilterBank
from .prdft import DFT_FilterBank, FullDFTFilterBank, PRDFTFilterBank, PolyphaseDFTFilterBank
from .prdft_modulated import (
    DFTModulatedFilterDesigner,
    FiniteLengthPRChecker,
    PRDFTAnalysisBank,
    PRDFTSynthesisBank,
    PrototypePairDesigner,
)
from .prdft_polyphase import (
    PolyphasePRDFTAnalysisBank,
    PolyphasePRDFTSynthesisBank,
    PolyphasePRPairDesigner,
)
from .prototype_bank import (
    PRChecker,
    PolyphaseDecomposition,
    PrototypeAnalysisDFTFilterBank,
    PrototypeFilter,
    PrototypeSynthesisDFTFilterBank,
)

# ワイルドカード import の結果を安定させることで、ノートブックや実験スクリプト側の
# 依存面を明示しつつ、内部補助関数の露出を避ける。
__all__ = [
    'design_hilbert_fir',
    'CausalAnalyticResult',
    'CausalAnalyticFrontend',
    'CausalAnalyticFrontendStreamer',
    'FormalBandPacket',
    'FormalComplexPRHalfbandStage',
    'FormalNonuniformAnalysisResult',
    'FormalNonuniformTreeFilterBank',
    'FormalPacketBlock',
    'FormalNonuniformTreeStreamingAnalyzer',
    'FormalNonuniformTreeStreamingSynthesizer',
    'NonuniformBandSpec',
    'NonuniformBandPacket',
    'NonuniformAnalysisResult',
    'NonuniformTreeFilterBank',
    'NonuniformPacketBlock',
    'NonuniformTreeStreamingAnalyzer',
    'NonuniformTreeStreamingSynthesizer',
    'PRDFTFilterBank',
    'FullDFTFilterBank',
    'PolyphaseDFTFilterBank',
    'PrototypeFilter',
    'PolyphaseDecomposition',
    'PrototypeAnalysisDFTFilterBank',
    'PrototypeSynthesisDFTFilterBank',
    'DFTModulatedFilterDesigner',
    'PRDFTAnalysisBank',
    'PRDFTSynthesisBank',
    'FiniteLengthPRChecker',
    'PrototypePairDesigner',
    'PolyphasePRDFTAnalysisBank',
    'PolyphasePRDFTSynthesisBank',
    'PolyphasePRPairDesigner',
    'PRChecker',
    'DFT_FilterBank',
]
