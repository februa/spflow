# SLC詳細設計書

## 1. 目的

本書は、固定整相後のビーム出力に対して適用する SLC、サイドローブキャンセラの詳細設計を定義する。

ここでは、データ形状、数式の実装、共分散更新、艦首変化を考慮した忘却制御、複数 target 対応、参照ビーム不足判定、クラス責務分離を扱う。

---

## 2. 前提

SLC の入力は固定整相後のビーム出力である。

```text
beam_output: [n_beam, n_sample]
```

SLC は、target beam と reference beam を分けて処理する。

```text
target_beams:    [n_target]
reference_beams: [n_ref]
```

複数 target を許容する。複数 target の周囲 guard はすべて保護領域として扱い、その外側を reference beam とする。

---

## 3. データ形状

### 3.1 入力

```text
beam_output: complex or real ndarray [n_beam, n_sample]
```

### 3.2 target 出力

```text
D = beam_output[target_beams, :]
D: [n_target, n_sample]
```

単一 target の場合は、

```text
d: [n_sample]
```

として扱ってもよい。

### 3.3 reference 出力

```text
U = beam_output[reference_beams, :]
U: [n_ref, n_sample]
```

### 3.4 SLC 係数

単一 target の場合、

```text
w: [n_ref]
```

複数 target の場合、

```text
W: [n_target, n_ref]
```

とする。

### 3.5 出力

単一 target の場合、

```text
y: [n_sample]
```

複数 target の場合、

```text
Y: [n_target, n_sample]
```

必要に応じて、干渉推定信号も出力する。

```text
C: [n_target, n_sample]
```

---

## 4. SLC の基本式

単一 target に対して、target beam 出力を \(d[n]\)、reference beam 群を \(\mathbf{u}[n]\) とする。

\[
\mathbf{u}[n]
=
\begin{bmatrix}
u_1[n] \\
u_2[n] \\
\vdots \\
u_K[n]
\end{bmatrix}
\]

干渉推定信号は、

\[
c[n]
=
\mathbf{w}^H\mathbf{u}[n]
\]

SLC 出力は、

\[
y[n]
=
d[n]-c[n]
=
d[n]-\mathbf{w}^H\mathbf{u}[n]
\]

である。

ブロック実装では、

```text
U: [n_ref, n_sample]
d: [n_sample]
```

に対して、

```python
cancel = np.conj(w) @ U
 y = d - cancel
```

となる。

実装上、`np.conj(w) @ U` は、数式の \(\mathbf{w}^H\mathbf{U}\) に対応する。

---

## 5. ブロック共分散方式

### 5.1 ブロック統計量

1 ブロック内の reference beam 信号を \(\mathbf{U}\)、target beam 信号を \(d\) とする。

参照ビームの相関行列は、

\[
\hat{\mathbf{R}}_{uu}
=
\frac{1}{N}\mathbf{U}\mathbf{U}^H
\]

である。

参照ビームと target beam の相互相関は、

\[
\hat{\mathbf{r}}_{ud}
=
\frac{1}{N}\mathbf{U}d^*
\]

である。

実装は次である。

```python
n_sample = U.shape[1]
R_hat = (U @ U.conj().T) / n_sample
r_hat = (U @ d.conj()) / n_sample
```

形状は次である。

```text
R_hat: [n_ref, n_ref]
r_hat: [n_ref]
```

### 5.2 複数 target の相互相関

複数 target の場合、

```text
D: [n_target, n_sample]
```

である。

target ごとの相互相関をまとめて計算する場合、

\[
\hat{\mathbf{R}}_{ud}
=
\frac{1}{N}\mathbf{D}\mathbf{U}^H
\]

として、

```python
r_hat_all = (D @ U.conj().T) / n_sample
```

と計算できる。

形状は、

```text
r_hat_all: [n_target, n_ref]
```

である。

このとき、target \(i\) の SLC 係数を求めるための右辺は、

```python
r = r_hat_all[i].conj()
```

のように式の定義に合わせて扱うか、単一 target と同じ向きで、

```python
r_hat_i = (U @ d_i.conj()) / n_sample
```

と target ごとに計算してもよい。

実装の混乱を避けるため、正式な方式では target ごとに次の式を使うことを推奨する。

```python
r_hat_i = (U @ d_i.conj()) / n_sample
```

---

## 6. 忘却平均

ブロック統計量は揺れが大きいため、忘却平均を行う。

\[
\mathbf{R}_{uu}[k]
=
\alpha[k]\mathbf{R}_{uu}[k-1]
+
(1-\alpha[k])\hat{\mathbf{R}}_{uu}[k]
\]

\[
\mathbf{r}_{ud}[k]
=
\alpha[k]\mathbf{r}_{ud}[k-1]
+
(1-\alpha[k])\hat{\mathbf{r}}_{ud}[k]
\]

複数 target の場合、\(\mathbf{R}_{uu}\) は reference 共通で 1 つ持ち、\(\mathbf{r}_{ud}\) は target ごとに持つ。

```text
R: [n_ref, n_ref]
r: [n_target, n_ref]
```

---

## 7. SLC 係数の計算

忘却平均後の \(\mathbf{R}_{uu}\) と \(\mathbf{r}_{ud}\) から、SLC 係数を求める。

\[
\mathbf{w}
=
(\mathbf{R}_{uu}+\lambda\mathbf{I})^{-1}\mathbf{r}_{ud}
\]

ここで、\(\lambda\) は対角ロードである。

実装は次である。

```python
A = R + loading * np.eye(n_ref, dtype=R.dtype)
w = np.linalg.solve(A, r)
```

複数 target の場合、\(A\) は共通であり、target ごとに右辺 \(r_i\) が異なる。

```python
W = np.zeros((n_target, n_ref), dtype=beam_output.dtype)
for i in range(n_target):
    W[i] = np.linalg.solve(A, r[i])
```

---

## 8. 干渉推定信号の生成

係数 \(\mathbf{w}\) が求まったら、reference beam 信号から干渉推定信号を生成する。

\[
c[n]
=
\mathbf{w}^H\mathbf{u}[n]
\]

ブロックでは、

```python
cancel = np.conj(w) @ U
```

である。

複数 target の場合、

```python
C = np.conj(W) @ U
```

により、

```text
C: [n_target, n_sample]
```

が得られる。

SLC 出力は、

```python
Y = D - C
```

である。

キャンセル強度 \(\eta\) を持たせる場合は、

```python
Y = D - eta * C
```

とする。

---

## 9. 艦首変化を考慮した忘却制御

### 9.1 目的

相対方位 SLC では、艦首方位が変化すると、同じ外部音源が相対ビーム上で移動する。

そのため、艦首変化が大きい場合は、過去の共分散を強く忘れる必要がある。

### 9.2 時間忘却係数

ブロック長を \(N\)、サンプリング周波数を \(f_s\)、保持したい時間を \(T\) とする。

ブロック時間は、

\[
T_b = \frac{N}{f_s}
\]

である。

通常の時間忘却係数は、

\[
\alpha_t
=
\exp\left(-\frac{T_b}{T}\right)
\]

で定義する。

### 9.3 艦首変化による追加忘却

ブロック \(k\) における艦首方位を \(\psi[k]\) とする。

前回ブロックからの艦首変化量を、

\[
\Delta \psi[k]
=
\mathrm{wrap}(\psi[k]-\psi[k-1])
\]

とする。

艦首変化による忘却係数を、

\[
\alpha_\psi[k]
=
\exp\left(-\frac{|\Delta\psi[k]|}{\psi_0}\right)
\]

とする。

ここで、\(\psi_0\) は、艦首変化に対してどの程度敏感に忘却するかを決める角度スケールである。

最終的な忘却係数は、

\[
\alpha[k]
=
\alpha_t\alpha_\psi[k]
\]

である。

### 9.4 方位センサ正常時と異常時

方位センサが正常な場合は、艦首変化連動忘却を使う。

```python
alpha = alpha_time * alpha_heading
```

方位センサが異常な場合は、艦首変化量を使わない。

```python
alpha = alpha_time
```

これにより、方位センサ故障時に異常なリセットや過剰忘却が起きることを防ぐ。

### 9.5 実装関数例

```python
import numpy as np


def wrap_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def compute_forgetting_factor(
    heading_deg: float,
    prev_heading_deg: float,
    heading_valid: bool,
    block_size: int,
    fs: float,
    memory_time_sec: float,
    heading_scale_deg: float,
    alpha_min: float = 0.0,
    alpha_max: float = 0.9999,
) -> float:
    block_time = block_size / fs
    alpha_time = np.exp(-block_time / memory_time_sec)

    if not heading_valid:
        return float(np.clip(alpha_time, alpha_min, alpha_max))

    delta_heading = abs(wrap_angle_deg(heading_deg - prev_heading_deg))
    alpha_heading = np.exp(-delta_heading / heading_scale_deg)
    alpha = alpha_time * alpha_heading

    return float(np.clip(alpha, alpha_min, alpha_max))
```

---

## 10. guard と reference beam の生成

### 10.1 保護マスク

`target_beams` と `guard` から保護マスクを作る。

```python
import numpy as np


def make_protected_mask(
    n_beam: int,
    target_beams: np.ndarray,
    guard: int,
) -> np.ndarray:
    protected = np.zeros(n_beam, dtype=bool)

    for b in target_beams:
        b0 = max(0, int(b) - guard)
        b1 = min(n_beam, int(b) + guard + 1)
        protected[b0:b1] = True

    return protected
```

`protected=True` のビームは、SLC 参照に使わない。

### 10.2 reference beam

```python
def make_reference_beams(
    n_beam: int,
    target_beams: np.ndarray,
    guard: int,
) -> np.ndarray:
    protected = make_protected_mask(n_beam, target_beams, guard)
    return np.where(~protected)[0]
```

---

## 11. 参照ビーム不足判定

### 11.1 変数定義

片舷 SLC で参照ビーム不足を決める主な変数は次である。

```text
N_b:
  片舷の待ち受けビーム数

N_t:
  target beam 数

G:
  target ごとの guard beam 数

N_sample:
  共分散推定に使うブロックサンプル数

L:
  時間タップ付き SLC を使う場合のタップ数

N_ref_min:
  SLC を有効とする最小参照ビーム数

rho:
  自由度あたりに必要なサンプル数
```

### 11.2 保護領域

保護領域集合は、

\[
\mathcal{P}
=
\bigcup_{i=1}^{N_t}
\{b_i-G, ..., b_i+G\}
\]

である。範囲外は \([0, N_b-1]\) にクリップする。

参照ビーム集合は、

\[
\mathcal{R}
=
\{0, 1, ..., N_b-1\}\setminus\mathcal{P}
\]

である。

参照ビーム数は、

\[
N_{ref}=|\mathcal{R}|
\]

である。

### 11.3 自由度

時間タップなしの場合、自由度は、

\[
D=N_{ref}
\]

である。

時間タップ長 \(L\) の FIR 型 SLC を使う場合、

\[
D=N_{ref}L
\]

である。

### 11.4 有効条件

SLC を有効にするための条件は次である。

\[
N_{ref} \ge N_{ref,min}
\]

かつ、

\[
N_{sample} \ge \rho D
\]

である。

この条件を満たさない場合、SLC を弱化または無効化する。

### 11.5 保守的な参照不足条件

target 同士の guard が重ならない最悪ケースでは、保護ビーム数は、

\[
N_{protect}
=
N_t(2G+1)
\]

である。

したがって、参照ビーム数の保守的見積もりは、

\[
N_{ref}
\approx
N_b - N_t(2G+1)
\]

である。

次を満たす場合、参照不足が起きやすい。

\[
N_b - N_t(2G+1) < N_{ref,min}
\]

ただし、実際には guard の重なりや端部クリップがあるため、正確には保護マスクから \(N_{ref}\) を計算する。

---

## 12. SLC 弱化・無効化

SLC が危険または不安定な条件では、キャンセル強度を下げる。

\[
Y = D - \eta C
\]

ここで、

```text
η = 1.0 : 通常SLC
η = 0.0 : SLC無効
```

である。

SLC 無効時は、

\[
Y = D
\]

である。

条件に応じた例を示す。

```text
NORMAL:
  η = 1.0

LIMITED_REFERENCE:
  η = 0.0 ～ 0.5
  または SLC 無効

UNSTABLE_WEIGHT:
  η を低下
  w_norm_max を低下
  loading を増加

DISABLED:
  η = 0.0
```

---

## 13. クラス設計

### 13.1 クラス一覧

SLC 詳細設計では、次のクラスに責務分離する。

```text
BeamGuardSelector
SlcReferenceCapacityChecker
HeadingAwareForgettingController
SlcCovarianceEstimator
BlockLeastSquaresSlcSolver
BeamDomainSLC
SlcState
SlcConfig
```

---

## 14. SlcConfig

SLC の設定値を保持する。

```python
from dataclasses import dataclass


@dataclass
class SlcConfig:
    guard: int
    loading: float
    memory_time_sec: float
    heading_scale_deg: float
    min_ref: int = 4
    sample_per_dof: float = 5.0
    tap_len: int = 1
    eta_normal: float = 1.0
    eta_limited: float = 0.5
    enable_heading_forgetting: bool = True
```

---

## 15. BeamGuardSelector

target beam と guard から保護マスク、reference beam を生成する責務を持つ。

```python
class BeamGuardSelector:
    def __init__(self, n_beam: int, guard: int):
        self.n_beam = n_beam
        self.guard = guard

    def make_protected_mask(self, target_beams: np.ndarray) -> np.ndarray:
        protected = np.zeros(self.n_beam, dtype=bool)
        for b in target_beams:
            b0 = max(0, int(b) - self.guard)
            b1 = min(self.n_beam, int(b) + self.guard + 1)
            protected[b0:b1] = True
        return protected

    def make_reference_beams(self, target_beams: np.ndarray) -> np.ndarray:
        protected = self.make_protected_mask(target_beams)
        return np.where(~protected)[0]
```

---

## 16. SlcReferenceCapacityChecker

SLC を有効化できるだけの参照ビーム数とサンプル数があるかを判定する。

```python
class SlcReferenceCapacityChecker:
    def __init__(
        self,
        min_ref: int,
        sample_per_dof: float,
        tap_len: int = 1,
    ):
        self.min_ref = min_ref
        self.sample_per_dof = sample_per_dof
        self.tap_len = tap_len

    def check(self, n_ref: int, block_size: int) -> dict:
        dof = n_ref * self.tap_len
        has_ref = n_ref >= self.min_ref
        has_samples = block_size >= self.sample_per_dof * dof
        return {
            "n_ref": n_ref,
            "block_size": block_size,
            "tap_len": self.tap_len,
            "dof": dof,
            "has_enough_reference_beams": has_ref,
            "has_enough_samples": has_samples,
            "is_feasible": has_ref and has_samples,
        }
```

---

## 17. HeadingAwareForgettingController

艦首変化を考慮した忘却係数を計算する。

方位センサ異常時は、通常時間忘却のみを返す。

```python
class HeadingAwareForgettingController:
    def __init__(
        self,
        fs: float,
        block_size: int,
        memory_time_sec: float,
        heading_scale_deg: float,
        alpha_min: float = 0.0,
        alpha_max: float = 0.9999,
    ):
        self.fs = fs
        self.block_size = block_size
        self.memory_time_sec = memory_time_sec
        self.heading_scale_deg = heading_scale_deg
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    @staticmethod
    def wrap_angle_deg(angle_deg: float) -> float:
        return (angle_deg + 180.0) % 360.0 - 180.0

    def compute(
        self,
        heading_deg: float,
        prev_heading_deg: float,
        heading_valid: bool,
    ) -> float:
        block_time = self.block_size / self.fs
        alpha_time = np.exp(-block_time / self.memory_time_sec)

        if not heading_valid:
            return float(np.clip(alpha_time, self.alpha_min, self.alpha_max))

        delta = abs(self.wrap_angle_deg(heading_deg - prev_heading_deg))
        alpha_heading = np.exp(-delta / self.heading_scale_deg)
        alpha = alpha_time * alpha_heading

        return float(np.clip(alpha, self.alpha_min, self.alpha_max))
```

---

## 18. SlcCovarianceEstimator

共分散と相互相関を推定し、忘却平均する責務を持つ。

```python
class SlcCovarianceEstimator:
    def __init__(self, n_ref: int, n_target: int, dtype=np.complex128):
        self.R = np.zeros((n_ref, n_ref), dtype=dtype)
        self.r = np.zeros((n_target, n_ref), dtype=dtype)

    def update(self, U: np.ndarray, D: np.ndarray, alpha: float):
        n_ref, n_sample = U.shape
        n_target = D.shape[0]

        R_hat = (U @ U.conj().T) / n_sample

        r_hat = np.zeros((n_target, n_ref), dtype=U.dtype)
        for i in range(n_target):
            r_hat[i] = (U @ D[i].conj()) / n_sample

        self.R = alpha * self.R + (1.0 - alpha) * R_hat
        self.r = alpha * self.r + (1.0 - alpha) * r_hat

        return self.R, self.r

    def reset(self):
        self.R[...] = 0
        self.r[...] = 0
```

---

## 19. BlockLeastSquaresSlcSolver

共分散と相互相関から SLC 係数を求める。

```python
class BlockLeastSquaresSlcSolver:
    def __init__(self, loading: float):
        self.loading = loading

    def solve(self, R: np.ndarray, r: np.ndarray) -> np.ndarray:
        n_ref = R.shape[0]
        n_target = r.shape[0]

        A = R + self.loading * np.eye(n_ref, dtype=R.dtype)

        W = np.zeros((n_target, n_ref), dtype=R.dtype)
        for i in range(n_target):
            W[i] = np.linalg.solve(A, r[i])

        return W
```

---

## 20. BeamDomainSLC

SLC 全体を統括するクラスである。

責務は次である。

```text
1. target beam から reference beam を決定する。
2. 参照ビーム不足を判定する。
3. 艦首変化に応じた忘却係数を決定する。
4. 共分散と相互相関を更新する。
5. SLC 係数を計算する。
6. 干渉推定信号を作る。
7. SLC 出力を返す。
8. 条件が悪い場合は SLC を弱化または無効化する。
```

概念コードを示す。

```python
class BeamDomainSLC:
    def __init__(
        self,
        n_beam: int,
        fs: float,
        block_size: int,
        config: SlcConfig,
    ):
        self.n_beam = n_beam
        self.fs = fs
        self.block_size = block_size
        self.config = config

        self.guard_selector = BeamGuardSelector(n_beam, config.guard)
        self.capacity_checker = SlcReferenceCapacityChecker(
            min_ref=config.min_ref,
            sample_per_dof=config.sample_per_dof,
            tap_len=config.tap_len,
        )
        self.forgetting_controller = HeadingAwareForgettingController(
            fs=fs,
            block_size=block_size,
            memory_time_sec=config.memory_time_sec,
            heading_scale_deg=config.heading_scale_deg,
        )
        self.solver = BlockLeastSquaresSlcSolver(config.loading)

        self.estimator = None
        self.prev_heading_deg = None

    def process(
        self,
        beam_output: np.ndarray,
        target_beams: np.ndarray,
        heading_deg: float | None = None,
        heading_valid: bool = False,
    ):
        n_beam, n_sample = beam_output.shape
        assert n_beam == self.n_beam

        reference_beams = self.guard_selector.make_reference_beams(target_beams)
        n_ref = len(reference_beams)
        n_target = len(target_beams)

        capacity = self.capacity_checker.check(n_ref=n_ref, block_size=n_sample)

        D = beam_output[target_beams, :]

        if not capacity["is_feasible"]:
            # 安全側: 固定整相出力をそのまま通す
            return {
                "Y": D.copy(),
                "C": np.zeros_like(D),
                "W": None,
                "reference_beams": reference_beams,
                "capacity": capacity,
                "mode": "DISABLED",
            }

        U = beam_output[reference_beams, :]

        if self.estimator is None or self.estimator.R.shape[0] != n_ref:
            self.estimator = SlcCovarianceEstimator(
                n_ref=n_ref,
                n_target=n_target,
                dtype=beam_output.dtype,
            )

        if self.prev_heading_deg is None or heading_deg is None:
            alpha = np.exp(-(n_sample / self.fs) / self.config.memory_time_sec)
        else:
            alpha = self.forgetting_controller.compute(
                heading_deg=heading_deg,
                prev_heading_deg=self.prev_heading_deg,
                heading_valid=heading_valid,
            )

        if heading_deg is not None:
            self.prev_heading_deg = heading_deg

        R, r = self.estimator.update(U=U, D=D, alpha=alpha)
        W = self.solver.solve(R=R, r=r)

        C = np.conj(W) @ U
        eta = self.config.eta_normal
        Y = D - eta * C

        mode = "NORMAL" if heading_valid else "DEGRADED_HEADING"

        return {
            "Y": Y,
            "C": C,
            "W": W,
            "reference_beams": reference_beams,
            "capacity": capacity,
            "alpha": alpha,
            "mode": mode,
        }
```

---

## 21. 状態とフォールバック

SLC の動作状態は以下とする。

```text
NORMAL:
  heading_valid = True
  reference capacity OK
  heading-aware forgetting 有効
  η = eta_normal

DEGRADED_HEADING:
  heading_valid = False
  reference capacity OK
  通常時間忘却のみ
  η = eta_normal

LIMITED_REFERENCE:
  reference capacity が境界的
  η = eta_limited
  または loading 増加

DISABLED:
  reference capacity 不足
  Y = D
```

正式な方式では、`LIMITED_REFERENCE` を省略せず、条件を満たさない場合の `DISABLED` と参照制限時の `LIMITED_REFERENCE` を区別する。

---

## 22. 片舷 SLC の設計チェック

片舷だけで SLC を行う場合、待ち受けビーム数が限られるため、参照ビーム不足が起きやすい。

設計時には、以下を必ず評価する。

```text
n_beam_side
n_target
guard
n_protected
n_ref
tap_len
block_size
dof = n_ref × tap_len
block_size / dof
```

チェック関数例を示す。

```python
def check_slc_reference_capacity(
    n_beam: int,
    target_beams: np.ndarray,
    guard: int,
    block_size: int,
    tap_len: int = 1,
    min_ref: int = 4,
    sample_per_dof: float = 5.0,
):
    protected = make_protected_mask(n_beam, target_beams, guard)
    reference_beams = np.where(~protected)[0]
    n_ref = len(reference_beams)
    dof = n_ref * tap_len

    has_ref = n_ref >= min_ref
    has_samples = block_size >= sample_per_dof * dof

    return {
        "n_beam": n_beam,
        "n_target": len(target_beams),
        "guard": guard,
        "n_protected": int(protected.sum()),
        "n_ref": int(n_ref),
        "tap_len": tap_len,
        "dof": int(dof),
        "block_size": block_size,
        "min_ref": min_ref,
        "sample_per_dof": sample_per_dof,
        "has_enough_reference_beams": bool(has_ref),
        "has_enough_samples": bool(has_samples),
        "is_feasible": bool(has_ref and has_samples),
        "reference_beams": reference_beams,
        "protected_mask": protected,
    }
```

---

## 23. 基準パラメータ案

方式検証では次の基準値から評価する。

```text
guard:
  ビームパターンから決定
  評価開始時は 2 ～ 4 beam 程度から sweep する

min_ref:
  4 または 8

sample_per_dof:
  5 ～ 10

tap_len:
  1

loading:
  参照信号平均パワーに対する相対値で設定
  基準値は 1e-3 ～ 1e-2 相当

memory_time_sec:
  1 ～ 10 秒程度から検証

heading_scale_deg:
  guard 幅または数ビーム分の方位変化を基準に設定

eta_normal:
  1.0

eta_limited:
  0.3 ～ 0.5
```

---

## 24. 評価項目

SLC の検証では、以下を確認する。

```text
1. SLC前後の target beam パワー
2. SLC前後の guard外干渉抑圧量
3. target信号の self-nulling の有無
4. SLC係数 W のノルム
5. 共分散行列 R の条件数
6. 参照ビーム数 n_ref
7. block_size / dof
8. 艦首変化時の alpha の変化
9. 方位センサ異常時に通常忘却へ戻ること
10. 参照不足時に固定整相出力へフォールバックすること
```

### 24.1 評価ロールの分離

SLC の評価は、次の 2 つのロールを分けて記録する。

```text
local_leakage_canceller:
  target beam に混入した特定方位・特定周波数の interferer 成分を下げる。
  reduction_at_marker_db、raw_interferer_reduction_db、target_power_delta_db、fallback の有無を見る。

BL_sidelobe_reducer:
  固定整相後 BL 全体の sidelobe envelope を下げる。
  first_sidelobe_reduction_db、guard_outside_peak_delta_db、max_guard_outside_worsening_db、
  percentile / integrated sidelobe、white_noise_gain_db を見る。
```

`local_leakage_canceller` として有効でも、`guard_outside_peak_delta_db` や `first_sidelobe_reduction_db` が悪化する場合は、BL 全体の sidelobe 低減方式とは判定しない。悪化時は safety gate により固定整相出力へ戻す。

---

## 25. 詳細設計まとめ

本詳細設計では、SLC を以下の責務に分離する。

```text
BeamGuardSelector:
  target と guard から reference beam を決定する。

SlcReferenceCapacityChecker:
  参照ビーム数とサンプル数から SLC 有効可否を判定する。

HeadingAwareForgettingController:
  艦首変化と方位センサ状態から忘却係数を決める。

SlcCovarianceEstimator:
  R_uu と r_ud を推定し、忘却平均する。

BlockLeastSquaresSlcSolver:
  対角ロード付き連立方程式から W を求める。

BeamDomainSLC:
  上記を統合し、Y = D - ηC を出力する。
```

安全側設計として、条件が悪い場合は以下を行う。

```text
方位センサ異常:
  艦首変化連動忘却を無効化し、通常時間忘却で更新する。

参照不足:
  guard を狭めず、SLC を弱化または無効化する。

SLC 無効時:
  固定整相出力をそのまま通す。
```
