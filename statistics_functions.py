"""statistics_functions.py - Common statistical analysis helpers, with sample tests.

Each function returns a **plain Python dict** with ready-to-serialize (JSON) values so
it can be reused later by any API layer.

Required packages
-----------------
- numpy
- scipy
- scikit-learn (for PCA, FactorAnalysis, KMeans)
- pandas (for sample data generation)

Usage example: モジュールとして import して各関数を利用可能。
また、スクリプトとして直接実行するとサンプルデータで各関数をテストします。

To run tests:
    python statistics_functions.py
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis

__all__ = [
    "t_test",
    "paired_t_test",
    "anova",
    "simple_linear_regression",
    "multiple_linear_regression",
    "principal_component_analysis",
    "factor_analysis",
    "kmeans_clustering",
]

Number = Union[int, float]
ArrayLike = Sequence[Number]


# ---------------------------------------------------------------------------
# 1. t-test ------------------------------------------------------------------
# ---------------------------------------------------------------------------


def t_test(
    group1: ArrayLike,
    group2: ArrayLike,
    *,
    equal_var: bool = False,
    nan_policy: str = "omit",
    group1_name: str = "Group1",
    group2_name: str = "Group2",
) -> Dict[str, float]:
    """2標本対応なしのt検定Independent two-sample *t*-test.

    Parameters
    ----------
    group1, group2 : array-like
        Numeric samples to compare.
    equal_var : bool, default ``False``
        Perform Welch's test (safer) when ``False``; classical Student's *t*
        when ``True``.
    nan_policy : {"propagate", "raise", "omit"}
        How to handle NaNs.
    """
    t_stat, p_val = stats.ttest_ind(
        group1, group2, equal_var=equal_var, nan_policy=nan_policy
    )
    # グラフ生成
    fig = go.Figure()
    fig.add_trace(go.Box(y=group1, name=group1_name))
    fig.add_trace(go.Box(y=group2, name=group2_name))
    fig.update_layout(title="グループ比較", yaxis_title="値")
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "plot_html": plot_html,
    }


def paired_t_test(
    group1: ArrayLike,
    group2: ArrayLike,
    *,
    nan_policy: str = "omit",
    group1_name: str = "Group1",
    group2_name: str = "Group2",
) -> Dict[str, float]:
    """2標本対応ありのt検定（Paired t-test） *t*-test.

    Parameters
    ----------
    group1, group2 : array-like
        Numeric samples to compare.
    equal_var : bool, default ``False``
        Perform Welch's test (safer) when ``False``; classical Student's *t*
        when ``True``.
    nan_policy : {"propagate", "raise", "omit"}
        How to handle NaNs.
    """
    t_stat, p_val = stats.ttest_rel(group1, group2, nan_policy=nan_policy)
    # グラフ生成
    fig = go.Figure()
    fig.add_trace(go.Box(y=group1, name=group1_name))
    fig.add_trace(go.Box(y=group2, name=group2_name))
    fig.update_layout(title="グループ比較", yaxis_title="値")
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "plot_html": plot_html,
    }


# ---------------------------------------------------------------------------
# 2. One-way ANOVA -----------------------------------------------------------
# ---------------------------------------------------------------------------


def anova(*groups: ArrayLike, nan_policy: str = "omit") -> Dict[str, float]:
    """One-way ANOVA (F-test for >1 groups)."""
    f_stat, p_val = stats.f_oneway(*groups, nan_policy=nan_policy)  # type: ignore[arg-type]
    return {"F_statistic": float(f_stat), "p_value": float(p_val)}


# ---------------------------------------------------------------------------
# 3. Simple linear regression -------------------------------------------------
# ---------------------------------------------------------------------------


def simple_linear_regression(
    x: ArrayLike, y: ArrayLike, *, x_name: str = "x", y_name: str = "y"
) -> Dict[str, float]:
    """Fit *y = intercept + slope·x* using SciPy's `linregress` and generate plot."""
    slope, intercept, r_value, p_val, std_err = stats.linregress(x, y)
    # グラフ生成
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="データ"))
    xs = np.array(x)
    ys = intercept + slope * xs
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="回帰直線"))
    fig.update_layout(
        title=f"{x_name} vs {y_name} の単回帰分析",
        xaxis_title=x_name,
        yaxis_title=y_name,
        height=500,
        margin=dict(t=40, l=50, r=10, b=50),
    )
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r_value": float(r_value),
        "r_squared": float(r_value**2),
        "p_value": float(p_val),
        "std_err": float(std_err),
        "plot_html": plot_html,
    }


# ---------------------------------------------------------------------------
# 4. Multiple linear regression ---------------------------------------------
# ---------------------------------------------------------------------------


def multiple_linear_regression(
    X: Sequence[Sequence[Number]], y: ArrayLike
) -> Dict[str, Union[float, List[float]]]:
    """Least-squares multiple regression.

    *X* shape = (n_samples, n_features)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2-D array-like.")
    if y.ndim != 1:
        raise ValueError("y must be 1-D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same length.")

    # Design matrix with intercept term
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    coefs, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    intercept = coefs[0]
    betas = coefs[1:]

    # R²
    y_pred = X_design @ coefs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return {
        "intercept": float(intercept),
        "coefficients": betas.tolist(),
        "r_squared": float(r_squared),
    }


# ---------------------------------------------------------------------------
# 5. Principal Component Analysis -------------------------------------------
# ---------------------------------------------------------------------------


def principal_component_analysis(
    data: Sequence[Sequence[Number]],
    n_components: int | None = None,
    *,
    scale: bool = True,
) -> Dict[str, List]:
    """Run PCA and return transformed components & metadata."""
    X = np.asarray(data, dtype=float)
    if scale:
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    return {
        "components": components.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "loadings": pca.components_.tolist(),
        "singular_values": pca.singular_values_.tolist(),
        "mean": pca.mean_.tolist(),
    }


# ---------------------------------------------------------------------------
# 6. Factor Analysis ---------------------------------------------------------
# ---------------------------------------------------------------------------


def factor_analysis(
    data: Sequence[Sequence[Number]],
    n_factors: int | None = None,
    *,
    random_state: int | None = 0,
) -> Dict[str, List]:
    """Run maximum-likelihood factor analysis."""
    fa = FactorAnalysis(n_components=n_factors, random_state=random_state)
    factors = fa.fit_transform(data)
    return {
        "factors": factors.tolist(),
        "loadings": fa.components_.tolist(),
        "noise_variance": fa.noise_variance_.tolist(),
    }


# ---------------------------------------------------------------------------
# 7. K-means clustering -------------------------------------------------------
# ---------------------------------------------------------------------------


def kmeans_clustering(
    data: Sequence[Sequence[Number]],
    *,
    n_clusters: int = 3,
    random_state: int | None = 0,
    max_iter: int = 300,
) -> Dict[str, Union[float, List, str]]:
    """Cluster *data* into *n_clusters* using Lloyd–Forgy k-means.

    Returns labels, cluster centres, and plot HTML.
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=max_iter,
    )
    labels = km.fit_predict(data)
    centroids = km.cluster_centers_

    # クラスタリング結果のプロット生成
    fig = go.Figure()

    # データポイントをプロット（クラスタごとに色分け）
    for i in range(n_clusters):
        cluster_data = np.array(data)[labels == i]
        fig.add_trace(
            go.Scatter(
                x=cluster_data[:, 0],
                y=cluster_data[:, 1] if len(data[0]) > 1 else [0] * len(cluster_data),
                mode="markers",
                name=f"Cluster {i+1}",
                marker=dict(size=8),
            )
        )

    # 重心をプロット
    fig.add_trace(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1] if len(data[0]) > 1 else [0] * len(centroids),
            mode="markers",
            name="Centroids",
            marker=dict(symbol="x", size=12, color="black", line=dict(width=2)),
        )
    )

    fig.update_layout(
        title=f"K-means Clustering (k={n_clusters})",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2" if len(data[0]) > 1 else "",
        height=500,
        margin=dict(t=40, l=50, r=10, b=50),
    )
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    return {
        "labels": labels.tolist(),
        "centroids": centroids.tolist(),
        "inertia": float(km.inertia_),
        "plot_html": plot_html,
    }


# ---------------------------------------------------------------------------
# 8. サンプルデータ生成および関数テスト用のメインブロック
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    np.random.seed(0)
    # サンプルデータフレーム生成
    data = pd.DataFrame(
        {
            # t検定 / ANOVA 用のグループ
            "group1": np.random.normal(loc=5.0, scale=2.0, size=100),
            "group2": np.random.normal(loc=5.5, scale=2.0, size=100),
            "group3": np.random.normal(
                loc=6.0, scale=2.5, size=100
            ),  # ANOVA 用に3つ目のグループ
            # 単回帰・重回帰分析用の説明変数・従属変数
            "independent_var1": np.random.rand(100) * 5,
            "independent_var2": np.random.rand(100) * 3,
            "dependent_var": None,
            # 主成分分析用の変数（3次極）
            "pca_var1": np.random.rand(100),
            "pca_var2": np.random.rand(100) * 2,
            "pca_var3": np.random.rand(100) * 0.5,
            # 因子分析用の変数（3次元）
            "factor_var1": np.random.rand(100),
            "factor_var2": np.random.rand(100) + 0.5 * np.random.rand(100),
            "factor_var3": np.random.rand(100) - 0.3 * np.random.rand(100),
            # クラスタリング用の変数（2次元）
            "cluster_var1": np.random.rand(100),
            "cluster_var2": np.random.rand(100),
        }
    )

    # dependent_var = 2*X1 + 3*X2 + ノイズ
    data["dependent_var"] = (
        2.0 * data["independent_var1"]
        + 3.0 * data["independent_var2"]
        + np.random.normal(loc=0.0, scale=2.0, size=100)
    )

    # 1. t検定
    tt_res = t_test(data["group1"], data["group2"], equal_var=False)
    print("=== t-test (Welch) between group1 & group2 ===")
    print(f"t-statistic: {tt_res['t_statistic']:.4f}")
    print(f"p-value    : {tt_res['p_value']:.4f}\n")

    # 2. ANOVA
    anova_res = anova(data["group1"], data["group2"], data["group3"])
    print("=== One-way ANOVA among group1, group2, group3 ===")
    print(f"F-statistic: {anova_res['F_statistic']:.4f}")
    print(f"p-value    : {anova_res['p_value']:.4f}\n")

    # 3. 単回帰分析
    slr_res = simple_linear_regression(data["independent_var1"], data["dependent_var"])
    print("=== Simple Linear Regression ===")
    print(f"intercept: {slr_res['intercept']:.4f}")
    print(f"slope    : {slr_res['slope']:.4f}")
    print(f"R-squared: {slr_res['r_squared']:.4f}")
    print(f"p-value  : {slr_res['p_value']:.4f}\n")

    # 4. 重回帰分析
    X_multi = data[["independent_var1", "independent_var2"]].values.tolist()
    y_multi = data["dependent_var"].values
    mlr_res = multiple_linear_regression(X_multi, y_multi)
    print("=== Multiple Linear Regression ===")
    print(f"intercept   : {mlr_res['intercept']:.4f}")
    print(f"coefficients: {mlr_res['coefficients']}")
    print(f"R-squared   : {mlr_res['r_squared']:.4f}\n")

    # 5. PCA
    pca_data = data[["pca_var1", "pca_var2", "pca_var3"]].values.tolist()
    pca_res = principal_component_analysis(pca_data, n_components=2, scale=True)
    print("=== PCA (2 components) ===")
    print(f"Explained Variance Ratio: {pca_res['explained_variance_ratio']}")
    print(f"First 2 Loading Vectors : {pca_res['loadings'][:2]}\n")

    # 6. 因子分析
    fa_data = data[["factor_var1", "factor_var2", "factor_var3"]].values.tolist()
    fa_res = factor_analysis(fa_data, n_factors=2, random_state=42)
    print("=== Factor Analysis (2 factors) ===")
    print(f"Noise variances: {fa_res['noise_variance']}")
    print(f"Loading matrix : {fa_res['loadings']}\n")

    # 7. K-means クラスタリング
    cluster_data = data[["cluster_var1", "cluster_var2"]].values.tolist()
    km_res = kmeans_clustering(cluster_data, n_clusters=3, random_state=0)
    print("=== K-means Clustering (3 clusters) ===")
    print(f"Inertia   : {km_res['inertia']:.4f}")
