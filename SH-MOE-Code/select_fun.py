# 函数调用（特征选择）
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Tuple, Optional
from Input_fun import time_series_cv_indices


def select_features_stability(
    X_df: pd.DataFrame,
    y: np.ndarray,
    k_splits: int = 3,  # k_splits 决定了每个随机种子下的“CV 次数”，即统计特征重要性的循环次数。
    seeds = (0,1,2,3,4),  # 每个种子对应一次新的模型初始化和 permutation importance 过程。
    topN: int = 10,  # 在每次 permutation importance 的结果里，只取前 N 个特征来记一次“入选”。
    freq_thr: float = 0.6,  # 频率阈值，表示一个特征要被保留下来，必须在所有循环中有 ≥60% 的次数进入前 N。
) -> Tuple[List[str], Dict[str, float]]:
    """
    返回：(稳定特征列表, 特征平均重要度字典，用于后续去冗余tie-break)
    """
    X = X_df.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    cols = list(X_df.columns)
    cv_indices = time_series_cv_indices(len(X), n_splits=max(2, k_splits))

    picked = Counter()
    imp_sum = defaultdict(float)
    imp_cnt = Counter()

    # 两类稳健模型（固定温和参数，避免大网格与过拟合）
    models = [
        ("RF",  RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1, max_depth=None)),
        ("LGB", LGBMRegressor(n_estimators=400, learning_rate=0.1, num_leaves=31, random_state=0,
                              n_jobs=1, verbose=-1)),
    ]

    # 对于每个随机种子与每一折：拟合->在验证集上做 permutation importance -> 取前TopN
    for sd in seeds:
        for _, base in models:
            for tr_idx, va_idx in cv_indices:
                est = clone(base).set_params(random_state=sd)
                est.fit(X[tr_idx], y[tr_idx])  # 这里的 y 已是 numpy
                pi = permutation_importance(est, X[va_idx], y[va_idx], n_repeats=8, random_state=sd, n_jobs=1)
                order = np.argsort(pi.importances_mean)[::-1]
                k = min(topN, len(cols))
                top_idx = order[:k]
                for j in top_idx:
                    picked[cols[j]] += 1
                # 同时累计均值重要度（用于后续冗余tie-break）
                for j, v in enumerate(pi.importances_mean):
                    imp_sum[cols[j]] += float(v)
                    imp_cnt[cols[j]] += 1

    total = len(seeds) * len(models) * len(cv_indices)
    freq = {c: picked[c] / total for c in cols}
    imp_mean = {c: (imp_sum[c] / max(imp_cnt[c], 1)) for c in cols}

    # 选频率≥阈值者；若一个都没有，则回退“按 imp_mean 排前 max(5, 50% 特征数)”
    sel = [c for c in cols if freq.get(c, 0.0) >= freq_thr]
    if not sel:
        k = max(5, int(0.5 * len(cols)))
        sel = [c for c, _ in sorted(imp_mean.items(), key=lambda kv: kv[1], reverse=True)[:k]]

    return sel, imp_mean


# ======== (b) 冗余去除：|Spearman| 与 VIF（mRMR式保留） ========
def _vif_scores(X_df: pd.DataFrame) -> Dict[str, float]:
    """用线性回归R^2来估算VIF，避免引入statsmodels依赖。"""
    cols = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)
    vifs = {}
    for j, col in enumerate(cols):
        y = X[:, j]
        X_others = np.delete(X, j, axis=1)
        if X_others.shape[1] == 0:
            vifs[col] = 1.0
            continue
        # 简单线性回归求 R^2
        lr = LinearRegression()
        lr.fit(X_others, y)
        r2 = float(lr.score(X_others, y))
        vifs[col] = float(1.0 / max(1.0 - r2, 1e-6))
    return vifs

def remove_redundancy_spearman_vif(
    X_df: pd.DataFrame,
    y: np.ndarray,
    imp_mean: Optional[Dict[str, float]] = None,
    corr_thr: float = 0.8,
    vif_thr: float = 10.0,
) -> List[str]:
    """
    迭代式去冗余：|Spearman|>corr_thr 或 VIF>vif_thr 时，保留“与目标|ρ|更高者”，
    若再tie，用 imp_mean 更高者。
    """
    imp_mean = imp_mean or {}
    features = list(X_df.columns)

    def _target_corr(col):
        # 与目标的 |Spearman| 相关
        return abs(pd.Series(X_df[col]).corr(pd.Series(y), method="spearman"))

    changed = True
    while changed:
        changed = False
        if len(features) <= 1:
            break

        X_now = X_df[features]
        # 1) 相关性冲突
        corr = X_now.corr(method="spearman").abs().fillna(0.0)
        np.fill_diagonal(corr.values, 0.0)
        # 找到最严重的一对
        max_pair = None
        max_r = 0.0
        for i, ci in enumerate(features):
            for j, cj in enumerate(features):
                if j <= i:
                    continue
                r = float(corr.loc[ci, cj])
                if r > max_r:
                    max_r = r
                    max_pair = (ci, cj)
        if max_r > corr_thr and max_pair:
            a, b = max_pair
            # 选保留者
            ta = _target_corr(a); tb = _target_corr(b)
            if abs(ta - tb) > 1e-12:
                drop = b if ta > tb else a
            else:
                ia = imp_mean.get(a, 0.0); ib = imp_mean.get(b, 0.0)
                drop = b if ia >= ib else a
            features.remove(drop)
            changed = True
            continue

        # 2) VIF 冲突
        vifs = _vif_scores(X_now)
        worst, worst_v = max(vifs.items(), key=lambda kv: kv[1])
        if worst_v > vif_thr:
            # 选择要保留谁：
            # 与目标相关更高/稳定性更高者保留 -> 把“更弱”的删除
            # 找到与worst强相关的若干个候选进行对比；简单起见直接删worst（保守）
            # 如想更严谨，可在此与其最相关的一个比较后删除更弱者
            features.remove(worst)
            changed = True

    return features