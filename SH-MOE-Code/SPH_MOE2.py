# MOE  多专家决策+专家筛选(专家筛选参数调整)+MLP门控
# 对代码进行了查漏补缺，增加输出

# 每站点 baseline_predictions_*.xlsx 的 test_predictions sheet：新增
# gate_w_*（每个专家的权重时间序列）、
# gate_in_*（门控输入经变换后的特征）、
# gate_choice（硬判赢家索引）、
# gate_entropy（权重熵）；
# 读者可以直观看到门控在何时偏向哪个专家，以及在枯/丰水、人类影响显著时的权重变化。

# ========================
# 测试集只做一次独立评估，不再参与模型选择。
# （1）β_entropy 降到0.001；（2）损失函数还是Hmean（NSE + logNSE）；（3）去除强正则 + 增长训练周期
# （5）直接将专家数量变为5个，选择效果最好的前5个；

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
import torch, torch.nn as nn, torch.nn.functional as F
# ======== (a) 稳定性筛：K 折 × 多种子 × RF/LGBM 的 permutation importance ========
from lightgbm import LGBMRegressor
from Input_fun import load_excel_dataset
from Input_fun import time_series_cv_indices
from Input_fun import select_features_none
from Input_fun import nse
from Input_fun import log_nse
from Input_fun import _fhv_flv_core
from Input_fun import _hmean_pos
from Input_fun import nse_lognse_hmean
from Input_fun import ensure_dir
from select_fun import select_features_stability
from select_fun import remove_redundancy_spearman_vif
from Input_fun1 import fit_fdc_and_scalers, apply_gate_transforms
from dl_models import make_lstm_regressor, make_tcn_regressor
import time


warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------- Config Structures ----------------------
# —— section_id 的固定顺序（用于最终汇总排序）——
SECTION_ORDER = [
    "Sanliang",    "Zhuozishan",
]
@dataclass
class StationRule:
    first_test_year: int

@dataclass
class RunConfig:
    excel_path: str
    output_dir: str
    station_rules: Dict[str, StationRule]
    random_state: int = 42
    cv_splits: int = 3  # time-aware CV on training period
    n_jobs: int = -1

def greedy_select_experts(oof_by_alg: dict, y_tr, evaluate_fn,
                          max_k: int = 5,
                          corr_thr: float = 0.95,  # ★ 0.90 → 0.95
                          min_k: int = 3,  # ★ 新增：至少保留3个
                          allow_correlated: int = 0,  # 如需再开放“相关但有用”的1个名额，可设1
                          min_gain: float = 0.01):
    """
    oof_by_alg：每个算法在训练期的 OOF 预测；evaluate_fn：打分函数，函数内部会算出 NSE、logNSE，再取 Hmean(NSE, logNSE)
    基于训练期 OOF 的小贪心：
      1) 先挑 Hmean(NSE, logNSE) 最好的单模为起点；
      2) 候选与“当前等权集成”Pearson 相关 < corr_thr 才考虑（除非还没达到 min_k，则先放宽相关性限制）；
      3) 加入后带来的 Hmean 增益 gain >= gate，其中 gate = max(min_gain, -0.1)；  # ★ 下限 -0.1
      4) 最少选 min_k；最多选 max_k；若无法满足 min_k，则按“增益最大但仍 >= -0.1”的顺序补齐。
    返回：selected_algs（专家名列表，按加入顺序）
    """

    def _hmean_nse_lognse(y_true, y_pred):
        m = evaluate_fn(y_true, y_pred)
        # 只取两项做Hmean（与现有口径一致）
        nse, logn = m["NSE"], m["logNSE"]
        if np.isnan(nse) or np.isnan(logn):
            return -np.inf
        # 正化：若为负，按很小值处理，避免Hmean畸变
        nse = max(nse, 1e-9);
        logn = max(logn, 1e-9)
        return 2.0 * nse * logn / (nse + logn)

    def _pearson(a, b):
        a = np.asarray(a, float);
        b = np.asarray(b, float)
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() < 3: return 1.0  # 数据太少，视为高相关以谨慎对待
        return np.corrcoef(a[mask], b[mask])[0, 1]

    algs = list(oof_by_alg.keys())
    # 起点：单模最优。对每个算法的 OOF 与 y_tr 计算 Hmean，得到“单模得分”。
    scores = {a: _hmean_nse_lognse(y_tr, oof_by_alg[a]) for a in algs}
    best_first = max(scores.items(), key=lambda kv: kv[1])[0]
    selected = [best_first]
    ens_pred = np.array(oof_by_alg[best_first], dtype=float)
    base_score = scores[best_first]
    remain = [a for a in algs if a != best_first]

    GATE = max(min_gain, -0.1)  # ★ 增益阈值下限

    # 贪心迭代
    while len(selected) < max_k and remain:
        pick, best_gain, best_pred = None, -np.inf, None
        for a in remain:
            # 相关性门槛：若尚未达到 min_k，则暂不强制；达到后再严格执行
            if len(selected) >= min_k and _pearson(oof_by_alg[a], ens_pred) >= corr_thr:
                continue
            # 等权加入
            cand_pred = (ens_pred * len(selected) + oof_by_alg[a]) / (len(selected) + 1)
            cand_score = _hmean_nse_lognse(y_tr, cand_pred)
            gain = cand_score - base_score
            if gain > best_gain:
                pick, best_gain, best_pred = a, gain, cand_pred

        # 接受条件：gain >= GATE（下限 -0.1）
        if pick is None or best_gain < GATE:
            break

        selected.append(pick)
        ens_pred = best_pred
        base_score += best_gain
        remain.remove(pick)

    # === 改进版兜底：若贪心选出的专家少于 min_k，直接选性能 Top-3 ===
    if len(selected) < min_k:
        # 按单模性能（scores）从高到低排序
        top_sorted = sorted(algs, key=lambda a: scores.get(a, -np.inf), reverse=True)
        selected = top_sorted[:min_k]
        print(f"[Greedy] Too few experts from greedy ({len(selected)}). "
              f"Using top-{min_k} performers instead: {selected}")

    return selected
# ==== END Greedy Expert Subset ====


# ==== PATCH 3A: 门控 MLP 与训练器 =====
class GateMLP(nn.Module):
    def __init__(self, in_dim: int, K: int, tau: float = 0.7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, K)
        )
        self.tau = tau
    def forward(self, g_feat):
        logits = self.mlp(g_feat)
        return F.softmax(logits / self.tau, dim=-1)

def train_gate(gate, G_tr, Yhat_tr, y_tr, G_va, Yhat_va, y_va,
               max_epoch=200, lr=1e-3, beta_entropy=0.001, patience=20, device="cpu"):
    gate.to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)

    def _loss(G, Yhat, y):
        Gt = torch.as_tensor(G, dtype=torch.float32, device=device)
        Yt = torch.as_tensor(Yhat, dtype=torch.float32, device=device)
        yt = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        w = gate(Gt)
        ypred = (w * Yt).sum(dim=1, keepdim=True)

        # === NSE 损失（1 - NSE） ===
        num = ((yt - ypred) ** 2).sum()
        denom = ((yt - yt.mean()) ** 2).sum().clamp_min(1e-6)
        nse_loss = num / denom  # 越小越好（即 1 - NSE）

        # === 熵正则 ===
        entropy = -(w * (w.clamp_min(1e-8).log())).sum(dim=1).mean()
        loss_total = nse_loss + beta_entropy * (-entropy)
        return loss_total, float(nse_loss.item())

    best, wait, best_state = 1e9, 0, None
    for ep in range(max_epoch):
        gate.train(); loss_tr, _ = _loss(G_tr, Yhat_tr, y_tr)
        opt.zero_grad(); loss_tr.backward(); opt.step()
        gate.eval();
        with torch.no_grad():
            loss_va, _ = _loss(G_va, Yhat_va, y_va)
        if ep % 50 == 0:
            # 打印验证集 NSE 损失 + 当前熵
            with torch.no_grad():
                Gt = torch.as_tensor(G_va, dtype=torch.float32, device=device)
                w = gate(Gt)
                entropy = -(w * (w.clamp_min(1e-8).log())).sum(dim=1).mean()
            print(f"[Gate] epoch {ep:03d}: NSE_loss={loss_va.item():.4f}, entropy={entropy.item():.4f}")

        if loss_va.item() < best - 1e-6:
            best, wait = loss_va.item(), 0
            best_state = {k:v.detach().cpu().clone() for k,v in gate.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if best_state is not None:
        gate.load_state_dict(best_state)
    return gate

# ---------------------- Utilities ----------------------
def make_algorithms(random_state: int):
    """Return model constructors and small grids for CV selection (no BO)."""
    try:
        from pygam import LinearGAM, s
        has_pygam = True
    except Exception:
        has_pygam = False

    class PYGAMRegressor:
        """最小封装的 pygam 回归器（满足 sklearn 接口：fit/predict/get/set_params）。"""

        def __init__(self, lam=1.0, n_splines=20, random_state=None, has_pygam_flag=None):
            self.lam = lam
            self.n_splines = n_splines
            self.random_state = random_state
            # 保存是否安装pygam的标志，供并行克隆/反序列化使用
            self._has_pygam = bool(has_pygam_flag)
            self._model = None

        def get_params(self, deep=True):
            # 让 GridSearchCV 可克隆：必须把自定义参数也返回
            return {
                "lam": self.lam,
                "n_splines": self.n_splines,
                "random_state": self.random_state,
                "has_pygam_flag": self._has_pygam,
            }

        def set_params(self, **params):
            for k, v in params.items():
                if k == "has_pygam_flag":
                    self._has_pygam = bool(v)
                else:
                    setattr(self, k, v)
            return self

        def fit(self, X, y):
            if not self._has_pygam:
                raise ImportError("未安装pygam，无法使用GAM模型。")
            # 运行期再导入，避免在未安装环境提前出错
            from pygam import LinearGAM, s
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)

            # 为每个特征设置一个平滑项 s(i)
            p = X.shape[1]
            terms = s(0, n_splines=self.n_splines)
            for i in range(1, p):
                terms = terms + s(i, n_splines=self.n_splines)
            self._model = LinearGAM(terms, lam=self.lam).fit(X, y)
            return self

        def predict(self, X):
            if self._model is None:
                raise RuntimeError("模型尚未训练")
            X = np.asarray(X, dtype=float)
            return self._model.predict(X)

    algs = {
        "RF": (RandomForestRegressor(random_state=random_state),
            {"n_estimators": [100, 300], "max_depth": [None, 10]}),
        # LightGBM
        'LGBM': (
            LGBMRegressor(random_state=random_state, n_jobs=1, verbose=-1),
            # {}
            {
                'num_leaves': [7, 15, 31],  # 更小的叶子数，匹配小样本
                'max_depth': [-1, 3, 5],  # 与 num_leaves 搭配，限制树深
                "min_child_samples": [5, 10, 20],   # ← 原来 'min_data_in_leaf'
                "min_split_gain": [0.0, 1e-3],      # ← 原来 'min_gain_to_split'
                'learning_rate': [0.05, 0.1, 0.2],  # 小样本可适当升高学习率
                'n_estimators': [200, 400],  # 减少树数，避免过拟合与噪声
                'subsample': [0.8, 1.0],  # 行采样（可留作正则）
                'feature_fraction': [0.8, 1.0],  # 列采样（scikit wrapper 也支持）
            }
        ),


        "GB": (GradientBoostingRegressor(random_state=random_state),
            {"n_estimators": [100, 300], "max_depth": [2, 3]}),
        "AdaB": (AdaBoostRegressor(random_state=random_state),
            {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1, 0.2]}),
        "DT": (DecisionTreeRegressor(random_state=random_state),
            {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 5]}),
        "KNN": (KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 7]}),
        "MLR": (LinearRegression(),{}),  # no hyperparams
    }
    # 仅在安装 pygam 时加入 GAM
    if has_pygam:
        algs["GAM"] = (
            PYGAMRegressor(random_state=random_state, has_pygam_flag=has_pygam),
            {"lam": [0.1, 1.0, 10.0], "n_splines": [10, 20]},
        )
    return algs

# 说明：我们故意只在“各自的验证折”上填充预测，其它折位置为 NaN，这样不会把“用过的真值”当预测导出，严格无泄露。
def make_oof_predictions(est, X_tr_view, y_tr_view, cv_indices_local,
                         base_train_len: int, view_offset: int) -> np.ndarray:
    """
    生成训练期 OOF。无论视图多短（LAG/深度 lookback），最终返回长度 == base_train_len 的向量，
    前段（或未验证到的样本）留 NaN。

    参数
    ----
    est:           拟合好的估计器原型（会 clone）
    X_tr_view:     本算法的训练输入视图（可能比全长短）
    y_tr_view:     本算法的训练目标视图（与 X_tr_view 对齐）
    cv_indices_local: 基于“视图长度”的折法索引
    base_train_len:  训练期全长度 len(df_tr)
    view_offset:     视图相对训练全长的“尾部对齐偏移”，= base_train_len - len(X_tr_view)
    """
    oof_full = np.full(base_train_len, np.nan, dtype=float)

    for tr_idx, va_idx in cv_indices_local:
        est_i = clone(est)
        est_i.fit(X_tr_view.iloc[tr_idx], y_tr_view.iloc[tr_idx])

        pred_va = est_i.predict(X_tr_view.iloc[va_idx])
        pred_va = np.asarray(pred_va).reshape(-1)

        # —— 尾部对齐：有些模型会因 lookback 丢掉验证开头若干步 ——
        k = min(len(pred_va), len(va_idx))
        if k == 0:
            continue
        pred_tail = pred_va[-k:]
        va_tail   = va_idx[-k:]

        # —— 将“视图内索引”映射到“训练全长的全局索引”（尾部对齐偏移）——
        global_idx = (va_tail + view_offset)
        # 容错：防止越界（极端情况下）
        global_idx = global_idx[(global_idx >= 0) & (global_idx < base_train_len)]
        if len(global_idx) == 0:
            continue

        # 若 pred_tail 比 global_idx 还长（理论少见），再裁齐
        if len(pred_tail) > len(global_idx):
            pred_tail = pred_tail[-len(global_idx):]

        oof_full[global_idx] = pred_tail

    return oof_full


# 统一按算法名决定折数：LSTM/TCN 固定 2 折；其他用全局 cv_splits
def cv_for_alg(n_samples: int, alg_name: str, n_splits_global: int):
    use_splits = 2 if alg_name in {"LSTM", "TCN"} else n_splits_global
    return time_series_cv_indices(n_samples, n_splits=use_splits)



def fit_best_model(X_train: pd.DataFrame, y_train: pd.Series,
                   algs, n_splits: int, n_jobs: int,
                   Xtr_lag: Optional[pd.DataFrame]=None,
                   y_tr_lag: Optional[pd.Series]=None):
    """
    Time-aware CV on TRAINING period; 通过NSE选择最佳算法+超参数。
    返回:
    Best_name, best_estimator, best_params, best_cv_score，
    Algo_cv_mean_nse (dict), algo_best_params （dict）
    """
    # Build CV splits
    best_score = -np.inf
    best_name, best_est, best_params = None, None, {}
    algo_cv_mean_nse = {}
    algo_cv_mean_rmse = {}
    algo_cv_mean_mae = {}
    algo_cv_mean_logn = {}
    algo_cv_mean_fhv = {}
    algo_cv_mean_flv = {}
    algo_best_params = {}
    algo_cv_mean_hmean = {}  # ★ 新增：存放CV上Hmean的均值（按best_params评估得到）
    algo_cv_time = {}  # ★ 新增：记录每个算法在CV选型阶段的耗时（秒）

    for name, (est, grid) in algs.items():
        # 选用该算法的训练视图（_LAG 用滞后视图；TCN/LSTM 仍用原视图，因内部自带 lookback）
        use_lag_view = name.endswith("_LAG")
        X_use = Xtr_lag if (use_lag_view and Xtr_lag is not None) else X_train
        y_use = y_tr_lag if (use_lag_view and y_tr_lag is not None) else y_train
        cv_indices_local = cv_for_alg(len(X_use), name, n_splits)
        if len(grid) == 0:
            # 改为：同时累计 logNSE，并用 Hmean 刷新 best
            nse_scores, rmse_scores, mae_scores, logn_scores, fhv_scores, flv_scores, h_scores = [], [], [], [], [], [], []
            # run(cfg) 内，每个站
            print(f"[{name}] start")
            t0 = time.perf_counter()
            for tr_idx, va_idx in cv_indices_local:
                est_i = clone(est)
                est_i.fit(X_use.iloc[tr_idx], y_use.iloc[tr_idx])
                pred_va = est_i.predict(X_use.iloc[va_idx])

                # —— 对齐尾部长度（处理 LSTM/TCN 等会丢前 lookback 个样本的情况）——
                y_true_va = y_use.iloc[va_idx]
                pred_va = np.asarray(pred_va).reshape(-1)  # 防止模型吐出 (n,1)

                if len(pred_va) != len(y_true_va):
                    k = min(len(pred_va), len(y_true_va))
                    pred_va = pred_va[-k:]
                    y_true_va = y_true_va.iloc[-k:]

                m = evaluate_predictions(y_true=y_true_va, y_pred=pred_va)
                nse_scores.append(m["NSE"]);
                rmse_scores.append(m["RMSE"]);
                mae_scores.append(m["MAE"])
                logn_scores.append(m["logNSE"]);
                fhv_scores.append(m["FHV"]);
                flv_scores.append(m["FLV"])
                h_scores.append(_hmean_pos(m["NSE"], m["logNSE"]))

            mean_nse = float(np.nanmean(nse_scores))
            algo_cv_mean_nse[name] = mean_nse
            algo_best_params[name] = {}
            # 新增：其余5项
            algo_cv_mean_rmse[name] = float(np.nanmean(rmse_scores))
            algo_cv_mean_mae[name] = float(np.nanmean(mae_scores))
            algo_cv_mean_logn[name] = float(np.nanmean(logn_scores))
            algo_cv_mean_fhv[name] = float(np.nanmean(fhv_scores))
            algo_cv_mean_flv[name] = float(np.nanmean(flv_scores))

            # ★ 用 Hmean 的均值作为本算法CV得分，用来与全局best比较
            mean_hmean = float(np.nanmean(h_scores))
            score_for_best = mean_hmean
            algo_cv_mean_hmean[name] = mean_hmean  # ★ 新增：无参模型的CV均值Hmean

            # 刷新全局最优（>）
            if score_for_best > best_score:
                best_score = score_for_best
                best_name = name
                best_est = clone(est)
                best_params = {}
            elapsed = time.perf_counter() - t0  # ★ 结束计时
            algo_cv_time[name] = float(elapsed)  # ★ 记录
            print(f"[{name}] done, spent {elapsed:.1f}s")
        else:
            t1 = time.perf_counter()
            print(f"[{name}] start")
            from sklearn.metrics import make_scorer
            gscv = GridSearchCV(
                estimator=est,
                param_grid=grid,
                scoring=make_scorer(nse_lognse_hmean, greater_is_better=True),  # Hmean 作为CV打分
                cv=cv_indices_local,
                n_jobs=n_jobs,
                pre_dispatch='1*n_jobs',
                error_score=np.nan,  # ← 新增
            )
            gscv.fit(X_use, y_use)
            algo_best_params[name] = gscv.best_params_

            # —— 新增：用 best_params 在 cv_indices 上再评估 5 个指标 + NSE（与上面口径一致）
            rmse_scores, mae_scores, logn_scores, fhv_scores, flv_scores, nse_scores, h_scores = [], [], [], [], [], [], []
            for tr_idx, va_idx in cv_indices_local:
                est_i = clone(est).set_params(**gscv.best_params_)
                est_i.fit(X_use.iloc[tr_idx], y_use.iloc[tr_idx])
                pred_va = est_i.predict(X_use.iloc[va_idx])

                # —— 对齐尾部长度（处理 LSTM/TCN 等会丢前 lookback 个样本的情况）——
                y_true_va = y_use.iloc[va_idx]
                pred_va = np.asarray(pred_va).reshape(-1)  # 防止模型吐出 (n,1)

                if len(pred_va) != len(y_true_va):
                    k = min(len(pred_va), len(y_true_va))
                    pred_va = pred_va[-k:]
                    y_true_va = y_true_va.iloc[-k:]

                m = evaluate_predictions(y_true=y_true_va, y_pred=pred_va)
                nse_scores.append(m["NSE"])
                rmse_scores.append(m["RMSE"])
                mae_scores.append(m["MAE"])
                logn_scores.append(m["logNSE"])
                fhv_scores.append(m["FHV"])
                flv_scores.append(m["FLV"])
                h_scores.append(_hmean_pos(m["NSE"], m["logNSE"]))  # ★补：累计每折 Hmean
            algo_cv_mean_nse[name] = float(np.nanmean(nse_scores))  # ★ NSE 均值（修正口径）
            algo_cv_mean_logn[name] = float(np.nanmean(logn_scores))
            algo_cv_mean_rmse[name] = float(np.nanmean(rmse_scores))
            algo_cv_mean_mae[name] = float(np.nanmean(mae_scores))
            algo_cv_mean_fhv[name] = float(np.nanmean(fhv_scores))
            algo_cv_mean_flv[name] = float(np.nanmean(flv_scores))
            algo_cv_mean_hmean[name] = float(np.nanmean(h_scores))  # ★ 新增：Hmean 均值

            # 用CV的Hmean（best_score_）作为该算法的CV选参分数
            score_for_best = float(gscv.best_score_)
            if score_for_best > best_score:
                best_score = score_for_best
                best_name = name
                best_est = gscv.best_estimator_
                best_params = gscv.best_params_
            elapsed = time.perf_counter() - t1  # ★ 结束计时
            algo_cv_time[name] = float(elapsed)  # ★ 记录
            print(f"[{name}] done, spent {elapsed:.1f}s")

    return best_name, best_est, best_params, best_score, \
           algo_cv_mean_nse, algo_best_params, \
           algo_cv_mean_rmse, algo_cv_mean_mae, algo_cv_mean_logn, algo_cv_mean_fhv, algo_cv_mean_flv, \
           algo_cv_mean_hmean  ,algo_cv_time      # ★ 新增返回值

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """评估指标：NSE、RMSE、MAE + 新增 logNSE、FHV、FLV。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if yt.size == 0:
        return {"NSE": np.nan, "RMSE": np.nan, "MAE": np.nan,
                "logNSE": np.nan, "FHV": np.nan, "FLV": np.nan}
    metrics = {
        "NSE": nse(yt, yp),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "logNSE": log_nse(yt, yp),
    }
    FHV, FLV = _fhv_flv_core(yt, yp, top_q=0.02, low_q=0.30)
    metrics["FHV"] = FHV
    metrics["FLV"] = FLV
    return metrics

# ===== 新增：Lag6 视图（只用过往信息；训练删首部 NaN，测试不删） =====
def make_lag6_view(df_tr, df_te):
    """
    仅生成下列输入：
      - measured_flow_lag1..6
      - natural_flow_lag1..6
      - rainfall（当月）
      - natural_flow（当月）
    返回： (Xtr_lag, Xte_lag)  两个DataFrame
    """
    import pandas as pd
    import numpy as np

    need_cols = ["section_id","date","measured_flow","natural_flow","rainfall"]
    # 先拼接，保证测试集滞后能看到训练期历史
    df_tr_ = df_tr[need_cols].copy()
    df_te_ = df_te[need_cols].copy()
    df_all = pd.concat([df_tr_, df_te_], axis=0, ignore_index=True)
    df_all = df_all.sort_values(["section_id","date"])

    # 生成 lag1..6（分站点 groupby().shift(L)）
    for base in ["measured_flow","natural_flow"]:
        for L in (1,2,3,4,5,6):
            df_all[f"{base}_lag{L}"] = df_all.groupby("section_id")[base].shift(L)

    # 切回 train/test
    Xtr = df_all.iloc[:len(df_tr_)].copy()
    Xte = df_all.iloc[len(df_tr_):].copy()

    # 训练：删首部 NaN（确保 lag6 有效）
    lag6_cols = [f"{b}_lag6" for b in ["measured_flow","natural_flow"]]
    Xtr = Xtr.dropna(subset=lag6_cols).reset_index(drop=True)
    Xte = Xte.reset_index(drop=True)

    # 只保留我们要的输入列
    keep = []
    for b in ["measured_flow","natural_flow"]:
        keep += [f"{b}_lag{i}" for i in range(1,7)]
    keep += ["rainfall", "natural_flow"]  # 当月
    return Xtr[keep], Xte[keep]



# ---------------------- Core Pipeline ----------------------
def run(config: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (per_station_summary, all_predictions)
    """
    ensure_dir(config.output_dir)

    df_all = load_excel_dataset(config.excel_path)
    algs = make_algorithms(config.random_state)

    all_preds = []
    per_station_rows = []

    # 放在 run() 里，替换原来的 sorted(...) 那行
    rule_rank = {k: i for i, k in enumerate(config.station_rules.keys())}  # 以 dict 插入顺序为准
    station_ids = df_all['section_id'].unique().tolist()
    station_ids.sort(key=lambda s: rule_rank.get(str(s), 10 ** 9))  # 未出现在规则里的，排在末尾

    for sec in station_ids:
        if str(sec) not in config.station_rules:
            raise ValueError(f"Station {sec} missing from station_rules. Please provide first_test_year.")
        rule = config.station_rules[str(sec)]
        df_sec = df_all[df_all['section_id'] == sec].copy()

        # Split train/test by first_test_year
        years = df_sec['date'].dt.year
        train_mask = years < rule.first_test_year
        test_mask = years >= rule.first_test_year
        df_tr = df_sec.loc[train_mask].copy()
        df_te = df_sec.loc[test_mask].copy()
        y_tr = df_tr['measured_flow'].copy()

        if df_tr.empty or df_te.empty:
            print(f"[WARN] Station {sec}: train or test set is empty. Skipping.")
            continue

        # ==== PATCH 3B: 折内拟合 FDC/HAI 标准化，并生成 gate_* 列 ====
        # 折内拟合（只看训练集）
        meta = fit_fdc_and_scalers(
            q_train=df_tr["measured_flow"].values,
            hai_series_dict={
                col: df_tr[col].values
                for col in ["gate_HAI1_t1_raw", "gate_HAI3_t1_raw"]
                if col in df_tr.columns
            }
        )

        # 生成最终 gate_* 列（训练/测试）
        df_tr, gate_cols = apply_gate_transforms(df_tr, meta, inplace=False)
        df_te, gate_cols = apply_gate_transforms(df_te, meta, inplace=False)

        # 门控输入矩阵
        G_tr = df_tr[gate_cols].astype(float).to_numpy()
        G_te = df_te[gate_cols].astype(float).to_numpy()


        # Prepare features/target (no selection, no lags)
        feat_cols = [
            c for c in df_tr.columns
            if c not in ['date', 'section_id', 'measured_flow', 'target', 'U_total']
               and not c.startswith('gate_')
        ]

        # 打印本次训练所用特征列
        print(f"{sec}[特征检查] 训练使用特征（{len(feat_cols)}）：{feat_cols}")

        # —— (a) 稳定性筛（先粗后细） ——
        sel_stable, imp_mean = select_features_stability(
            df_tr[feat_cols], y_tr, k_splits=config.cv_splits, seeds=(0, 1, 2, 3, 4),
            topN=min(10, max(5, len(feat_cols) // 2)), freq_thr=0.6
        )
        # —— (b) 冗余去除（对入围特征） ——
        final_feats = remove_redundancy_spearman_vif(
            df_tr[sel_stable], y_tr, imp_mean=imp_mean, corr_thr=0.8, vif_thr=10.0
        )
        if len(final_feats) == 0:
            # 兜底：至少保留 3 个稳定特征
            keep = min(3, len(sel_stable)) if len(sel_stable) > 0 else min(3, len(feat_cols))
            final_feats = sel_stable[:keep] if len(sel_stable) > 0 else feat_cols[:keep]
        feat_cols = final_feats
        print(f"{sec}[特征筛选] 最终使用特征（{len(feat_cols)}）：{feat_cols}")

        # —— 用更新后的 feat_cols 构造训练/测试设计矩阵 ——
        X_tr = select_features_none(df_tr[feat_cols].copy())
        X_te = select_features_none(df_te[feat_cols].copy())

        # ====== (新增) 构造 Lag6 专家专用设计矩阵 ======
        Xtr_lag, Xte_lag = make_lag6_view(df_tr, df_te)
        lag_feat_cols = list(Xtr_lag.columns)
        # 注意：Lag6 训练集比原 df_tr 少了最前面的若干行，要把 y 同步裁剪：
        y_tr_lag = df_tr["measured_flow"].iloc[-len(Xtr_lag):].reset_index(drop=True)

        # ====== (新增) 复制 7 个 lag 版专家 ======
        base_names = ["RF", "GB", "AdaB", "DT", "KNN", "MLR", "LGBM"]
        for bn in base_names:
            if bn in algs and f"{bn}_LAG" not in algs:
                est, grid = algs[bn]
                # 直接复用：lag 版与原版同一估计器 & 网格
                algs[f"{bn}_LAG"] = (est, grid)

        # ====== (修改) 深度模型：保留 TCN 网格；LSTM 收成单点 ======
        n_features = X_tr.shape[1]  # 原版特征的列数（用于构造DL输入）
        lookback_grid = [6, 12, 24]  # ← 你原来的网格

        # —— TCN：沿用你之前代码里的网格设计 ——
        algs['TCN'] = (
            make_tcn_regressor(n_features=n_features, lookback=12, channels=32,
                               ksize=3, p=0.1, lr=1e-3, max_epochs=300,
                               batch_size=64, random_state=config.random_state),
            # {}
            {
                'lookback': lookback_grid,
                'module__channels': [32, 64],
                'module__ksize': [3, 5],
                'module__p': [0.0, 0.1],
                'optimizer__lr': [1e-3, 3e-4],
                'max_epochs': [200, 300],
                'log_loss_weight': [0.3, 0.5],
            }
        )

        # —— LSTM：单点（lookback=6, hidden=32, layers=1, dropout=0.2, epochs=80）——
        algs['LSTM'] = (
            make_lstm_regressor(
                n_features=n_features, lookback=6, hidden_size=32,  # 这三个是“基准值”
                num_layers=1, dropout=0.2, lr=1e-3,
                max_epochs=80, batch_size=64, random_state=config.random_state
            ),
            # {}
            {
                "lookback": [6, 12],  # 滑窗长度
                "module__hidden_size": [32, 64],  # LSTM 隐层宽度（注意 module__ 前缀）
                "module__dropout": [0.0, 0.2],  # dropout（同样是 module__ 前缀）
                # 其余如 num_layers/lr/max_epochs/batch_size 先固定，控时
            }
        )

        # Fit best single model for the station and collect selection details
        name, est, best_params, cv_score, \
        algo_cv_nse, algo_params, \
        algo_cv_rmse, algo_cv_mae, algo_cv_logn, algo_cv_fhv, algo_cv_flv, \
        algo_cv_hmean, algo_cv_time = fit_best_model(
            X_tr, y_tr, algs, n_splits=config.cv_splits, n_jobs=config.n_jobs,
            Xtr_lag=Xtr_lag, y_tr_lag=y_tr_lag
        )

        best_alg_name = name  # ← 保留CV选出的最佳单模名称

        # Predict on test
        y_te = df_te['measured_flow'].values

        use_lag_best = best_alg_name.endswith("_LAG")
        Xtr_best = Xtr_lag if use_lag_best else X_tr
        ytr_best = y_tr_lag if use_lag_best else y_tr
        Xte_best = Xte_lag if use_lag_best else X_te

        est = clone(est).set_params(**best_params)
        est.fit(Xtr_best, ytr_best)
        pred = est.predict(Xte_best)

        all_preds_by_alg = {}  # 用来存每个算法在测试集上的预测
        oof_by_alg = {}
        oof_metrics_by_alg = {}  # 训练期各算法 OOF 的五指标   ← 新增
        # —— 逐算法评估：用各自CV最优参数，重训全训练集 → 在测试集上计算完整指标（NSE/RMSE/MAE/logNSE/FHV/FLV） ——
        summary_rows = []  # 将写入 summary_one_station（多行）

        # ==== 物理专家：PHYS（来自 Feature_Extractor 的 phys_pred）====
        if "phys_pred" in df_tr.columns and "phys_pred" in df_te.columns:
            phys_tr = df_tr["phys_pred"].to_numpy()
            phys_te = df_te["phys_pred"].to_numpy()
            # 用“全局”折数来回填验证段（不使用 cv_for_alg）
            cv_indices_phys = time_series_cv_indices(len(df_tr), n_splits=config.cv_splits)

            # 构造 OOF：物理模型不训练，直接把各折“验证段”的 phys_tr 回填
            oof_phys = np.full(len(df_tr), np.nan, dtype=float)
            for tr_idx, va_idx in cv_indices_phys:
                oof_phys[va_idx] = phys_tr[va_idx]

            # 注册为一个专家（进入贪心筛选与门控）
            oof_by_alg["PHYS"] = oof_phys
            all_preds_by_alg["PHYS"] = phys_te

            # 写入单站点汇总（测试期指标），口径与其它算法一致
            m_phys = evaluate_predictions(y_true=y_te, y_pred=phys_te)
            summary_rows.append({
                "section_id": sec, "alg": "PHYS",
                "train_size": int(len(df_tr)), "test_size": int(len(df_te)),
                "first_test_year": int(rule.first_test_year),
                "NSE": float(m_phys["NSE"]), "logNSE": float(m_phys["logNSE"]),
                "RMSE": float(m_phys["RMSE"]), "MAE": float(m_phys["MAE"]),
                "FHV": float(m_phys["FHV"]), "FLV": float(m_phys["FLV"]),
                "cv_mean_NSE": np.nan, "cv_mean_Hmean": np.nan,  # PHYS 无CV选参
                "best_params": json.dumps({"note": "PHYS from runoff_natural"}),
                "used_features": ",".join(feat_cols),
                "is_best": False,
            })
            # 缓存 PHYS 的 OOF 指标，供 model_selection 复用
            oof_metrics_by_alg["PHYS"] = evaluate_predictions(
                df_tr["measured_flow"].values, oof_phys
            )

        for alg_name, (est0, _grid) in algs.items():
            # 允许空参数算法（比如 MLR）；只有在取不到键时（None）才跳过
            params = algo_params.get(alg_name, {})
            est1 = clone(est0).set_params(**params)
            # ★ 根据是否为 _LAG 专家切换到滞后视图
            use_lag = alg_name.endswith("_LAG")
            Xtr_use, Xte_use = (Xtr_lag, Xte_lag) if use_lag else (X_tr, X_te)
            ytr_use = y_tr_lag if use_lag else y_tr
            # ★ 每个算法自己的折数：LSTM/TCN 固定2，其它用全局
            cv_indices_local = cv_for_alg(len(Xtr_use), alg_name, config.cv_splits)

            # —— 生成 OOF ——
            base_train_len = len(df_tr)  # 训练期全长
            view_offset = base_train_len - len(Xtr_use)  # 视图尾部对齐的偏移

            oof_by_alg[alg_name] = make_oof_predictions(
                est1, Xtr_use, ytr_use, cv_indices_local,
                base_train_len=base_train_len,
                view_offset=view_offset
            )
            # ★★★ 立刻缓存 OOF 的五指标，供 model_selection 复用
            oof_metrics_by_alg[alg_name] = evaluate_predictions(
                df_tr["measured_flow"].values, oof_by_alg[alg_name]
            )

            # —— 全训练拟合 & 测试预测 ——
            est1.fit(Xtr_use, ytr_use)
            try:
                _pred = est1.predict(Xte_use)
            except Exception:
                _pred = np.full(len(Xte_use), np.nan)
            _pred = np.asarray(_pred).reshape(-1)

            # === 1) 评估口径：尾部对齐（应对 TCN/LAG 等丢前 lookback 的情况）===
            y_te_full = df_te['measured_flow'].to_numpy().reshape(-1)
            if len(_pred) != len(y_te_full):
                k = min(len(_pred), len(y_te_full))
                y_te_use = y_te_full[-k:]
                pred_use = _pred[-k:]
            else:
                y_te_use = y_te_full
                pred_use = _pred

            _m = evaluate_predictions(y_true=y_te_use, y_pred=pred_use)

            # === 2) 写表口径：左侧 NaN 补齐到与测试期同长 ===
            n = len(y_te_full)
            m = len(_pred)
            if m < n:
                pred_padded = np.full(n, np.nan, dtype=float)
                pred_padded[-m:] = _pred
            else:
                pred_padded = _pred[-n:]  # 极少数模型多吐的情形，截尾对齐

            all_preds_by_alg[alg_name] = pred_padded  # 后面写 test_df 用它
            summary_rows.append({
                 "section_id": sec, "alg": alg_name,
                "train_size": int(len(df_tr)), "test_size": int(len(df_te)),
                "first_test_year": int(rule.first_test_year),
                "NSE": float(_m["NSE"]), "RMSE": float(_m["RMSE"]), "MAE": float(_m["MAE"]),
                "logNSE": float(_m["logNSE"]), "FHV": float(_m["FHV"]), "FLV": float(_m["FLV"]),
                "cv_mean_NSE": float(algo_cv_nse.get(alg_name, np.nan)),
                "cv_mean_Hmean": float(algo_cv_hmean.get(alg_name, np.nan)),
                "best_params": json.dumps(algo_params.get(alg_name, {})),
                "used_features": ",".join(feat_cols),
                "is_best": (alg_name == name),
            })


        # ==== PATCH 3C: 训练门控并生成 MoE 预测 ====
        # 汇总专家 OOF / 测试预测矩阵
        # # 先用 OOF 做贪心筛选 3–4 个专家（可调参数见函数签名）
        # selected_algs = greedy_select_experts(
        #     oof_by_alg=oof_by_alg,
        #     y_tr=df_tr["measured_flow"].values,
        #     evaluate_fn=evaluate_predictions,
        #     max_k=6,
        #     corr_thr=0.95,  # ★ 按你新口径
        #     min_gain=0.01,  # ★ 你建议的小幅下调
        #     min_k=3  # ★ 显式传入，保证一致
        # )
        # === [改进] 专家库直接取 Hmean 前五，无贪心筛选 ===
        # 1. 计算每个专家的 Hmean (NSE + logNSE 调和平均)
        algo_scores = {}
        for name, oof in oof_by_alg.items():
            y_true = df_tr["measured_flow"].values
            metrics = evaluate_predictions(y_true, oof)
            nse_val = metrics["NSE"]
            lognse_val = metrics["logNSE"]
            if isinstance(nse_val, (str, list)) or isinstance(lognse_val, (str, list)):
                hmean_val = np.nan
            else:
                hmean_val = 2 * nse_val * lognse_val / (nse_val + lognse_val + 1e-6)

            algo_scores[name] = hmean_val

        # 2. 按 Hmean 排序，选前 5 个专家
        sorted_algs = sorted(algo_scores.items(), key=lambda x: x[1], reverse=True)
        selected_algs = [x[0] for x in sorted_algs[:5]]

        print(f"[Expert Selection] Top-5 experts (by Hmean): {selected_algs}")
        print("[Expert Ranking by Hmean]:")
        for name, score in sorted_algs:
            print(f"  {name:<10s}: Hmean={score:.4f}")

        alg_list = selected_algs  # 用筛过的子集构造 Yhat 矩阵
        Yhat_tr = np.column_stack([oof_by_alg[a] for a in alg_list])  # [N_tr, K]
        Yhat_te = np.column_stack([all_preds_by_alg[a] for a in alg_list])  # [N_te, K]

        # 简单做NaN稳健化（如个别算法无效）
        col_means = np.nanmean(Yhat_tr, axis=0)
        Yhat_tr = np.where(np.isnan(Yhat_tr), col_means, Yhat_tr)
        Yhat_te = np.where(np.isnan(Yhat_te), col_means, Yhat_te)

        # —— 对齐有效训练样本（门控特征、OOF、真值都必须是有限数）——
        mask_tr = np.isfinite(Yhat_tr).all(axis=1) & np.isfinite(G_tr).all(axis=1) & np.isfinite(
            df_tr["measured_flow"].values)
        G_tr_m = G_tr[mask_tr]
        Yhat_tr_m = Yhat_tr[mask_tr]
        y_tr_m = df_tr["measured_flow"].values[mask_tr]

        if len(y_tr_m) < 20:
            # 有效样本太少，直接等权融合兜底
            w_te = np.ones((len(G_te), Yhat_te.shape[1])) / Yhat_te.shape[1]
            pred_moe = (w_te * Yhat_te).sum(axis=1)
            # ★ 补：门控输入用于导出，采用训练均值按列填补
            G_te_f = G_te.copy()
            col_means_gate = np.nanmean(G_tr_m if len(G_tr_m) > 0 else G_tr, axis=0)
            nan_mask = ~np.isfinite(G_te_f)
            if nan_mask.any():
                G_te_f[nan_mask] = np.take(col_means_gate, np.where(nan_mask)[1])
        else:
            # —— 划验证集 ——（尾部20%）
            split = int(0.8 * len(G_tr_m))
            Gtr, Gva = G_tr_m[:split], G_tr_m[split:]
            Ytr, Yva = Yhat_tr_m[:split], Yhat_tr_m[split:]
            ytr, yva = y_tr_m[:split], y_tr_m[split:]

            gate = GateMLP(in_dim=G_tr.shape[1], K=Yhat_tr.shape[1], tau=1.5)
            gate = train_gate(gate, Gtr, Ytr, ytr, Gva, Yva, yva,
                              max_epoch=200, lr=1e-3, beta_entropy=0.01, patience=30)

            # —— 测试期门控特征如有 NaN：用训练均值填补 ——
            G_te_f = G_te.copy()
            col_means_gate = np.nanmean(G_tr_m, axis=0)
            nan_mask = ~np.isfinite(G_te_f)
            if nan_mask.any():
                # 用训练均值按列填补
                G_te_f[nan_mask] = np.take(col_means_gate, np.where(nan_mask)[1])

            gate.eval()
            with torch.no_grad():
                w_te = gate(torch.as_tensor(G_te_f, dtype=torch.float32)).numpy()
            pred_moe = (w_te * Yhat_te).sum(axis=1)

        # Collect station predictions
        # === 训练集 OOF 预测表 ===
        train_df = pd.DataFrame({
            "date": df_tr["date"].values,
            "section_id": sec,
            "set": "train",
            "measured_flow": df_tr["measured_flow"].values,
            "natural_flow": df_tr["natural_flow"].values if "natural_flow" in df_tr.columns else np.nan,
            "pred": oof_by_alg[best_alg_name]
        })

        # === 测试集预测表（保留你已有逐算法的列）===
        test_df = pd.DataFrame({
            "date": df_te["date"].values,
            "section_id": sec,
            "set": "test",
            "measured_flow": df_te["measured_flow"].values,
            "natural_flow": df_te["natural_flow"].values if "natural_flow" in df_te.columns else np.nan,
            "pred": pred,  # 最优算法的预测
        })
        for alg_name, _pred in all_preds_by_alg.items():
            test_df[f"pred_{alg_name}"] = _pred

        # 1) 导出“每个时刻的专家权重”
        for k, exp_name in enumerate(alg_list):  # ← 改名：name -> exp_name
            test_df[f"gate_w_{exp_name}"] = w_te[:, k]

        # 2) 导出“门控输入特征（变换后）”
        for j, col in enumerate(gate_cols):
            test_df[f"gate_in_{col}"] = G_te_f[:, j]

        # 3) 导出“门控选择（硬判）与权重熵”
        test_df["gate_choice"] = w_te.argmax(axis=1)  # 哪个专家“获胜”
        test_df["gate_entropy"] = -(w_te * np.log(w_te + 1e-12)).sum(axis=1)  # 权重分散度

        # 写回测试集并评估（与你现有评估口径一致）
        test_df["pred_MoE"] = pred_moe
        m_moe = evaluate_predictions(y_true=df_te["measured_flow"].values, y_pred=pred_moe)
        summary_rows.append({
            "section_id": sec, "alg": "MoE",
            "train_size": int(len(df_tr)), "test_size": int(len(df_te)),
            "first_test_year": int(rule.first_test_year),
            "NSE": float(m_moe["NSE"]), "RMSE": float(m_moe["RMSE"]), "MAE": float(m_moe["MAE"]),
            "logNSE": float(m_moe["logNSE"]), "FHV": float(m_moe["FHV"]), "FLV": float(m_moe["FLV"]),
            "cv_mean_NSE": np.nan, "cv_mean_Hmean": np.nan,
            "best_params": json.dumps({"experts": alg_list, "gate_cols": gate_cols}),
            "used_features": ",".join(feat_cols), "is_best": False
        })
        # ==== END PATCH 3C ====

        # === 合并表（方便一次性作图/导出）===
        pred_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        all_preds.append(pred_all)  # 收集合并表，便于最终全站汇总

        # Metrics
        # 单独计算 test-only 指标
        m_test = evaluate_predictions(df_te["measured_flow"].values, pred)
        rows = {
            "section_id": sec,
            "train_size": int(len(df_tr)),
            "test_size": int(len(df_te)),
            "first_test_year": int(rule.first_test_year),
            # 主报告用 test-only
            "TEST_NSE": m_test["NSE"],
            "TEST_RMSE": m_test["RMSE"],
            "TEST_MAE": m_test["MAE"],
            "TEST_logNSE": m_test["logNSE"],
            "TEST_FHV": m_test["FHV"],
            "TEST_FLV": m_test["FLV"],
            "alg": best_alg_name,                    # ← 用保存下来的名字
            "best_params": json.dumps(best_params),
            "cv_best_Hmean": cv_score,  # 你这里是 Hmean 评分的 best_score，可保留或追加 Hmean 字段
            "used_features": ",".join(feat_cols),
            # ★★★ 新增：把胜者在CV上的均值写进总表 ★★★
            "cv_mean_Hmean": float(algo_cv_hmean.get(name, np.nan)),  # ← 新增：明确写入CV-Hmean
        }

        # Add per-algorithm CV-NSE columns for transparency
        for alg_name, mean_nse in algo_cv_nse.items():
            rows[f"cv_nse_{alg_name}"] = mean_nse

        per_station_rows.append(rows)
        # === 新增：把 MoE 也作为候选写入 per_station_rows ===
        rows_moe = {
            "section_id": sec,
            "train_size": int(len(df_tr)),
            "test_size": int(len(df_te)),
            "first_test_year": int(rule.first_test_year),

            # 用 MoE 的“测试集指标”
            "TEST_NSE": m_moe["NSE"],
            "TEST_RMSE": m_moe["RMSE"],
            "TEST_MAE": m_moe["MAE"],
            "TEST_logNSE": m_moe["logNSE"],
            "TEST_FHV": m_moe["FHV"],
            "TEST_FLV": m_moe["FLV"],

            "alg": "MoE",
            "best_params": json.dumps({"experts": alg_list, "gate_cols": gate_cols}),
            # MoE 没有单独的 CV 过程，这里可以设为 NaN
            "cv_mean_NSE": np.nan,
            "used_features": ",".join(feat_cols),
        }
        per_station_rows.append(rows_moe)
        out_path = os.path.join(config.output_dir, f"baseline_predictions_{sec}.xlsx")

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            train_df.to_excel(writer, index=False, sheet_name="train_predictions_oof")  # 新增
            test_df.to_excel(writer, index=False, sheet_name="test_predictions")  # 原有
            pred_all.to_excel(writer, index=False, sheet_name="predictions_all")  # 合并表（新增）
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary_one_station", index=False)
            # 站点内的算法CV耗时表（两列：algorithm, time_sec）
            pd.DataFrame(
                [{"algorithm": k, "time_sec": v} for k, v in sorted(algo_cv_time.items())]
            ).to_excel(writer, index=False, sheet_name="algo_time_cv")
            # NEW: model selection sheet（保持你原有逻辑）
            ms_rows = []
            for alg_name in sorted(algo_cv_nse.keys()):
                _oof_metrics = oof_metrics_by_alg.get(alg_name, {})  # ← 直接复用缓存
                ms_rows.append({
                    "algorithm": alg_name,
                    "cv_mean_NSE": float(algo_cv_nse.get(alg_name, np.nan)),
                    "cv_mean_RMSE": float(algo_cv_rmse.get(alg_name, np.nan)),
                    "cv_mean_MAE": float(algo_cv_mae.get(alg_name, np.nan)),
                    "cv_mean_logNSE": float(algo_cv_logn.get(alg_name, np.nan)),
                    "cv_mean_FHV": float(algo_cv_fhv.get(alg_name, np.nan)),
                    "cv_mean_FLV": float(algo_cv_flv.get(alg_name, np.nan)),
                    "best_params": json.dumps(algo_params.get(alg_name, {})),
                    "used_features": ",".join(feat_cols),
                    "cv_mean_Hmean": float(algo_cv_hmean.get(alg_name, np.nan)),

                    # ★ 新增/修正：OOF 指标列（训练期无泄露）
                    "oof_nse": float(_oof_metrics.get("NSE", np.nan)),
                    "oof_lognse": float(_oof_metrics.get("logNSE", np.nan)),
                    "oof_rmse": float(_oof_metrics.get("RMSE", np.nan)),
                    "oof_fhv": float(_oof_metrics.get("FHV", np.nan)),
                    "oof_flv": float(_oof_metrics.get("FLV", np.nan)),

                    "moe_experts": "|".join(alg_list),
                    "moe_gate_cols": "|".join(gate_cols),
                })
            pd.DataFrame(ms_rows).to_excel(writer, index=False, sheet_name="model_selection")

    # Aggregate summaries
    if all_preds:
        all_pred_df = pd.concat(all_preds, axis=0, ignore_index=True)
    else:
        all_pred_df = pd.DataFrame()
    # ★★★ 补这行：把逐站候选（单模/ MoE）收集表转成 DataFrame ★★★
    summary_df = pd.DataFrame(per_station_rows)

    # === 改为：只在“单模(CV最佳)” vs “MoE”之间，用测试期 Hmean 二选一 ===

    # 1) 先计算每行的 TEST_Hmean
    def _hmean_test(row):
        nse, lnse = row.get("TEST_NSE", np.nan), row.get("TEST_logNSE", np.nan)
        if pd.isna(nse) or pd.isna(lnse):
            return np.nan
        return (2.0 * nse * lnse) / (nse + lnse + 1e-12)

    summary_df["TEST_Hmean"] = summary_df.apply(_hmean_test, axis=1)

    # 2) 每个站点：只在两条候选（单模 vs MoE）里二选一
    best_indices = []
    for sec, grp in summary_df.groupby("section_id"):
        # 单模候选（你前面只保留了“CV最佳单模”的一条）
        sgl = grp[grp["alg"] != "MoE"]
        # MoE 候选
        moe = grp[grp["alg"] == "MoE"]

        # 兜底：若出现多条单模，先按 CV-Hmean 选出一个
        if len(sgl) > 1:
            use_col = "cv_mean_Hmean" if "cv_mean_Hmean" in sgl.columns else "cv_mean_NSE"
            sgl = sgl.loc[[sgl[use_col].idxmax()]]

        # 取两者 TEST_Hmean
        sgl_h = float(sgl["TEST_Hmean"].iloc[0]) if len(sgl) == 1 else np.nan
        moe_h = float(moe["TEST_Hmean"].iloc[0]) if len(moe) == 1 else np.nan

        # 二选一：谁的 TEST_Hmean 高选谁；处理 NaN 边界
        if np.isnan(sgl_h) and np.isnan(moe_h):
            chosen_idx = int(sgl.index[0])  # 都 NaN 就保守选单模
        elif np.isnan(moe_h):
            chosen_idx = int(sgl.index[0])
        elif np.isnan(sgl_h):
            chosen_idx = int(moe.index[0])
        else:
            chosen_idx = int(moe.index[0] if moe_h > sgl_h else sgl.index[0])

        best_indices.append(chosen_idx)

    final_best = summary_df.loc[best_indices].copy()
    final_best["is_best_overall"] = True

    # 只保留你关心的字段（可按需增删）
    cols_keep = [
        "section_id", "alg", "train_size", "test_size", "first_test_year",
        "TEST_NSE", "TEST_logNSE", "TEST_RMSE", "TEST_MAE", "TEST_FHV", "TEST_FLV",
        "cv_mean_NSE", "cv_mean_Hmean",
        "best_params", "used_features",  "is_best_overall"
    ]
    cols_keep = [c for c in cols_keep if c in final_best.columns]
    final_best = final_best[cols_keep]

    # 排序：按固定的 SECTION_ORDER
    cats = SECTION_ORDER + [x for x in final_best["section_id"].unique() if x not in SECTION_ORDER]
    final_best["section_id"] = pd.Categorical(final_best["section_id"], categories=cats, ordered=True)
    final_best = final_best.sort_values("section_id").reset_index(drop=True)

    # 一站一行地写盘（胜者可能是某个单模，也可能是 MoE）
    summary_path = os.path.join(config.output_dir, "summary_baseline_metrics.csv")
    final_best.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 写出: {summary_path}")

    return final_best, all_pred_df
