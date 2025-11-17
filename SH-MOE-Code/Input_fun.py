# 函数调用（读取，指标）；拟合（fit） 与 变换（transform）函数
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import os

def load_excel_dataset(excel_path: str) -> pd.DataFrame:
    """
    读取单个 sheet: 'Features'
    必含列：date, section_id, runoff_measured, runoff_natural
    读取后统一命名：
      measured_flow <- runoff_measured
      natural_flow  <- runoff_natural
    其他数值列一律作为特征保留。
    """
    df = pd.read_excel(excel_path, sheet_name='Features')
    df = df.dropna(how='all').copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 列重命名：与训练代码内部保持一致字段名
    rename_map = {}
    if 'runoff_measured' in df.columns:
        rename_map['runoff_measured'] = 'measured_flow'
    if 'runoff_natural' in df.columns:
        rename_map['runoff_natural'] = 'natural_flow'
    df = df.rename(columns=rename_map)

    # 仅保留数值列 + 基础ID列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = ['date', 'section_id'] + sorted(set(num_cols + ['measured_flow', 'natural_flow']))
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(['section_id', 'date']).reset_index(drop=True)
    return df

def time_series_cv_indices(n_samples: int, n_splits: int):
    """
    Forward-chaining CV indices so validation is later than training.
    Equivalent to sklearn.model_selection.TimeSeriesSplit without gaps.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(np.arange(n_samples)))


def select_features_none(X: pd.DataFrame) -> pd.DataFrame:
    """No feature selection; drop target and IDs only if present."""
    drop_cols = set(['measured_flow', 'section_id'])
    keep = [c for c in X.columns if c not in drop_cols]
    return X[keep]

# ---------- 指标辅助函数（中文注释） ----------
def _eps_from_obs(qobs: np.ndarray) -> float:
    """根据观测值自适应设置对数平滑项eps；若全为非正，则回退到0.1"""
    qobs = np.asarray(qobs, dtype=float)
    pos = qobs[qobs > 0]
    if pos.size == 0:
        return 0.1
    eps = 0.01 * float(np.median(pos))
    return max(eps, 0.1)

def nse(y_true, y_pred) -> float:
    """Nash-Sutcliffe效率系数（保持与原函数一致命名/语义；若已存在同名函数，下面替换逻辑会避免重复）"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return np.nan
    denom = np.sum((y_true - np.mean(y_true))**2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred)**2) / denom

def log_nse(y_true, y_pred, eps: float = None) -> float:
    """对数域的NSE（强调低流量拟合）；y<0截为0，再加eps取对数"""
    y_true = np.asarray(y_true, dtype=float).copy()
    y_pred = np.asarray(y_pred, dtype=float).copy()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    y_true[y_true < 0] = 0.0
    y_pred[y_pred < 0] = 0.0
    if y_true.size == 0:
        return np.nan
    if eps is None:
        eps = _eps_from_obs(y_true)
    yt = np.log(y_true + eps)
    yp = np.log(y_pred + eps)
    return nse(yt, yp)

def _fhv_flv_core(y_true: np.ndarray, y_pred: np.ndarray, top_q: float = 0.02, low_q: float = 0.30):
    """按流量持续曲线获取高/低流量索引并计算体积偏差（返回FHV%、FLV%）"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return np.nan, np.nan

    # 高流量（按值从大到小排序），取前top_q比例
    order_desc = np.argsort(-y_true)
    k_top = max(1, int(np.ceil(top_q * y_true.size)))
    idx_top = order_desc[:k_top]
    num_top = float(np.sum(y_true[idx_top]))
    fhv = np.nan
    if num_top > 0:
        fhv = 100.0 * float(np.sum(y_pred[idx_top] - y_true[idx_top])) / num_top

    # 低流量（从小到大排序），取后low_q比例
    order_asc = np.argsort(y_true)
    k_low = max(1, int(np.ceil(low_q * y_true.size)))
    idx_low = order_asc[:k_low]
    num_low = float(np.sum(y_true[idx_low]))
    flv = np.nan
    if num_low > 0:
        flv = 100.0 * float(np.sum(y_pred[idx_low] - y_true[idx_low])) / num_low

    return fhv, flv

# ======== 打分：NSE 与 logNSE 的调和平均 ========
def _hmean_pos(a: float, b: float, eps: float = 1e-6) -> float:
    try:
        a = float(a); b = float(b)
    except Exception:
        return float("nan")
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    a = max(a, eps); b = max(b, eps)
    return 2.0 * a * b / (a + b)

def nse_lognse_hmean(y_true, y_pred, eps: float = 1e-6) -> float:
    n = nse(y_true, y_pred)
    l = log_nse(y_true, y_pred)
    return _hmean_pos(n, l, eps=eps)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ==== PATCH 2: 折内 FDC 拟合 + 门控特征稳健化工具 ====
# 门控训练/推理将调用它们来生成最终的 gate_* 列。
def fit_fdc_and_scalers(q_train: np.ndarray,
                        hai_series_dict: dict,
                        qtiles: int = 100):
    """
    折内拟合：
      - 经验FDC网格（q_grid/p_grid）用于把 Q_{t-1} -> 分位 r_{t-1}
      - HAI类特征（滞后原始）的均值/方差（做 tanh z-score 稳健化用）
    参数:
      q_train: 训练折的 runoff_measured（或你选定口径）的数组
      hai_series_dict: 例如 {"gate_HAI1_t1_raw": series_train, "gate_HAI3_t1_raw": series_train}
    返回:
      meta: dict 包含 {"q_grid","p_grid","scalers":{col:(mu,sigma),...}}
    """
    q_train = np.asarray(pd.Series(q_train).dropna(), dtype=float)
    # FDC 网格（升序分位点）
    p_grid = np.linspace(0, 1, qtiles + 1)
    if len(q_train) == 0:
        q_grid = np.linspace(0, 1, qtiles + 1) * 0.0
    else:
        q_grid = np.quantile(q_train, p_grid)

    scalers = {}
    for col, s in hai_series_dict.items():
        s = pd.Series(s).dropna()
        mu = float(s.mean()) if len(s) else 0.0
        sd = float(s.std()) if len(s) else 1.0
        if sd < 1e-8: sd = 1.0
        scalers[col] = (mu, sd)

    return {"q_grid": q_grid.tolist(), "p_grid": p_grid.tolist(), "scalers": scalers}


def apply_gate_transforms(df: pd.DataFrame,
                          meta: dict,
                          inplace: bool = False):
    """
    用折内 meta 对 df 变换，生成最终 gate_* 列：
      - gate_logQ_t1 = log1p(Q_{t-1})（已在原始层生成，这里只做补充确保存在）
      - gate_r_t1：Q_{t-1} 在折内FDC上的分位[0,1]
      - gate_LowFlag_t1 / gate_HighFlag_t1：分位旗标（0/1）
      - gate_HAI1_t1 / gate_HAI3_t1：对原始比值做 tanh z-score 稳健化
    """
    if not inplace:
        df = df.copy()

    eps = 1e-6
    # 1) 确保 logQ_t1 存在
    if "gate_logQ_t1_raw" in df.columns and "gate_logQ_t1" not in df.columns:
        df["gate_logQ_t1"] = df["gate_logQ_t1_raw"]
    elif "gate_logQ_t1" not in df.columns:
        df["gate_logQ_t1"] = np.log1p(df["measured_flow"].shift(1))

    # 2) FDC 分位
    q_grid = np.asarray(meta["q_grid"], dtype=float)
    p_grid = np.asarray(meta["p_grid"], dtype=float)

    def _to_rank(q):
        if pd.isna(q): return np.nan
        # 线性插值到[0,1]；超界截断
        r = float(np.interp(q, q_grid, p_grid))
        if r < 0: r = 0.0
        if r > 1: r = 1.0
        return r

    df["gate_r_t1"] = df["measured_flow"].shift(1).apply(_to_rank)
    # df["gate_LowFlag_t1"] = (df["gate_r_t1"] <= 0.30).astype(float)
    # df["gate_HighFlag_t1"] = (df["gate_r_t1"] >= 0.90).astype(float)

    # 3) HAI 稳健化（tanh z-score；均值/方差来自折内）
    for raw_col, (mu, sd) in meta["scalers"].items():
        if raw_col not in df.columns:
            continue
        z = (df[raw_col] - mu) / (sd + eps)
        out_col = raw_col.replace("_raw", "")  # gate_HAI1_t1_raw -> gate_HAI1_t1
        df[out_col] = np.tanh(z)

    # 返回最终门控列清单（最小集合）
    gate_cols = [c for c in [
        "season_sin","season_cos",
        "gate_logQ_t1","gate_r_t1",
        # "gate_LowFlag_t1","gate_HighFlag_t1",
        "gate_HAI1_t1","gate_HAI3_t1"
    ] if c in df.columns]

    return df if inplace else (df, gate_cols)