# Feature_Extractor.py
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

# （可选）给定的断面顺序，用于排序
SECTION_ORDER = [

    "Sanliang",
    "Zhuozishan",

]



def load_merged_df(db_dir: str) -> pd.DataFrame:
    """读取两张CSV并按(date, section_id)合并为一张表；统一列名与类型。"""
    fp_hydro = os.path.join(db_dir, '月度水文数据.csv')
    fp_demand = os.path.join(db_dir, '用水信息.csv')

    mh = pd.read_csv(fp_hydro, encoding='gbk')
    wd = pd.read_csv(fp_demand, encoding='gbk')

    # 时间列转为datetime（YYYYMM）
    mh['date'] = pd.to_datetime(mh['date'], format='%Y%m', errors='coerce')
    wd['date'] = pd.to_datetime(wd['date'], format='%Y%m', errors='coerce')

    # 合并
    df = mh.merge(wd, on=['date','section_id'], how='left')

    # ✅ 在 df 合并完成、各数值列转型之后，加上一行：
    df["phys_pred"] = df["runoff_natural"]  # 物理模型=天然径流，直接复制

    # 建议数值化（防止上游是字符串）
    df["phys_pred"] = pd.to_numeric(df["phys_pred"], errors="coerce")

    # 关键列转数值
    for c in ['rainfall','evaporation','runoff_measured','runoff_natural',
              'domestic','agricultural','industrial','ecological']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')


    # 统一目标：天然-实测
    if 'target' not in df.columns and {'runoff_natural','runoff_measured'} <= set(df.columns):
        df['target'] = df['runoff_natural'] - df['runoff_measured']

    # —— 新增特征 ——
    # 1) 用水强度比 I_t： (domestic + agricultural + industrial + ecological) / 基准流量
    #    基准流量优先用 runoff_natural，否则回退到 runoff_measured；分母做非负与极小值保护
    _use_cols_all = [c for c in ['domestic', 'agricultural', 'industrial', 'ecological'] if c in df.columns]
    df['_total_use_all'] = df[_use_cols_all].sum(axis=1) if _use_cols_all else 0.0
    _den = df['runoff_natural'] if 'runoff_natural' in df.columns else df.get('runoff_measured')
    if _den is not None:
        _den = _den.clip(lower=0)
        df['I_t'] = df['_total_use_all'] / _den.replace(0, 1e-9)
    else:
        df['I_t'] = np.nan

    # 2) 结构比例三项：以（ag, ind, dom）三项之和为分母
    if all(c in df.columns for c in ['agricultural', 'industrial', 'domestic']):
        _sum3 = (df['agricultural'].clip(lower=0) +
                 df['industrial'].clip(lower=0) +
                 df['domestic'].clip(lower=0)).replace(0, 1e-9)
        df['ag_share'] = df['agricultural'].clip(lower=0) / _sum3
        df['ind_share'] = df['industrial'].clip(lower=0) / _sum3
        df['dom_share'] = df['domestic'].clip(lower=0) / _sum3
    else:
        df['ag_share'] = df['ind_share'] = df['dom_share'] = np.nan

    # 3) 季节相位：对月份做周期编码（sin/cos，周期=12）
    if 'date' in df.columns:
        _m = df['date'].dt.month.astype(float)
        df['season_sin'] = np.sin(2 * np.pi * (_m - 1) / 12.0)
        df['season_cos'] = np.cos(2 * np.pi * (_m - 1) / 12.0)
    else:
        df['season_sin'] = df['season_cos'] = np.nan

    # —— 批2：新增的两个特征（本回答的重点） ——
    # # 1) 蓄水记忆 M^Q：以天然径流为主，缺失时回退到实测；6个月移动平均
    # if 'runoff_natural' in df.columns:
    #     df['M_nat6'] = df.groupby('section_id')['runoff_natural'] \
    #         .transform(lambda s: s.rolling(window=6, min_periods=1).mean())
    # elif 'runoff_measured' in df.columns:
    #     df['M_nat6'] = df.groupby('section_id')['runoff_measured'] \
    #         .transform(lambda s: s.rolling(window=6, min_periods=1).mean())
    # else:
    #     df['M_nat6'] = np.nan

    # （你也可以改成EWMA）
    if 'runoff_natural' in df.columns:
        df['M_nat6'] = df.groupby('section_id')['runoff_natural'] \
            .transform(lambda s: s.ewm(alpha=0.2, adjust=False).mean())
    elif 'runoff_measured' in df.columns:
        df['M_nat6'] = df.groupby('section_id')['runoff_measured'] \
            .transform(lambda s: s.ewm(alpha=0.2, adjust=False).mean())
    else:
        df['M_nat6'] = np.nan

    # 2) 干旱相位 D_t：P-ET<0 为干旱月（布尔→0/1）
    if {'rainfall', 'evaporation'} <= set(df.columns):
        df['D_phase'] = ((df['rainfall'] - df['evaporation']) < 0).astype(int)
    else:
        df['D_phase'] = np.nan


    # 列顺序（其余列放后面）
    base_cols = ['date', 'section_id', 'rainfall', 'evaporation',
                 'runoff_measured', 'runoff_natural',
                 'domestic', 'agricultural', 'industrial', 'ecological', 'target',
                 # 季节相位
                'season_sin', 'season_cos',
                # 用水信息
                 'I_t', 'ag_share', 'ind_share', 'dom_share',
                # 蓄水记忆
                'M_nat6', 'D_phase',
                 # "phys_pred",  # ← 新增：物理模型预测
                 ]

    # 仅按照 base_cols 提取并重排列（不存在的列会被自动跳过）
    cols = [c for c in base_cols if c in df.columns]  # 1) 基于 base_cols 生成实际存在的列顺序
    df = df.loc[:, cols]  # 2) 按 cols 的新顺序重排列

    # 3) 再按 section_id、date 排序行，并重置索引为 0..n-1
    df = df.sort_values(['section_id', 'date']).reset_index(drop=True)
    eps = 1e-6
    use_cols = [c for c in ["domestic", "agricultural", "industrial", "ecological"] if c in df.columns]
    if len(use_cols) == 0:
        # 若你有“总用水”列（例如 U_total），改成相应列名
        df["U_total"] = 0
    else:
        df["U_total"] = df[use_cols].sum(axis=1)

    # 人类活动强度（滞后，原始比值）
    # 按站点分组做滞后与分母，避免跨站点串值
    den_t1 = df.groupby("section_id")["runoff_natural"].shift(1)
    den_t1 = np.maximum(den_t1, eps)  # 分母是上月的天然径流（加一个极小量 eps 防止除零）。
    # 门控将用到的“滞后原始量”  ；.shift(1)：滞后 1 期 → 取上一个月（t−1）的值。
    df["gate_logQ_t1_raw"] = df.groupby("section_id")["runoff_measured"].shift(1).pipe(np.log1p)
    # 实测流量相对于天然流量的偏差比例
    df["gate_HAI1_t1_raw"] = (df.groupby("section_id")["runoff_measured"].shift(1)
                              - df.groupby("section_id")["runoff_natural"].shift(1)) / den_t1
    df["gate_HAI3_t1_raw"] = df.groupby("section_id")["U_total"].shift(1) / den_t1

    return df

def write_features_single_sheet(df: pd.DataFrame, output_path: str,
                                sections: Optional[List[str]] = None,
                                sort_by_section_order: bool = True) -> str:
    """把单表直接写成一个sheet（Features）。可筛选断面+按固定顺序排序。"""
    out = df.copy()
    if sections:
        out = out[out['section_id'].isin(sections)].copy()

    if sort_by_section_order and 'section_id' in out.columns:
        # 只保留在 SECTION_ORDER 中出现的站点
        out = out[out['section_id'].isin(SECTION_ORDER)].copy()
        out['section_id'] = pd.Categorical(out['section_id'], categories=SECTION_ORDER, ordered=True)
        out = out.sort_values(['section_id', 'date']).reset_index(drop=True)

    with pd.ExcelWriter(output_path) as w:
        out.to_excel(w, sheet_name='Features', index=False)
    print(f"[OK] 单表已写出: {output_path}")
    return output_path

def run_features(db_dir: str, output_path: str,
                 sections: Optional[List[str]] = None,
                 sort_by_section_order: bool = True) -> Tuple[pd.DataFrame, str]:
    """一键：读 → 合并 → 写；返回(df, 输出路径)。"""
    df_all = load_merged_df(db_dir)
    path = write_features_single_sheet(df_all, output_path, sections, sort_by_section_order)
    return df_all, path
