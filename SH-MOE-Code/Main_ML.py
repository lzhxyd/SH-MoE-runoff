from SPH_MOE2 import RunConfig, StationRule, run
from datetime import datetime
import os
from Feature_Extractor import run_features
import matplotlib.pyplot as plt

# 主程序入口
if __name__ == "__main__":
    # 在程序开始时设置字体，以解决乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 执行主函数1 （特征提取）
    db_dir = r"E:\...\data"
    output_dir = r"E:\...\OUT"
    # 输出路径
    output_path = os.path.join(output_dir, f"features_analysis_{timestamp}.xlsx")
    df_all, feature_output_path = run_features(db_dir, output_path)
    print("完成特征提取")

    # 执行主函数 2 (特征相关性及拟合分析)
    # 长江15个；大黑河10个
    STATION_CONFIG_RULES = {
        "Sanliang": StationRule(first_test_year=2018),
        "Zhuozishan": StationRule(first_test_year=2018),
    }

    cfg = RunConfig(
        excel_path=output_path,
        output_dir=output_dir,
        station_rules=STATION_CONFIG_RULES,
        random_state=42,
        cv_splits=3,
        n_jobs=1
    )
    # 3) 运行
    summary_df, all_pred_df = run(cfg)

