# main.py
import joblib
import numpy as np
import os

from config import config
from data_loader import load_and_preprocess_data
from feature_engineer import create_time_features, create_warning_features
from dynamics_model import fit_dynamics_model, fit_dynamics_model_improved, predict_full_model
from warning_system import train_warning_model
from visualization import plot_sentiment_trends, plot_model_validation_comparison # 使用新函数

def main():
    print("====== 网络舆情情感演化分析系统 (动态参数改进版) ======")
    
    ensure_output_dir()

    # 1. 数据加载与预处理
    raw_df = load_and_preprocess_data()
    
    # 2. 特征工程
    time_features = create_time_features(raw_df)
    plot_sentiment_trends(time_features)
    
    # 3. 划分训练/测试集
    train_size = int(len(time_features) * config.TRAIN_SIZE)
    train_features = time_features.iloc[:train_size]
    
    # --- 动力学模型训练与对比 ---
    
    # 4a. 训练【原始】模型
    print("\n--- 正在训练【原始】动力学模型 ---")
    params_original = fit_dynamics_model(train_features)
    print(f"原始模型最优参数: β_p={params_original[0]:.4f}, β_n={params_original[1]:.4f}, α_p={params_original[2]:.4f}, α_n={params_original[3]:.4f}, γ={params_original[4]:.4f}")
    
    # 4b. 训练【改进】模型
    print("\n--- 正在训练【改进的】动力学模型 ---")
    params_improved = fit_dynamics_model_improved(train_features)
    print(f"改进模型最优参数: base_β_p={params_improved[0]:.4f}, base_β_n={params_improved[1]:.4f}, α_p={params_improved[2]:.4f}, α_n={params_improved[3]:.4f}, γ={params_improved[4]:.4f}, k_p={params_improved[5]:.4f}, k_n={params_improved[6]:.4f}")
    
    np.save(f"{config.OUTPUT_DIR}/dynamics_params_improved.npy", params_improved)
    
    # 5. 使用两种模型进行全时段预测
    print("\n--- 正在生成模型预测结果 ---")
    predictions_original = predict_full_model(params_original, time_features, improved=False)
    predictions_improved = predict_full_model(params_improved, time_features, improved=True)

    # 6. 可视化对比两种模型的预测结果
    plot_model_validation_comparison(time_features, predictions_original, predictions_improved, train_size)
    
    # --- 舆情预警系统构建 ---
    print("\n--- 正在构建舆情预警系统 ---")
    # 使用改进模型的预测作为特征
    X, y = create_warning_features(time_features, model_pred_negative=predictions_improved[2])
    
    # 划分预警模型的训练集和测试集 (严格按时间)
    split_idx = train_size
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:-1]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:-1]
    
    # 训练并评估预警模型
    if not X_train.empty and not y_train.empty and not X_test.empty and not y_test.empty:
        warning_model = train_warning_model(X_train, y_train, X_test, y_test)
        joblib.dump(warning_model, f"{config.OUTPUT_DIR}/warning_model.pkl")
        print("✓ 预警模型已保存。")
    else:
        print("警告：用于预警模型的训练或测试数据为空，跳过训练。")

    print(f"\n分析完成! 所有结果已保存在 '{config.OUTPUT_DIR}/' 目录中。")

def ensure_output_dir():
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

if __name__ == "__main__":
    main()