# main.py

import joblib
import numpy as np
from data_loader import load_and_preprocess_data
from feature_engineer import create_time_features, create_warning_features
from dynamics_model import fit_dynamics_model, predict_full_model
from warning_system import train_warning_model
from visualization import plot_sentiment_trends, plot_model_validation
from config import config

def main():
    print("====== 网络舆情情感演化分析系统 ======")
    
    # 1. 数据加载和预处理
    raw_df = load_and_preprocess_data()
    
    # 2. 特征工程
    time_features = create_time_features(raw_df)
    plot_sentiment_trends(time_features)
    
    # 3. 划分训练集和测试集
    train_size = int(len(time_features) * config.TRAIN_SIZE)
    train_features = time_features.iloc[:train_size]
    test_features = time_features.iloc[train_size:]
    
    # 4. 训练动力学模型
    print("\n训练情感演化动力学模型...")
    params = fit_dynamics_model(train_features)
    print(f"最优参数: β_p={params[0]:.4f}, β_n={params[1]:.4f}, α_p={params[2]:.4f}, α_n={params[3]:.4f}, γ={params[4]:.4f}")
    
    # 保存模型参数
    np.save(f"{config.OUTPUT_DIR}/dynamics_params.npy", params)
    
    # 5. 全时间段预测
    predictions = predict_full_model(params, time_features)
    plot_model_validation(time_features, predictions, train_size)
    
    # 6. 构建预警系统
    print("\n构建舆情预警系统...")
    # 准备预警特征
    X, y = create_warning_features(time_features, model_pred_negative=1-predictions[1])
    
    # 划分训练集和测试集
    split_idx = train_size
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:-1]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:-1]
    
    # 训练预警模型
    warning_model = train_warning_model(X_train, y_train)
    
    # 保存预警模型
    joblib.dump(warning_model, f"{config.OUTPUT_DIR}/warning_model.pkl")
    
    print("\n分析完成! 结果保存在 results/ 目录")

if __name__ == "__main__":
    main()
