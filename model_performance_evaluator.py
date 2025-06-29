# model_performance_evaluator.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import config
from data_loader import load_and_preprocess_data
from feature_engineer import create_time_features
from dynamics_model import fit_dynamics_model, fit_dynamics_model_improved, predict_full_model

def calculate_model_performance_metrics():
    """计算并对比两个模型的性能指标：MSE、MAE、R²"""
    print("====== 模型性能评估分析 ======")
    
    # 1. 数据加载与预处理
    print("正在加载数据...")
    raw_df = load_and_preprocess_data()
    time_features = create_time_features(raw_df)
    
    # 2. 划分训练/测试集
    train_size = int(len(time_features) * config.TRAIN_SIZE)
    train_features = time_features.iloc[:train_size]
    test_features = time_features.iloc[train_size:]
    
    print(f"训练集大小: {len(train_features)} 个时间窗口")
    print(f"测试集大小: {len(test_features)} 个时间窗口")
    
    # 3. 训练原始SIR模型
    print("\n正在训练原始SIR模型...")
    params_original = fit_dynamics_model(train_features)
    print(f"原始模型参数: β_p={params_original[0]:.4f}, β_n={params_original[1]:.4f}, α_p={params_original[2]:.4f}, α_n={params_original[3]:.4f}, γ={params_original[4]:.4f}")
    
    # 4. 训练改进的动态耦合SIR模型
    print("\n正在训练动态耦合SIR模型...")
    params_improved = fit_dynamics_model_improved(train_features)
    print(f"改进模型参数: base_β_p={params_improved[0]:.4f}, base_β_n={params_improved[1]:.4f}, α_p={params_improved[2]:.4f}, α_n={params_improved[3]:.4f}, γ={params_improved[4]:.4f}, k_p={params_improved[5]:.4f}, k_n={params_improved[6]:.4f}")
    
    # 5. 在测试集上进行预测
    print("\n正在生成测试集预测...")
    test_predictions_original = predict_full_model(params_original, test_features, improved=False)
    test_predictions_improved = predict_full_model(params_improved, test_features, improved=True)
    
    # 6. 计算性能指标（主要关注负面情感预测，因为这是舆情监控的重点）
    y_true_negative = test_features['negative_ratio'].values
    y_pred_original = test_predictions_original[2]  # 负面情感预测
    y_pred_improved = test_predictions_improved[2]  # 负面情感预测
    
    # 计算原始模型指标
    mse_original = mean_squared_error(y_true_negative, y_pred_original)
    mae_original = mean_absolute_error(y_true_negative, y_pred_original)
    r2_original = r2_score(y_true_negative, y_pred_original)
    
    # 计算改进模型指标
    mse_improved = mean_squared_error(y_true_negative, y_pred_improved)
    mae_improved = mean_absolute_error(y_true_negative, y_pred_improved)
    r2_improved = r2_score(y_true_negative, y_pred_improved)
    
    # 计算性能提升百分比
    mse_improvement = ((mse_original - mse_improved) / mse_original) * 100
    mae_improvement = ((mae_original - mae_improved) / mae_original) * 100
    r2_improvement = ((r2_improved - r2_original) / abs(r2_original)) * 100 if r2_original != 0 else 0
    
    # 7. 输出结果表格
    print("\n" + "="*60)
    print("模型预测性能对比结果")
    print("="*60)
    
    print(f"{'模型':<20} {'MSE':<12} {'MAE':<12} {'R²':<12}")
    print("-" * 60)
    print(f"{'经典SIR模型':<20} {mse_original:<12.4f} {mae_original:<12.4f} {r2_original:<12.4f}")
    print(f"{'动态耦合SIR模型':<20} {mse_improved:<12.4f} {mae_improved:<12.4f} {r2_improved:<12.4f}")
    print("-" * 60)
    print(f"{'性能提升':<20} {mse_improvement:<12.1f}% {mae_improvement:<12.1f}% {r2_improvement:<12.1f}%")
    print("="*60)
    
    
    # 9. 分析动态影响系数
    print(f"\n动态影响系数分析:")
    print(f"正面情感动态系数 k_p = {params_improved[5]:.4f}")
    print(f"负面情感动态系数 k_n = {params_improved[6]:.4f}")
    
    if params_improved[5] > 0 and params_improved[6] > 0:
        print("✓ 两个动态系数均为正值，验证了互动量增加会促进情感传播的理论假设")
    
    coeff_diff = abs(params_improved[5] - params_improved[6])
    if coeff_diff < 0.02:  # 差异小于0.02认为相近
        print("✓ 正面和负面情感的动态系数较为接近，表明两种情感对外部刺激具有相似的敏感性")
    


    return {
        'original': {'mse': mse_original, 'mae': mae_original, 'r2': r2_original},
        'improved': {'mse': mse_improved, 'mae': mae_improved, 'r2': r2_improved},
        'improvement': {'mse': mse_improvement, 'mae': mae_improvement, 'r2': r2_improvement},
        'dynamic_coeffs': {'k_p': params_improved[5], 'k_n': params_improved[6]}
    }

if __name__ == "__main__":
    performance_metrics = calculate_model_performance_metrics()
