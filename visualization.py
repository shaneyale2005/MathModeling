# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import config

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
sns.set_style("whitegrid")

def ensure_output_dir():
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

def plot_sentiment_trends(time_features):
    ensure_output_dir()
    plt.figure(figsize=(16, 10))
    # ... (此部分与您原代码一致，此处省略以节约篇幅)
    plt.subplot(2, 1, 1)
    plt.plot(time_features['time_window'], time_features['avg_sentiment'], 'o-', label='平均情感得分')
    plt.title('情感得分与互动量趋势')
    plt.ylabel('平均情感得分')
    plt.legend()
    
    ax2 = plt.gca().twinx()
    ax2.bar(time_features['time_window'], time_features['total_interaction'], alpha=0.3, color='orange', width=0.02, label='总互动量')
    ax2.set_ylabel('总互动量')
    ax2.legend(loc='upper right')
    
    plt.subplot(2, 1, 2)
    plt.plot(time_features['time_window'], time_features['positive_ratio'], 'g-o', label='正面比例')
    plt.plot(time_features['time_window'], time_features['negative_ratio'], 'r-o', label='负面比例')
    plt.axhline(y=config.NEGATIVE_RATIO_THRESHOLD, color='orange', linestyle=':', label=f'负面预警阈值 ({config.NEGATIVE_RATIO_THRESHOLD})')
    plt.title('正负面情感比例变化')
    plt.xlabel('时间窗口')
    plt.ylabel('情感比例')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/sentiment_trends.png", dpi=300)
    plt.close()
    print("✓ 情感趋势图已保存。")

def plot_model_validation_comparison(time_features, pred_original, pred_improved, train_size):
    """【新】绘制原始模型与改进模型的预测结果对比图"""
    ensure_output_dir()
    plt.figure(figsize=(18, 12))
    
    # --- 正面情感比例对比 ---
    plt.subplot(2, 1, 1)
    plt.plot(time_features['time_window'], time_features['positive_ratio'], 'o-', 
             label='真实值 (正面)', color='green', linewidth=2, markersize=5, alpha=0.7)
    plt.plot(time_features['time_window'], pred_original[1], '--', 
             label='原始模型预测', color='gray', linewidth=2)
    plt.plot(time_features['time_window'], pred_improved[1], '-', 
             label='改进模型预测', color='darkgreen', linewidth=2.5, alpha=0.9)
    plt.axvline(x=time_features['time_window'].iloc[train_size], 
                color='r', linestyle='--', linewidth=2, alpha=0.8, label='训练/测试集分割')
    plt.ylabel('正面情感比例')
    plt.title('模型预测对比 - 正面情感', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # --- 负面情感比例对比 ---
    plt.subplot(2, 1, 2)
    plt.plot(time_features['time_window'], time_features['negative_ratio'], 'o-', 
             label='真实值 (负面)', color='red', linewidth=2, markersize=5, alpha=0.7)
    plt.plot(time_features['time_window'], pred_original[2], '--', 
             label='原始模型预测', color='gray', linewidth=2)
    plt.plot(time_features['time_window'], pred_improved[2], '-', 
             label='改进模型预测', color='darkred', linewidth=2.5, alpha=0.9)
    plt.axvline(x=time_features['time_window'].iloc[train_size], 
                color='r', linestyle='--', linewidth=2, alpha=0.8, label='训练/测试集分割')
    plt.xlabel('时间窗口')
    plt.ylabel('负面情感比例')
    plt.title('模型预测对比 - 负面情感', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/model_validation_comparison.png", dpi=300)
    plt.close()
    print("✓ 模型预测对比图已保存。")