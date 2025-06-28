# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from config import config

# 设置绘图风格，适配中文
# 优先使用在Ubuntu上常见的开源中文字体，然后逐步回退
# 请根据您fc-list的输出结果，选择您系统上确实存在的字体名称
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei',  # 文泉驿微米黑 (常用且推荐)
    'WenQuanYi Zen Hei',    # 文泉驿正黑 (常用且推荐)
    'Noto Sans CJK SC',     # 思源黑体 简体中文 (高质量推荐)
    'DejaVu Sans',          # Matplotlib默认的英文字体，作为非中文部分的备用
    'sans-serif'            # 通用无衬线字体家族，作为最终备用
]
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号 (处理负号乱码)
sns.set_style("whitegrid")
sns.set_palette("husl")

def ensure_output_dir():
    """Ensure the output directory exists"""
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

def plot_sentiment_trends(time_features):
    """绘制情感趋势图 (您提供的版本)"""
    ensure_output_dir()
    plt.figure(figsize=(16, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_features['time_window'], time_features['avg_sentiment'], 'o-', 
             label='平均情感得分', linewidth=2, markersize=6)
    plt.plot(time_features['time_window'], time_features['sentiment_change'], 'x--', 
             label='情感变化率', linewidth=2, markersize=8)
    plt.axhline(y=config.POSITIVE_THRESHOLD, color='green', linestyle=':', alpha=0.7, 
                label=f'正面阈值 ({config.POSITIVE_THRESHOLD})')
    plt.axhline(y=config.NEGATIVE_THRESHOLD, color='red', linestyle=':', alpha=0.7, 
                label=f'负面阈值 ({config.NEGATIVE_THRESHOLD})')
    plt.ylabel('情感指标')
    plt.title('情感演化趋势分析', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_features['time_window'], time_features['positive_ratio'], 's-', 
             label='正面比例', linewidth=2, markersize=6)
    plt.plot(time_features['time_window'], time_features['negative_ratio'], 'd-', 
             label='负面比例', linewidth=2, markersize=6)
    plt.axhline(y=config.NEGATIVE_RATIO_THRESHOLD, color='orange', linestyle=':', 
                alpha=0.7, label=f'负面预警阈值 ({config.NEGATIVE_RATIO_THRESHOLD})')
    plt.ylabel('情感比例')
    plt.title('正/负面情感比例变化', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    bars = plt.bar(time_features['time_window'], time_features['total_interaction'], 
                   width=0.02, color='purple', alpha=0.7, label='总互动量')
    plt.xlabel('时间窗口')
    plt.ylabel('互动量')
    plt.title('用户互动量变化', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    heights = [bar.get_height() for bar in bars]
    if heights:
        max_height = max(heights)
        for i, bar in enumerate(bars):
            # 确保文本标签显示在bar上方，并根据bar高度调整偏移
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_height*0.01,
                    f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/sentiment_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 情感趋势图已保存。")

def plot_model_validation_comparison(time_features, pred_original, pred_improved, train_size):
    """【整合修改】绘制原始模型与改进模型的预测结果对比图"""
    ensure_output_dir()
    plt.figure(figsize=(18, 12))
    
    # 正面比例预测对比
    plt.subplot(2, 1, 1)
    plt.plot(time_features['time_window'], time_features['positive_ratio'], 'o-', 
             label='真实值 (正面)', color='green', linewidth=2, markersize=5, alpha=0.8)
    plt.plot(time_features['time_window'], pred_original[1], '--', 
             label='原始模型预测', color='gray', linewidth=2, alpha=0.8)
    plt.plot(time_features['time_window'], pred_improved[1], '-', 
             label='改进模型预测', color='darkgreen', linewidth=2.5)
    plt.axvline(x=time_features['time_window'].iloc[train_size], 
                color='r', linestyle='--', linewidth=2, alpha=0.8, label='训练/测试集分割')
    plt.ylabel('正面情感比例')
    plt.title('模型预测对比 - 正面情感', fontsize=16, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 负面比例预测对比
    plt.subplot(2, 1, 2)
    plt.plot(time_features['time_window'], time_features['negative_ratio'], 'o-', 
             label='真实值 (负面)', color='red', linewidth=2, markersize=5, alpha=0.8)
    plt.plot(time_features['time_window'], pred_original[2], '--', 
             label='原始模型预测', color='gray', linewidth=2, alpha=0.8)
    plt.plot(time_features['time_window'], pred_improved[2], '-', 
             label='改进模型预测', color='darkred', linewidth=2.5)
    plt.axvline(x=time_features['time_window'].iloc[train_size], 
                color='r', linestyle='--', linewidth=2, alpha=0.8, label='训练/测试集分割')
    plt.xlabel('时间窗口')
    plt.ylabel('负面情感比例')
    plt.title('模型预测对比 - 负面情感', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/model_validation_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 模型预测对比图已保存。")

