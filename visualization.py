import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from config import config
from scipy import stats

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

def ensure_output_dir():
    """Ensure the output directory exists"""
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

def plot_sentiment_trends(time_features):
    """绘制情感趋势图"""
    ensure_output_dir()
    plt.figure(figsize=(16, 12))
    
    # 情感分数和变化率
    plt.subplot(3, 1, 1)
    plt.plot(time_features['time_window'], time_features['avg_sentiment'], 'o-', 
             label='Average Sentiment Score', linewidth=2, markersize=6)
    plt.plot(time_features['time_window'], time_features['sentiment_change'], 'x--', 
             label='Sentiment Change Rate', linewidth=2, markersize=8)
    plt.axhline(y=config.POSITIVE_THRESHOLD, color='green', linestyle=':', alpha=0.7, 
                label=f'Positive Threshold ({config.POSITIVE_THRESHOLD})')
    plt.axhline(y=config.NEGATIVE_THRESHOLD, color='red', linestyle=':', alpha=0.7, 
                label=f'Negative Threshold ({config.NEGATIVE_THRESHOLD})')
    plt.ylabel('Sentiment Metrics')
    plt.title('Sentiment Evolution Trend Analysis', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 情感比例
    plt.subplot(3, 1, 2)
    plt.plot(time_features['time_window'], time_features['positive_ratio'], 's-', 
             label='Positive Ratio', linewidth=2, markersize=6)
    plt.plot(time_features['time_window'], time_features['negative_ratio'], 'd-', 
             label='Negative Ratio', linewidth=2, markersize=6)
    plt.axhline(y=config.NEGATIVE_RATIO_THRESHOLD, color='orange', linestyle=':', 
                alpha=0.7, label=f'Negative Warning Threshold ({config.NEGATIVE_RATIO_THRESHOLD})')
    plt.ylabel('Sentiment Ratio')
    plt.title('Positive/Negative Sentiment Ratio Change', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 互动量
    plt.subplot(3, 1, 3)
    bars = plt.bar(time_features['time_window'], time_features['total_interaction'], 
                   width=0.02, color='purple', alpha=0.7)
    plt.xlabel('Time Window')
    plt.ylabel('Interaction Volume')
    plt.title('User Interaction Volume Change', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 为最高的几个柱子添加数值标签
    heights = [bar.get_height() for bar in bars]
    max_height = max(heights)
    for i, bar in enumerate(bars):
        if bar.get_height() > max_height * 0.8:  # 只标注最高的20%
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_height*0.01,
                    f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/sentiment_trends.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_validation(time_features, model_predictions, train_size):
    """绘制模型验证结果"""
    ensure_output_dir()
    plt.figure(figsize=(16, 10))
    
    # 正面比例预测对比
    plt.subplot(2, 1, 1)
    plt.plot(time_features['time_window'], time_features['positive_ratio'], 'o-', 
             label='Actual Value', color='green', linewidth=2, markersize=6)
    plt.plot(time_features['time_window'], model_predictions[1], '--', 
             label='Predicted Value', color='darkgreen', linewidth=2)
    plt.axvline(x=time_features['time_window'].iloc[train_size], 
                color='r', linestyle='--', linewidth=2, alpha=0.8, label='Train/Test Split')
    plt.ylabel('Positive Sentiment Ratio')
    plt.title('Model Prediction Validation - Positive Sentiment', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 负面比例预测对比
    plt.subplot(2, 1, 2)
    plt.plot(time_features['time_window'], time_features['negative_ratio'], 'o-', 
             label='Actual Value', color='red', linewidth=2, markersize=6)
    plt.plot(time_features['time_window'], model_predictions[2], '--', 
             label='Predicted Value', color='darkred', linewidth=2)
    plt.axvline(x=time_features['time_window'].iloc[train_size], 
                color='r', linestyle='--', linewidth=2, alpha=0.8, label='Train/Test Split')
    plt.xlabel('Time Window')
    plt.ylabel('Negative Sentiment Ratio')
    plt.title('Model Prediction Validation - Negative Sentiment', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/model_validation.png", dpi=300, bbox_inches='tight')
    plt.close()

