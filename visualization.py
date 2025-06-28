# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from config import config

# Set plotting style (no specific font for CJK characters needed now)
plt.rcParams['font.sans-serif'] = [
    'DejaVu Sans',          # Matplotlib's default sans-serif font
    'Arial',                # Common sans-serif font
    'sans-serif'            # Generic sans-serif family
]
plt.rcParams['axes.unicode_minus'] = False # Still good practice for proper minus sign display
sns.set_style("whitegrid")
sns.set_palette("husl")

def ensure_output_dir():
    """Ensure the output directory exists"""
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

def plot_sentiment_trends(time_features):
    """Plots sentiment trends over time."""
    ensure_output_dir()
    plt.figure(figsize=(16, 12))
    
    # Subplot 1: Average Sentiment and Sentiment Change Rate
    plt.subplot(3, 1, 1)
    plt.plot(time_features['time_window'], time_features['avg_sentiment'], 'o-', 
             label='Average Sentiment Score', linewidth=2, markersize=6)
    plt.plot(time_features['time_window'], time_features['sentiment_change'], 'x--', 
             label='Sentiment Change Rate', linewidth=2, markersize=8)
    plt.axhline(y=config.POSITIVE_THRESHOLD, color='green', linestyle=':', alpha=0.7, 
                label=f'Positive Threshold ({config.POSITIVE_THRESHOLD})')
    plt.axhline(y=config.NEGATIVE_THRESHOLD, color='red', linestyle=':', alpha=0.7, 
                label=f'Negative Threshold ({config.NEGATIVE_THRESHOLD})')
    plt.ylabel('Sentiment Indicators')
    plt.title('Sentiment Evolution Trend Analysis', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Positive and Negative Ratio Changes
    plt.subplot(3, 1, 2)
    plt.plot(time_features['time_window'], time_features['positive_ratio'], 's-', 
             label='Positive Ratio', linewidth=2, markersize=6)
    plt.plot(time_features['time_window'], time_features['negative_ratio'], 'd-', 
             label='Negative Ratio', linewidth=2, markersize=6)
    plt.axhline(y=config.NEGATIVE_RATIO_THRESHOLD, color='orange', linestyle=':', 
                alpha=0.7, label=f'Negative Alert Threshold ({config.NEGATIVE_RATIO_THRESHOLD})')
    plt.ylabel('Sentiment Ratio')
    plt.title('Positive/Negative Sentiment Ratio Changes', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: User Interaction Volume
    plt.subplot(3, 1, 3)
    bars = plt.bar(time_features['time_window'], time_features['total_interaction'], 
                   width=0.02, color='purple', alpha=0.7, label='Total Interaction Volume')
    plt.xlabel('Time Window')
    plt.ylabel('Interaction Volume')
    plt.title('User Interaction Volume Changes', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    heights = [bar.get_height() for bar in bars]
    if heights:
        max_height = max(heights)
        for i, bar in enumerate(bars):
            # Ensure text labels appear above bars and adjust offset based on bar height
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_height*0.01,
                    f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/sentiment_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Sentiment trends plot saved.")

def plot_model_validation_comparison(time_features, pred_original, pred_improved, train_size):
    """Plots comparison of prediction results between original and improved models."""
    ensure_output_dir()
    plt.figure(figsize=(18, 12))
    
    # Subplot 1: Positive Ratio Prediction Comparison
    plt.subplot(2, 1, 1)
    plt.plot(time_features['time_window'], time_features['positive_ratio'], 'o-', 
             label='True Value (Positive)', color='green', linewidth=2, markersize=5, alpha=0.8)
    plt.plot(time_features['time_window'], pred_original[1], '--', 
             label='Original Model Prediction', color='gray', linewidth=2, alpha=0.8)
    plt.plot(time_features['time_window'], pred_improved[1], '-', 
             label='Improved Model Prediction', color='darkgreen', linewidth=2.5)
    plt.axvline(x=time_features['time_window'].iloc[train_size], 
                color='r', linestyle='--', linewidth=2, alpha=0.8, label='Train/Test Split')
    plt.ylabel('Positive Sentiment Ratio')
    plt.title('Model Prediction Comparison - Positive Sentiment', fontsize=16, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 2: Negative Ratio Prediction Comparison
    plt.subplot(2, 1, 2)
    plt.plot(time_features['time_window'], time_features['negative_ratio'], 'o-', 
             label='True Value (Negative)', color='red', linewidth=2, markersize=5, alpha=0.8)
    plt.plot(time_features['time_window'], pred_original[2], '--', 
             label='Original Model Prediction', color='gray', linewidth=2, alpha=0.8)
    plt.plot(time_features['time_window'], pred_improved[2], '-', 
             label='Improved Model Prediction', color='darkred', linewidth=2.5)
    plt.axvline(x=time_features['time_window'].iloc[train_size], 
                color='r', linestyle='--', linewidth=2, alpha=0.8, label='Train/Test Split')
    plt.xlabel('Time Window')
    plt.ylabel('Negative Sentiment Ratio')
    plt.title('Model Prediction Comparison - Negative Sentiment', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/model_validation_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Model prediction comparison plot saved.")
