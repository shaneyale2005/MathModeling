import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import config

# Set matplotlib to support English and minus sign
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use a common English font
plt.rcParams['axes.unicode_minus'] = False         # Properly display minus sign

def ensure_output_dir():
    """Ensure the output directory exists"""
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

def plot_sentiment_trends(time_features):
    """Plot sentiment trends"""
    ensure_output_dir()
    plt.figure(figsize=(14, 10))
    
    # Sentiment score and change rate
    plt.subplot(3, 1, 1)
    plt.plot(time_features['time_window'], time_features['avg_sentiment'], 'o-', label='Average Sentiment')
    plt.plot(time_features['time_window'], time_features['sentiment_change'], 'x--', label='Sentiment Change Rate')
    plt.ylabel('Sentiment Metrics')
    plt.title('Sentiment Evolution Trends')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Emotion ratios
    plt.subplot(3, 1, 2)
    plt.plot(time_features['time_window'], time_features['positive_ratio'], 's-', label='Positive Ratio')
    plt.plot(time_features['time_window'], time_features['negative_ratio'], 'd-', label='Negative Ratio')
    plt.ylabel('Emotion Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Interaction volume
    plt.subplot(3, 1, 3)
    plt.bar(time_features['time_window'], time_features['total_interaction'], 
           width=0.02, color='purple', alpha=0.7)
    plt.xlabel('Time Window')
    plt.ylabel('Interaction Volume')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/sentiment_trends.png", dpi=300)
    plt.close()

def plot_model_validation(time_features, model_predictions, train_size):
    """Plot model validation results"""
    ensure_output_dir()
    plt.figure(figsize=(14, 10))
    
    # Positive ratio comparison
    plt.subplot(2, 1, 1)
    plt.plot(time_features['time_window'], time_features['positive_ratio'], 'o-', label='Actual', color='green')
    plt.plot(time_features['time_window'], model_predictions[1], '--', label='Predicted', color='darkgreen')
    plt.axvline(x=time_features['time_window'].iloc[train_size], 
                color='r', linestyle='--', label='Train/Test Split')
    plt.ylabel('Positive Sentiment Ratio')
    plt.title('Model Prediction Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Negative ratio comparison
    plt.subplot(2, 1, 2)
    plt.plot(time_features['time_window'], time_features['negative_ratio'], 'o-', label='Actual', color='red')
    plt.plot(time_features['time_window'], model_predictions[2], '--', label='Predicted', color='darkred')
    plt.xlabel('Time Window')
    plt.ylabel('Negative Sentiment Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/model_validation.png", dpi=300)
    plt.close()
