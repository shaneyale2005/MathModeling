import pandas as pd
from config import config
import numpy as np

def create_time_features(df):
    df['time_window'] = df['publish_time'].dt.floor(config.TIME_WINDOW)
    time_features = df.groupby('time_window').agg(
        post_count=('content_id', 'count'),
        avg_sentiment=('sentiment_score', 'mean'),
        positive_ratio=('sentiment_label', lambda x: (x == '正面').mean()),
        negative_ratio=('sentiment_label', lambda x: (x == '负面').mean()),
        avg_followers=('user_followers', 'mean'),
        total_interaction=('total_interaction', 'sum')
    ).reset_index()
    time_features['sentiment_change'] = time_features['avg_sentiment'].diff()
    time_features['interaction_change'] = time_features['total_interaction'].diff()
    time_features = time_features.fillna(0)
    return time_features

def create_warning_features(time_features, model_pred_negative=None):
    time_features['crisis'] = np.where(
        (time_features['negative_ratio'] > config.NEGATIVE_RATIO_THRESHOLD) & 
        (time_features['sentiment_change'] < config.SENTIMENT_CHANGE_THRESHOLD) &
        (time_features['interaction_change'] > config.INTERACTION_CHANGE_THRESHOLD),
        1, 0
    )
    features = time_features[['avg_sentiment', 'negative_ratio', 
                              'sentiment_change', 'total_interaction']].copy()
    if model_pred_negative is not None:
        features = features.iloc[:len(model_pred_negative)]
        features['model_pred_negative'] = model_pred_negative
    y = time_features['crisis'].shift(-1).fillna(0).astype(int)
    return features, y
