# data_loader.py
import pandas as pd
from config import config

def load_and_preprocess_data():
    """加载并预处理原始数据"""
    # 加载数据，【重要】使用gbk编码
    df = pd.read_csv(config.DATA_PATH, parse_dates=['publish_time'], encoding='gbk')
    print(f"原始数据量: {len(df)}条")

    # 移除低置信度数据
    original_len = len(df)    
    confidence_threshold = df['confidence'].quantile(config.CONFIDENCE_QUANTILE_THRESHOLD)
    print(f"\n根据设定的 {config.CONFIDENCE_QUANTILE_THRESHOLD:.0%} 分位数，计算出的置信度阈值为: {confidence_threshold:.4f}")
    df = df[df['confidence'] >= confidence_threshold]
    print(f"通过分位数检验移除了 {original_len - len(df)} 条低置信度数据。")
    
    # 修正情绪标签
    df['sentiment_label'] = df.apply(
        lambda x: '正面' if x['sentiment_score'] > config.POSITIVE_THRESHOLD else 
                 ('负面' if x['sentiment_score'] < config.NEGATIVE_THRESHOLD else '中性'), 
        axis=1
    )
    
    # 计算总互动量
    df['total_interaction'] = df['repost_count'] + df['comment_count'] + df['quote_count']
    
    print(f"清洗后数据量: {len(df)}条")
    return df