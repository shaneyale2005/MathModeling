# data_loader.py
import pandas as pd
from config import config

def load_and_preprocess_data():
    """加载并预处理原始数据"""
    # 加载数据
    df = pd.read_csv(config.DATA_PATH, parse_dates=['publish_time'], encoding='gbk')
    print(f"原始数据量: {len(df)}条")
    print(df.head())

#另一种办法是，将置信度数据的分布“拉近”正态，然后使用删正态分布极端值的办法处理。（在于有无必要在这种地方大费篇章）

    # 移除低置信度数据
    original_len = len(df)    
    # 例如，如果这个值为0.7, 那么所有confidence < 0.7 的数据都将被移除
    confidence_threshold = df['confidence'].quantile(config.CONFIDENCE_QUANTILE_THRESHOLD)
        
    print(f"\n根据设定的 {config.CONFIDENCE_QUANTILE_THRESHOLD:.0%} 分位数，计算出的置信度阈值为: {confidence_threshold:.4f}")
        
        # 2. 根据计算出的阈值过滤 DataFrame
        # 保留所有 confidence 大于或等于该阈值的数据
    df = df[df['confidence'] >= confidence_threshold]
        
    print(f"通过分位数检验移除了 {original_len - len(df)} 条低置信度数据。")
    
    # 修正情绪标签
    df['sentiment_label'] = df.apply(
        lambda x: '正面' if x['sentiment_score'] > config.POSITIVE_THRESHOLD else 
                 ('负面' if x['sentiment_score'] < config.NEGATIVE_THRESHOLD else '中性'), 
        axis=1
    )
    
    # 计算总互动量（此处可加入数学模型，毕竟转发，评论等操作的权重不可能一样）
    df['total_interaction'] = df['repost_count'] + df['comment_count'] + df['quote_count']
    
    print(f"清洗后数据量: {len(df)}条")
    return df