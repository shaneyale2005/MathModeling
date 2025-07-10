import pandas as pd
import numpy as np
from config import config

def load_and_preprocess_data():
    df = pd.read_csv(config.DATA_PATH, parse_dates=['publish_time'], encoding='gbk')
    print(f"原始数据量: {len(df)}条")

    original_len = len(df)    
    confidence_threshold = df['confidence'].quantile(config.CONFIDENCE_QUANTILE_THRESHOLD)
    print(f"\n根据设定的 {config.CONFIDENCE_QUANTILE_THRESHOLD:.0%} 分位数，计算出的置信度阈值为: {confidence_threshold:.4f}")
    df = df[df['confidence'] >= confidence_threshold]
    print(f"通过分位数检验移除了 {original_len - len(df)} 条低置信度数据。")
    
    df['sentiment_label'] = df.apply(
        lambda x: '正面' if x['sentiment_score'] > config.POSITIVE_THRESHOLD else 
                 ('负面' if x['sentiment_score'] < config.NEGATIVE_THRESHOLD else '中性'), 
        axis=1
    )
    
    criteria_labels = ['quote_count', 'comment_count', 'repost_count']
    pairwise = {
        ('quote_count',   'comment_count'): 3,
        ('quote_count',   'repost_count'):  5,
        ('comment_count', 'repost_count'):  3
    }
    n = len(criteria_labels)
    A = np.ones((n, n), dtype=float)
    for i, ci in enumerate(criteria_labels):
        for j, cj in enumerate(criteria_labels):
            if i == j:
                A[i, j] = 1.0
            else:
                if (ci, cj) in pairwise:
                    A[i, j] = pairwise[(ci, cj)]
                elif (cj, ci) in pairwise:
                    A[i, j] = 1.0 / pairwise[(cj, ci)]
    geom_means = np.prod(A, axis=1) ** (1.0 / n)
    weights = geom_means / geom_means.sum()
    weight_dict = dict(zip(criteria_labels, weights))

    print(" AHP 计算得到的权重：")
    for k, v in weight_dict.items():
        print(f"  {k:13s} : {v:.4f}")

    df = df.copy()
    df['total_interaction'] = (
        df['quote_count']   * weight_dict['quote_count'] +
        df['comment_count'] * weight_dict['comment_count'] +
        df['repost_count']  * weight_dict['repost_count']
    )
    
    print(f"清洗后数据量: {len(df)}条")
    return df
