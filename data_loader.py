# data_loader.py
import pandas as pd
import numpy as np
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
    
    #层次分析法 (Analytic Hierarchy Process, AHP) 由 Thomas L. Saaty 提出，核心思想是将复杂问题分解为目标、准则、方案等层次，然后通过两两比较的方式确定各元素的相对重要性。
#构建判断矩阵，转发(R)、评论(C)、引用(Q)，两两比较它们对于“总互动价值”的重要性
#判断矩阵 A
#repost。成本最低。用户认可内容，并愿意用自己的社交信誉为其背书，直接将内容推送给自己的关注者
#Comment:成本较高。用户需要花费时间思考并打字，是深度参与的表现。但其传播范围通常局限于原内容的评论区，对扩大触及新用户的作用小于转发
#Quote:成本最高。用户不仅要转发，还需要组织自己的语言进行二次创作。这既扩大了传播，又增加了内容的丰富度和讨论的深度
#支持“内容增值”和“二次创作”价值的论文
#Boyd, D., Golder, S., & Lotan, G. (2010). "Tweet, tweet, retweet: Conversational aspects of retweeting on Twitter". 2010 43rd Hawaii International Conference on System Sciences.
#它建立了一个经典的“参与度阶梯”或“参与度金字塔”模型。模型指出，从潜水、点赞、到评论、再到发布原创内容，用户的参与成本和对社区的贡献是逐级递增的。您的 Quote > Comment > Repost 模型完全符合这个阶梯理论。“引用”可以被看作是“发布原创内容”的一种形式，成本最高；“评论”是深度互动；而“转发”则是一种相对低成本的分享行为。
    # 计算总互动量
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

    # 打印结果检查
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