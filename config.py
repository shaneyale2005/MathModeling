# config.py
class Config:

    DATA_PATH = './dataset/data.csv'
    
    # 情感标签修正阈值
    POSITIVE_THRESHOLD = 0.3
    NEGATIVE_THRESHOLD = -0.3
    
    # 时间窗口配置
    TIME_WINDOW = '6h'  # 6小时时间窗口
    
    # 模型训练配置
    TRAIN_SIZE = 0.7  # 训练集比例
    
    # 预警阈值配置
    NEGATIVE_RATIO_THRESHOLD = 0.4
    SENTIMENT_CHANGE_THRESHOLD = -0.15
    INTERACTION_CHANGE_THRESHOLD = 500
    
    # 可视化输出路径
    OUTPUT_DIR = 'results'
    
config = Config()
