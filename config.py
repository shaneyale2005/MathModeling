# config.py
class Config:

    DATA_PATH = './dataset/data.csv'
      # 情感标签修正阈值 (基于均值±0.4标准差的统计学优化)
    POSITIVE_THRESHOLD = 0.204
    NEGATIVE_THRESHOLD = -0.194
    
    # 时间窗口配置
    TIME_WINDOW = '6h'  # 6小时时间窗口
      # 模型训练配置 (确保充足训练数据和可靠测试集)
    TRAIN_SIZE = 0.75  # 训练集比例
      # 预警阈值配置 (基于数据分位数的科学设定)
    NEGATIVE_RATIO_THRESHOLD = 0.317  # 75%分位数，识别异常负面情绪聚集
    SENTIMENT_CHANGE_THRESHOLD = -0.063  # 10%分位数，捕获显著情感下降
    INTERACTION_CHANGE_THRESHOLD = 106275  # 75%分位数，识别异常互动增长
    
    # 可视化输出路径
    OUTPUT_DIR = 'results'
    
config = Config()
