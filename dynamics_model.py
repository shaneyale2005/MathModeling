# dynamics_model.py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from config import config
#我认为可以加入互动这个创新点，就是正面对中性和负面有影响，负面同样，然后中性只能被影响（然后我们拿一篇心理学方面的文章，佐证我们的想法，即负面和正面人价值会影响，我们创新点就是将该文章运用到舆论分析的数学模型里面）
def sentiment_dynamics(t, y, beta_p, beta_n, alpha_p, alpha_n, gamma):
    """定义情感演化动力学方程"""
    S, I_p, I_n, R = y
    dSdt = -beta_p * I_p * S - beta_n * I_n * S + gamma * R
    dIpdt = beta_p * I_p * S - alpha_p * I_p
    dIndt = beta_n * I_n * S - alpha_n * I_n
    dRdt = alpha_p * I_p + alpha_n * I_n - gamma * R
    return [dSdt, dIpdt, dIndt, dRdt]

def fit_dynamics_model(train_features):
    """训练动力学模型"""
    # 初始化状态变量
    Ip0 = train_features['positive_ratio'].iloc[0]
    In0 = train_features['negative_ratio'].iloc[0]
    S0 = 0.8  # 假设80%初始为中立
    R0 = 1 - S0 - Ip0 - In0
    y0 = [S0, Ip0, In0, R0]
    t_train = np.arange(len(train_features))
    
    # 定义损失函数
    def model_loss(params):
        beta_p, beta_n, alpha_p, alpha_n, gamma = params
        sol = solve_ivp(
            lambda t, y: sentiment_dynamics(t, y, beta_p, beta_n, alpha_p, alpha_n, gamma),
            [t_train[0], t_train[-1]],
            y0,
            t_eval=t_train,
            method='RK45'
        )
        # 计算拟合误差（聚焦正面和负面比例）
        Ip_error = np.mean((sol.y[1] - train_features['positive_ratio'])**2)
        In_error = np.mean((sol.y[2] - train_features['negative_ratio'])**2)
        return Ip_error + In_error
    
    # 参数优化
    initial_params = [0.5, 0.5, 0.1, 0.1, 0.05]  # 初始参数
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 0.2)]  # 参数边界
    result = minimize(model_loss, initial_params, bounds=bounds, method='L-BFGS-B')
    
    return result.x

def predict_full_model(params, time_features):
    """使用优化参数进行完整预测"""
    Ip0 = time_features['positive_ratio'].iloc[0]
    In0 = time_features['negative_ratio'].iloc[0]
    S0 = 0.8
    R0 = 1 - S0 - Ip0 - In0
    y0 = [S0, Ip0, In0, R0]
    
    beta_p, beta_n, alpha_p, alpha_n, gamma = params
    t_eval = np.arange(len(time_features))
    
    sol = solve_ivp(
        lambda t, y: sentiment_dynamics(t, y, beta_p, beta_n, alpha_p, alpha_n, gamma),
        [t_eval[0], t_eval[-1]],
        y0,
        t_eval=t_eval,
        method='RK45'
    )
    
    return sol.y
