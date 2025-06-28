# dynamics_model.py
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from config import config

# --- 原始模型 ---
def sentiment_dynamics(t, y, beta_p, beta_n, alpha_p, alpha_n, gamma):
    S, I_p, I_n, R = y
    dSdt = -beta_p * I_p * S - beta_n * I_n * S + gamma * R
    dIpdt = beta_p * I_p * S - alpha_p * I_p
    dIndt = beta_n * I_n * S - alpha_n * I_n
    dRdt = alpha_p * I_p + alpha_n * I_n - gamma * R
    return [dSdt, dIpdt, dIndt, dRdt]

def fit_dynamics_model(train_features):
    Ip0 = train_features['positive_ratio'].iloc[0]
    In0 = train_features['negative_ratio'].iloc[0]
    S0 = 1 - Ip0 - In0
    R0 = 0
    y0 = [S0, Ip0, In0, R0]
    t_train = np.arange(len(train_features))
    
    def model_loss(params):
        beta_p, beta_n, alpha_p, alpha_n, gamma = params
        sol = solve_ivp(
            lambda t, y: sentiment_dynamics(t, y, beta_p, beta_n, alpha_p, alpha_n, gamma),
            [t_train[0], t_train[-1]], y0, t_eval=t_train, method='RK45'
        )
        if sol.status != 0: return np.inf
        Ip_error = np.mean((sol.y[1] - train_features['positive_ratio'])**2)
        In_error = np.mean((sol.y[2] - train_features['negative_ratio'])**2)
        return Ip_error + In_error
    
    initial_params = [0.5, 0.5, 0.1, 0.1, 0.05]
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 0.2)]
    result = minimize(model_loss, initial_params, bounds=bounds, method='L-BFGS-B')
    return result.x

# --- 【改进后】的模型 ---
def sentiment_dynamics_improved(t, y, params, time_features_df):
    base_beta_p, base_beta_n, alpha_p, alpha_n, gamma, k_p, k_n = params
    S, I_p, I_n, R = y
    
    current_idx = min(int(round(t)), len(time_features_df) - 1)
    interaction_change = time_features_df['interaction_change'].iloc[current_idx]
    
    interaction_factor = interaction_change / 100000.0
    
    beta_p_t = max(0.01, base_beta_p * (1 + k_p * interaction_factor))
    beta_n_t = max(0.01, base_beta_n * (1 + k_n * interaction_factor))

    dSdt = -beta_p_t * I_p * S - beta_n_t * I_n * S + gamma * R
    dIpdt = beta_p_t * I_p * S - alpha_p * I_p
    dIndt = beta_n_t * I_n * S - alpha_n * I_n
    dRdt = alpha_p * I_p + alpha_n * I_n - gamma * R
    return [dSdt, dIpdt, dIndt, dRdt]

def fit_dynamics_model_improved(train_features):
    Ip0 = train_features['positive_ratio'].iloc[0]
    In0 = train_features['negative_ratio'].iloc[0]
    S0 = 1 - Ip0 - In0
    R0 = 0
    y0 = [S0, Ip0, In0, R0]
    t_train = np.arange(len(train_features))
    
    train_features_reset = train_features.reset_index(drop=True)

    def model_loss(params):
        sol = solve_ivp(
            lambda t, y: sentiment_dynamics_improved(t, y, params, train_features_reset),
            [t_train[0], t_train[-1]], y0, t_eval=t_train, method='RK45'
        )
        if sol.status != 0: return np.inf
            
        Ip_pred, In_pred = sol.y[1], sol.y[2]
        Ip_error = np.mean((Ip_pred - train_features['positive_ratio'])**2)
        In_error = np.mean((In_pred - train_features['negative_ratio'])**2)
        return Ip_error + In_error
    
    initial_params = [0.5, 0.5, 0.1, 0.1, 0.05, 0.1, 0.1]
    bounds = [(0, 2), (0, 2), (0, 1), (0, 1), (0, 0.2), (-5, 5), (-5, 5)]
    
    print("开始优化改进后的动力学模型参数...")
    result = minimize(model_loss, initial_params, bounds=bounds, method='L-BFGS-B')
    
    if not result.success:
        print(f"警告: 动力学模型参数优化可能未收敛。消息: {result.message}")

    return result.x

def predict_full_model(params, time_features, improved=False):
    Ip0 = time_features['positive_ratio'].iloc[0]
    In0 = time_features['negative_ratio'].iloc[0]
    S0 = 1 - Ip0 - In0
    R0 = 0
    y0 = [S0, Ip0, In0, R0]
    t_eval = np.arange(len(time_features))
    
    if improved:
        time_features_reset = time_features.reset_index(drop=True)
        dynamics_func = lambda t, y: sentiment_dynamics_improved(t, y, params, time_features_reset)
    else:
        beta_p, beta_n, alpha_p, alpha_n, gamma = params
        dynamics_func = lambda t, y: sentiment_dynamics(t, y, beta_p, beta_n, alpha_p, alpha_n, gamma)

    sol = solve_ivp(dynamics_func, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, method='RK45')
    
    return sol.y