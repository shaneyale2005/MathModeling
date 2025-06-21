# warning_system.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def train_warning_model(X, y, test_size=0.2):
    """训练预警分类模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    # 训练随机森林模型
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"预警模型准确率: {accuracy:.4f}")
    print("混淆矩阵:")
    print(cm)
    
    return model
