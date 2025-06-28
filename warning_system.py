# warning_system.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_warning_model(X_train, y_train, X_test, y_test):
    """训练并评估预警分类模型"""
    print("\n训练预警模型(随机森林)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced', # 处理类别不平衡
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"预警模型在测试集上的准确率: {accuracy:.4f}")
    print("预警模型混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print("预警模型分类报告:")
    print(classification_report(y_test, y_pred))
    
    return model