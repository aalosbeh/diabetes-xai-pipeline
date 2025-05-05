from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_mlp(X_train_scaled, y_train_bal):
    clf = MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=100, random_state=42)
    clf.fit(X_train_scaled, y_train_bal)
    return clf

def evaluate_model(clf, X_test_scaled, y_test):
    y_pred = clf.predict(X_test_scaled)
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'conf_matrix': confusion_matrix(y_test, y_pred)
    }
    return results