import json
import platform
import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                            recall_score, f1_score, confusion_matrix)

DATA = [[1.0, 0.2, 0.1, 0.8, 0.1, 1.0],
 [2.0, 0.3, 0.2, 0.7, 0.2, 1.0],
 [3.0, 0.4, 0.15, 0.9, 0.15, 1.0],
 [4.0, 0.5, 0.3, 0.8, 0.1, 1.0],
 [5.0, 0.6, 0.25, 0.75, 0.2, 1.0],
 [6.0, 0.7, 0.35, 0.85, 0.1, 1.0],
 [7.0, 0.8, 0.4, 0.9, 0.15, 1.0],
 [8.0, 0.25, 0.5, 0.7, 0.4, 1.0],
 [9.0, 0.35, 0.45, 0.8, 0.35, 1.0],
 [10.0, 0.45, 0.55, 0.75, 0.45, 1.0],
 [11.0, 0.55, 0.5, 0.85, 0.3, 1.0],
 [12.0, 0.65, 0.6, 0.8, 0.4, 1.0],
 [13.0, 0.2, 0.8, 0.4, 0.75, 0.0],
 [14.0, 0.3, 0.7, 0.35, 0.8, 0.0],
 [15.0, 0.4, 0.85, 0.45, 0.7, 0.0],
 [16.0, 0.5, 0.75, 0.3, 0.85, 0.0],
 [17.0, 0.6, 0.9, 0.4, 0.8, 0.0],
 [18.0, 0.7, 0.8, 0.35, 0.9, 0.0],
 [19.0, 0.8, 0.85, 0.45, 0.75, 0.0],
 [20.0, 0.9, 0.7, 0.3, 0.85, 0.0],
 [21.0, 0.25, 0.9, 0.5, 0.8, 0.0],
 [22.0, 0.45, 0.95, 0.4, 0.9, 0.0],
 [23.0, 0.65, 0.85, 0.3, 0.95, 0.0],
 [24.0, 0.85, 0.9, 0.35, 0.85, 0.0]]

def run():
    data = np.array(DATA, dtype=float)
    ids = data[:, 0].astype(int)
    X, y = data[:, 1:5], data[:, 5].astype(int)
    X_train, X_test = X[:18], X[18:]
    y_train, y_test = y[:18], y[18:]
    models = {
        'DecisionTree': DecisionTreeClassifier(max_depth=2, random_state=42),
        'LogisticRegression': make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000,
                               random_state=42)),
    }
    # Өзгеретін жалғыз гиперпараметр: жасырын қабат нейрондарының саны.
    for width in (4, 8, 16):
        models[f'MLP_{width}'] = make_pipeline(
            StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(width,), activation='relu',
                solver='lbfgs', alpha=0.0001, max_iter=5000,
                tol=0.0001, random_state=42))
    results = {}
    for name, model in models.items()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        results[name] = {
            'accuracy': float(accuracy_score(y_test, prediction)),
            'precision_stable_1': float(precision_score(
                y_test, prediction, pos_label=1, zero_division=0)),
            'recall_stable_1': float(recall_score(
                y_test, prediction, pos_label=1, zero_division=0)),
            'f1_stable_1': float(f1_score(
                y_test, prediction, pos_label=1, zero_division=0)),
            'confusion_matrix_labels_0_1': confusion_matrix(
                y_test, prediction, labels=[0, 1]).tolist(),
            'predictions': prediction.astype(int).tolist(),
            'p_stable': model.predict_proba(X_test)[:, 1].tolist(),
            'train_accuracy': float(model.score(X_train, y_train)),
        }
        if name.startswith('MLP'):
            results[name]['iterations'] = int(model[-1].n_iter_)
            results[name]['loss'] = float(model[-1].loss_)
    train_means = {
        str(cls): X_train[y_train == cls].mean(axis=0).tolist()
        for cls in (0, 1)
    }
    all_means = {
        str(cls): X[y == cls].mean(axis=0).tolist() for cls in (0, 1)
    }
    tree = models['DecisionTree']
    output = {
        'versions': {'python': platform.python_version(),
                     'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'train_ids': ids[:18].tolist(), 'test_ids': ids[18:].tolist(),
        'y_test': y_test.tolist(),
        'feature_order': ['Vibration', 'Slip', 'SurfaceGrip', 'ObstacleRate'],
        'train_class_means': train_means,
        'all_class_means': all_means,
        'tree_feature_index': int(tree.tree_.feature[0]),
        'tree_threshold': float(tree.tree_.threshold[0]),
        'results': results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print('\nӘр тест жағдайын тексеру:')
    for i, sample_id in enumerate(ids[18:]):
        print(f'ID={sample_id}; нақты={y_test[i]}; '
              f'DT={results["DecisionTree"]["predictions"][i]}; '
              f'MLP4={results["MLP_4"]["predictions"][i]}; '
              f'MLP8={results["MLP_8"]["predictions"][i]}; '
              f'MLP16={results["MLP_16"]["predictions"][i]}')
    print('\nЕскерту: тестте Stable=1 үлгілері жоқ. '
          'Бұл класс үшін precision/recall/F1 математикалық '
          'түрде анықталмайды; zero_division=0 оларды 0 деп береді. '
          'Бұл жалпы жалпылауды дәлелдемейді. '
          'Матрица жолдары — нақты, бағандары — болжанған [0,1].')
    return output

if __name__ == '__main__':
    run()

