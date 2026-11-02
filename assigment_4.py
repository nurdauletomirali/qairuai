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
