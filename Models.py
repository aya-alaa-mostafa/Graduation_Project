import numpy as np
import optuna as optuna
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from imblearn import over_sampling, under_sampling, pipeline
from sklearn import ensemble, svm, linear_model, neural_network
from sklearn import impute, feature_selection, preprocessing, model_selection
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from export import export_model
df = pd.read_csv('data/cyp3a4_labeled_data.csv', low_memory=False)
features = df.drop(['smile', 'label'], axis=1)
smile_features = np.load('data/cyp3a4_smile_features.npy', allow_pickle=True)
smile_structure = np.load('data/cyp3a4_smile_structure.npy', allow_pickle=True)
labels = df.label.values.reshape(-1)
db_path = 'sqlite:///data/p450_ml.db'
best_scores = {'Logistic Regression': [], 'Random Forest': [], 'Support Vector Machine': [],
'Neural Network': [], 'SMILE Auto-Extractor': [], 'Structure Auto-Extractor': []}
def create_pipe(trial):
    pipe = []
    pipe.append(impute.SimpleImputer())
    pipe.append(preprocessing.MinMaxScaler())
    pipe.append(feature_selection.VarianceThreshold(trial.suggest_uniform('var_thresh', 0, 0.25)))
    balance = trial.suggest_int('balance', 0, 2)
    if balance == 2:
        pipe.append(over_sampling.SMOTE())
    elif balance == 1:
        pipe.append(under_sampling.RandomUnderSampler())
    return pipe
def log_score(scores, name):
    try:
        if scores.mean() > study.best_value:
             best_scores[name] = scores
    except ValueError:
        best_scores[name] = scores
def objective(trial):

    pipe = create_pipe(trial)
    pipe.append(linear_model.LogisticRegression(C=trial.suggest_loguniform('c', 1e-5, 1e5)))
    classifier = pipeline.make_pipeline(*pipe)
    scores = model_selection.cross_val_score(classifier, features, labels, scoring='accuracy',
                                             cv=model_selection.StratifiedKFold(3, shuffle=True), n_jobs=3)
    log_score(scores, 'Logistic Regression')
    export_model(scores.mean(), classifier, features, labels, 'models/lr-model.joblib', study)
    return scores.mean()
study = optuna.create_study(study_name='lr', storage=db_path, direction='maximize', load_if_exists=True)
study.optimize(objective, n_trials=50)
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig = optuna.visualization.plot_slice(study)
fig.show()
def objective(trial):
    pipe = create_pipe(trial)
    pipe.append(ensemble.RandomForestClassifier(max_features=trial.suggest_loguniform('max_features', 0.01,
                                                                                      1), n_estimators=trial.suggest_int('n_estimators', 1, 1000)))
    classifier = make_pipeline(*pipe)
    scores = model_selection.cross_val_score(classifier, features, labels, scoring='accuracy',
                                             cv=model_selection.StratifiedKFold(3, shuffle=True), n_jobs=3)
    log_score(scores, 'Random Forest')
    export_model(scores.mean(), classifier, features, labels, 'models/rf-model.joblib', study)
    return scores.mean()
study = optuna.create_study(study_name='rf', storage=db_path, direction='maximize', load_if_exists=True)
study.optimize(objective, n_trials=50)
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig = optuna.visualization.plot_slice(study)
fig.show()

def objective(trial):

    pipe = create_pipe(trial)
    pipe.append(svm.SVC(C=trial.suggest_loguniform('c', 1e-5, 1e5), gamma=trial.suggest_loguniform('gamma',
                                                                                                   1e-5, 1e5), probability=True))
    classifier = make_pipeline(*pipe)
    scores = model_selection.cross_val_score(classifier, features, labels, scoring='accuracy',
                                             cv=model_selection.StratifiedKFold(3, shuffle=True), n_jobs=3)
    log_score(scores, 'Support Vector Machine')
    export_model(scores.mean(), classifier, features, labels, 'models/svm-model.joblib', study)
    return scores.mean()
study = optuna.create_study(study_name='svm', storage=db_path, direction='maximize', load_if_exists=True)
study.optimize(objective, n_trials=15)
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig = optuna.visualization.plot_slice(study)
fig.show()
def objective(trial):

    pipe = create_pipe(trial)
    layers = []
    for i in range(trial.suggest_int('layers', 1, 3)):
        n_units = trial.suggest_int(f'units_{i}', 1, 300)
        layers.append(n_units)
    pipe.append(neural_network.MLPClassifier(hidden_layer_sizes=tuple(layers),
                                             alpha=trial.suggest_loguniform('alpha', 1e-10, 1e10)))
    classifier = make_pipeline(*pipe)
    scores = model_selection.cross_val_score(classifier, features, labels, scoring='accuracy',
                                             cv=model_selection.StratifiedKFold(3, shuffle=True), n_jobs=3)
    log_score(scores, 'Neural Network')
    export_model(scores.mean(), classifier, features, labels, 'models/nn-model.joblib', study)
    return scores.mean()
study = optuna.create_study(study_name='nn', storage=db_path, direction='maximize', load_if_exists=True)
study.optimize(objective, n_trials=50)
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig = optuna.visualization.plot_slice(study)
fig.show()
def build_cnn_model_1d(cnn_layers=(64, 3, 1, 0.4), dense_layers=(32, 0.4), learning_rate=0.001,shape=(250, 28)):
    model = Sequential()
    model.add(Input(shape=shape))
    for layer in cnn_layers:
        model.add(Conv1D(filters=layer[0], kernel_size=layer[1], strides=layer[2], activation='relu'))
        model.add(BatchNormalization(axis=2))
        if layer[3] > 0:
            model.add(Dropout(layer[3]))
    model.add(Flatten())
    for layer in dense_layers:
        model.add(Dense(units=layer[0], activation='relu'))
        model.add(BatchNormalization(axis=1))
        if layer[1] > 0:
            model.add(Dropout(layer[1]))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model
def objective(trial):
    cnn_layers = []
    for i in range(trial.suggest_int('cnn_layers', 1, 2)):
        filters = trial.suggest_int(f'filter_{i}', 1, 50)
        kernel = trial.suggest_int(f'kernel_{i}', 1, 5)
        stride = trial.suggest_int(f'stride_{i}', 1, 5)
        dropout = trial.suggest_uniform(f'dropout_cnn_{i}', 0.2, 0.5)
        cnn_layers.append((filters, kernel, stride, dropout))
    dense_layers = []
    for i in range(trial.suggest_int('dense_layers', 0, 1)):
        n_units = trial.suggest_int(f'unit_{i}', 1, 50)
        dropout = trial.suggest_uniform(f'dropout_nn_{i}', 0.2, 0.5)
        dense_layers.append((n_units, dropout))
        classifier = KerasClassifier(build_fn=build_cnn_model_1d, epochs=100, batch_size=32, learning_rate=0.0005,
                                     verbose=0, cnn_layers=tuple(cnn_layers), dense_layers=tuple(dense_layers))
        scores = model_selection.cross_val_score(classifier, smile_features, labels, scoring='accuracy',
                                         cv=model_selection.StratifiedKFold(3, shuffle=True))
    log_score(scores, 'SMILE Auto-Extractor')
    export_model(scores.mean(), classifier, smile_features, labels, 'models/cnn-model.h5', study)
    return scores.mean()

study = optuna.create_study(study_name='cnn', storage=db_path, direction='maximize', load_if_exists=True)
study.optimize(objective, n_trials=50)
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig = optuna.visualization.plot_slice(study)
fig.show()
def build_cnn_model_2d(cnn_layers=(64, 3, 1, 0.4), dense_layers=(32, 0.4), learning_rate=0.001,shape=(100, 100, 2)):

    model = Sequential()
    model.add(Input(shape=shape))
    for layer in cnn_layers:
        model.add(Conv2D(filters=layer[0], kernel_size=layer[1], strides=layer[2], padding="same",activation='relu'))
        model.add(BatchNormalization(axis=2))
        if layer[3] > 0:
            model.add(Dropout(layer[3]))
    model.add(Flatten())
    for layer in dense_layers:
        model.add(Dense(units=layer[0], activation='relu'))
        model.add(BatchNormalization(axis=1))
    if layer[1] > 0:
        model.add(Dropout(layer[1]))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def objective(trial):
    cnn_layers = []
    for i in range(trial.suggest_int('cnn_layers', 1, 6)):
        filters = trial.suggest_int(f'filter_{i}', 1, 20)
        kernel = trial.suggest_int(f'kernel_{i}', 1, 25)
        stride = trial.suggest_int(f'stride_{i}', 1, 5)
        dropout = trial.suggest_uniform(f'dropout_cnn_{i}', 0.2, 0.7)
        cnn_layers.append((filters, kernel, stride, dropout))
    dense_layers = []
    for i in range(trial.suggest_int('dense_layers', 0, 1)):
        n_units = trial.suggest_int(f'unit_{i}', 75, 250)
        dropout = trial.suggest_uniform(f'dropout_nn_{i}', 0, 0.5)
        dense_layers.append((n_units, dropout))
    classifier = KerasClassifier(build_fn=build_cnn_model_2d, epochs=75, batch_size=64, learning_rate=0.001,verbose=0, cnn_layers=tuple(cnn_layers), dense_layers=tuple(dense_layers))
    scores = model_selection.cross_val_score(classifier, smile_structure, labels, scoring='accuracy',cv=model_selection.StratifiedKFold(3, shuffle=True))
    log_score(scores, 'Structure Auto-Extractor')
    export_model(scores.mean(), classifier, smile_structure, labels, 'models/2d-cnn-model.h5', study)
    return scores.mean()


study = optuna.create_study(study_name='2d-cnn', storage=db_path, direction='maximize',load_if_exists=True)study.optimize(objective, n_trials=50)
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig = optuna.visualization.plot_slice(study)
fig.show()
scores = list(best_scores.values())[:4]
labels = list(best_scores.keys())[:4]
plt.figure(figsize=(6, 3), dpi=300)
plt.rcParams.update({'font.size': 7})
plt.xticks(rotation=0)
plt.ylim(0.74, 0.86)
plt.ylabel('3-Fold Cross-Validation Accuracy')
plt.boxplot(scores, labels=labels, boxprops={"linewidth": 0.8},medianprops={"color": 'black', "linewidth": 0.8},whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8})
for i in range(len(scores)):
    y = scores[i]
    x = np.random.normal(i+0.7, 0, size=len(y))
    plt.plot(x, y, 'k.', markersize=2)
plt.savefig('images/cv_accuracies_standard.svg')
scores = list(best_scores.values())
scores = [scores[i] for i in [1, 4, 5]]
labels = list(best_scores.keys())
labels = [labels[i] for i in [1, 4, 5]]
plt.figure(figsize=(6, 3), dpi=300)
plt.rcParams.update({'font.size': 7})
plt.xticks(rotation=0)
plt.ylim(0.74, 0.86)
plt.ylabel('3-Fold Cross-Validation Accuracy')
plt.boxplot(scores, labels=labels, boxprops={"linewidth": 0.8},medianprops={"color": 'black', "linewidth": 0.8},whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8})
for i in range(len(scores)):
    y = scores[i]
    x = np.random.normal(i+0.7, 0, size=len(y))
    plt.plot(x, y, 'k.', markersize=2)
plt.savefig('images/cv_accuracies_auto_extract.svg')


study = optuna.load_study(study_name='lr', storage=db_path)
study.best_params
study = optuna.load_study(study_name='rf', storage=db_path)
study.best_params
study = optuna.load_study(study_name='svm', storage=db_path)
study.best_params
study = optuna.load_study(study_name='nn', storage=db_path)
study.best_params
study = optuna.load_study(study_name='cnn', storage=db_path)
study.best_params
study = optuna.load_study(study_name='2d-cnn', storage=db_path)
study.best_params