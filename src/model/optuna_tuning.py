import optuna
from .train_model import train_model

def objective(trial, train_dataset, validation_dataset):
    pooling = trial.suggest_categorical("pooling", ["avg", "max"])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    base_learning_rate = trial.suggest_float("base_learning_rate", 1e-5, 1e-2, log=True)

    _, history = train_model(train_dataset, validation_dataset, pooling, dropout, base_learning_rate)
    last_val_accuracy = history.history['val_accuracy'][-1]
    return last_val_accuracy

def tune_hyperparameters(train_dataset, validation_dataset, n_trials=25):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_dataset, validation_dataset), n_trials=n_trials)
    return study
