import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.model_selection import StratifiedKFold
import ctypes
import cloudpickle as cp
import gc
import os
import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

import matplotlib.pyplot as plt

import optuna
import optuna-integration[lightgbm]


def amex_metric_np(target: np.ndarray, preds: np.ndarray) -> float:
    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)

def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric_np(y_true, y_pred),
            True)
    
def release_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    
train = pl.read_parquet("../../data/gold/train_data")

feature_names = train.drop(["customer_ID","target"]).columns

X = train.drop(["customer_ID", "target"]).to_numpy()
y = train["target"].to_numpy()

del train
release_memory()

# Hyperparamter Search Space

def lgbm_classifier_search_space(trial):

    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "n_estimators": 2000,

        # tree complexity (centered around best trial)
        "num_leaves": trial.suggest_int("num_leaves", 150, 350), # increased num leaves range from 100-300 to 150-350
        "max_depth": trial.suggest_int("max_depth", 8, 16),

        # learning rate
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.025, 0.06, log=True
        ),
        
        # regularization
        "min_child_samples": trial.suggest_int("min_child_samples",200, 260), # increased minimum child samples lower bound to 200 and upper bound to 260

        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-3, 1.0, log=True
        ),

        # sampling
        "subsample": trial.suggest_float("subsample",0.9, 1.0), # increased lower bound from 0.7 to 0.9
        "subsample_freq": trial.suggest_int("subsample_freq",10, 15), # range changed from 7-12 to 10-15.

        "colsample_bytree": trial.suggest_float("colsample_bytree",0.85, 1.0),

        # regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 0.5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 3.0, log=True),# increased lower bound from 0.1 to 0.3

        "min_split_gain": trial.suggest_float("min_split_gain",0.0, 0.2),

        # imbalance
        "scale_pos_weight": trial.suggest_float("scale_pos_weight",4.0, 7.0),

        # allow near trial 3 but prevent explosion
        "max_bin": trial.suggest_int("max_bin",200, 400),
    }
    
def objective_tree(trial):
    params = lgbm_classifier_search_space(trial)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(iter(skf.split(X, y)))

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "amex")

    model = LGBMClassifier(**params)
    model.fit(
        X[train_idx],
        y[train_idx],
        feature_name=feature_names,
        eval_set=[(X[val_idx], y[val_idx])],
        eval_metric=[lgb_amex_metric],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=100),
            pruning_callback
        ]
    )
    preds = model.predict(X[val_idx], raw_score=True)
    return amex_metric_np(y[val_idx], preds)

# Use MedianPruner — kills trials performing below median at any checkpoint
release_memory()
study_tree = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=100)
)

study_tree.optimize(objective_tree, n_trials=30)

best_params = {
    **lgbm_classifier_search_space(optuna.trial.FixedTrial(study_tree.best_trial.params)),
    **study_tree.best_trial.params
}
print(f"The best parameters after hyperparameter tuning are: {best_params}.")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

final_models = []
scores = []
importances = []
best_params['n_estimators']=5000
print("Training the final models:")
for tr, va in skf.split(X, y):

    model = lgb.LGBMClassifier(**best_params)

    model.fit(
        X[tr],
        y[tr],
        feature_name=feature_names,
        eval_set=[(X[va], y[va])],
        eval_metric=[lgb_amex_metric],
        callbacks=[lgb.early_stopping(100),lgb.log_evaluation(period=100)]
    )

    # Record validation R² score
    preds = model.predict(X[va], raw_score=True)

    importances.append(model.booster_.feature_importance(importance_type="gain") / model.booster_.feature_importance(importance_type="gain").sum())
    
    scores.append(amex_metric_np(y[va],preds))
    final_models.append(model)
        

print("CV AMEX:", np.mean(scores))


# average importance
importance = pd.Series(
    np.mean(importances, axis=0),
    index=feature_names
).sort_values(ascending=False)

top_n = 30

plt.figure(figsize=(10,12))
importance.head(top_n)[::-1].plot(kind="barh")

plt.title("Top Feature Importances (LightGBM)")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()
plt.savefig('feature_importance.png')

release_memory()

SAVE_DIR = "../../models/"
os.makedirs(SAVE_DIR, exist_ok=True)

cp.dump(scores, open(f"{SAVE_DIR}/cv_scores.pkl", "wb"))

for fold, model in enumerate(final_models):
    with open(f"{SAVE_DIR}/lgbm_fold_{fold}.pkl", "wb") as f:
        cp.dump(model, f)
        
del X, y
release_memory()

test = pl.read_parquet("../../data/gold/test_data")

test_ids = test["customer_ID"].to_numpy()
X_test = test.select(pl.exclude("customer_ID")).to_numpy()

del test
release_memory()

test_preds = np.zeros(len(X_test))

for model in final_models:
    test_preds += model.predict(X_test, raw_score=True) / len(final_models)

submission = pd.DataFrame({
    "customer_ID": test_ids,
    "prediction": test_preds
})

submission.to_csv("../../submissions/submission.csv", index=False)

release_memory()