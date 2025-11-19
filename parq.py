import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor

# ===== НАСТРОЙКИ =====

DATA_DIR = "."          # рабочая папка: ./train.parquet, ./test.parquet, ./sample_submission.csv
TRAIN_FILE = "train.parquet"
TEST_FILE = "test.parquet"
SAMPLE_FILE = "sample_submission.csv"

TARGET_COL = "transaction_sum"
ID_COL = "location_id"

VAL_SIZE = 0.2
RANDOM_STATE = 42

USE_LOG_TARGET = False  # если True — модель учит log(1+y)

# =======================


def mae_log10(y_true, y_pred):
    y_true_log = np.log10(y_true + 1)
    y_pred_log = np.log10(y_pred + 1)
    return mean_absolute_error(y_true_log, y_pred_log)


def main():

    # --- Проверяем наличие файлов ---
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    test_path = os.path.join(DATA_DIR, TEST_FILE)
    sample_path = os.path.join(DATA_DIR, SAMPLE_FILE)

    assert os.path.exists(train_path), f"{train_path} not found"
    assert os.path.exists(test_path), f"{test_path} not found"
    assert os.path.exists(sample_path), f"{sample_path} not found"

    print("[INFO] Reading data...")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    sample_sub = pd.read_csv(sample_path)

    print("[INFO] Train shape:", train.shape)
    print("[INFO] Test  shape:", test.shape)

    # --- Разделяем X/y ---
    y = train[TARGET_COL]
    X = train.drop(columns=[TARGET_COL])

    # Лог-таргет (по желанию)
    if USE_LOG_TARGET:
        y = np.log1p(y)

    # Категории
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print("[INFO] Categorical columns:", cat_cols)

    # --- Разделение на train/val ---
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )

    # --- Модель ---
    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.07,
        iterations=1500,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=RANDOM_STATE,
        verbose=200
    )

    print("[INFO] Training model...")
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_cols
    )

    # --- Оценка ---
    val_pred = model.predict(X_val)

    if USE_LOG_TARGET:
        y_val_true = np.expm1(y_val)
        val_pred_true = np.expm1(val_pred)
    else:
        y_val_true = y_val
        val_pred_true = val_pred

    mae = mean_absolute_error(y_val_true, val_pred_true)
    mae_l = mae_log10(y_val_true, val_pred_true)
    score = 1 / (1 + mae_l)

    print(f"[VAL] MAE:       {mae:.4f}")
    print(f"[VAL] MAE_LOG10: {mae_l:.6f}")
    print(f"[VAL] Score:     {score:.6f}")

    # --- Обучаемся на всём train ---
    print("[INFO] Training on full data...")
    model.fit(X, y, cat_features=cat_cols, verbose=False)

    # --- Предсказание test ---
    print("[INFO] Predicting test...")
    test_pred = model.predict(test)

    if USE_LOG_TARGET:
        test_pred = np.expm1(test_pred)

    # --- Submission ---
    sub = sample_sub.copy()
    sub[TARGET_COL] = test_pred

    sub.to_csv("submission.csv", index=False)
    print("[INFO] Saved submission.csv")
    print(sub.head())


if __name__ == "__main__":
    main()
