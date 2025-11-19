import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

# ========= НАСТРОЙКИ (МЕНЯТЬ ПОД ЗАДАЧУ) =========
TRAIN_PATH = "train.csv"               # TODO: путь к train
TEST_PATH = "test.csv"                 # TODO: путь к test
SAMPLE_SUB_PATH = "sample_submission.csv"  # TODO: путь к sample_submission

TARGET_COL = "transaction_sum"         # TODO: ИМЯ ТАРГЕТА
ID_COL = "location_id"                 # TODO: ИМЯ ID В test и sample_submission

VAL_SIZE = 0.2
RANDOM_STATE = 42


def mae_log10(y_true, y_pred):
    """MAE на логарифмах (если метрика как в задаче A)."""
    y_true_log = np.log10(y_true + 1)
    y_pred_log = np.log10(y_pred + 1)
    return mean_absolute_error(y_true_log, y_pred_log)


def main():
    # 1. Читаем данные
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print("[INFO] Train shape:", train.shape)
    print("[INFO] Test  shape:", test.shape)
    print("[INFO] Columns:", train.columns.tolist())

    # 2. Разделяем на X, y
    y = train[TARGET_COL]
    X = train.drop(columns=[TARGET_COL])

    # Таргет логарифмируем (при необходимости)
    use_log_target = False  # TODO: если нужно, поменять на True
    if use_log_target:
        y = np.log1p(y)

    # 3. Определяем категориальные фичи (объектные столбцы)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print("[INFO] Categorical columns:", cat_cols)

    # 4. Сплит на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )

    # 5. Модель CatBoost
    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.07,
        iterations=1500,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=RANDOM_STATE,
        verbose=200
    )

    print("[INFO] Training...")
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_cols
    )

    # 6. Валидация
    val_pred = model.predict(X_val)

    if use_log_target:
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

    # 7. Обучаемся на всех данных и делаем сабмит
    print("[INFO] Training on full train...")
    model.fit(
        X, y,
        cat_features=cat_cols,
        verbose=False
    )

    print("[INFO] Predicting on test...")
    test_pred = model.predict(test)

    if use_log_target:
        test_pred = np.expm1(test_pred)

    submission = sample_sub.copy()
    submission[TARGET_COL] = test_pred

    OUT_PATH = "submission.csv"
    submission.to_csv(OUT_PATH, index=False)
    print(f"[INFO] Saved submission to {OUT_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
