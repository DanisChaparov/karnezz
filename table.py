"""
task_tabular_classification_catboost.py

Шаблон под табличную КЛАССИФИКАЦИЮ с CatBoostClassifier.

Ожидание по данным:
- train.csv  — содержит столбец с таргетом (класс) и id.
- test.csv   — те же признаки без таргета.
- sample_submission.csv — есть колонка ID_COL и TARGET_COL (или только ID_COL).

ПЕРЕД ЗАПУСКОМ ОБЯЗАТЕЛЬНО:
- Проверь/поменяй:
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH
    TARGET_COL, ID_COL
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# === НАСТРОЙКИ ===
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

TARGET_COL = "target_class"   # имя столбца с классом
ID_COL = "id"                 # идентификатор объекта

RANDOM_STATE = 42
VAL_SIZE = 0.2


def main():
    if CatBoostClassifier is None:
        print("CatBoostClassifier не установлен. Установи catboost или используй RandomForestClassifier.")
        return

    print("=== ЗАГРУЗКА ДАННЫХ ===")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print("\ntrain.head():")
    print(train.head())
    print("\ntrain.info():")
    print(train.info())

    # --- таргет и признаки ---
    print("\n=== ФОРМИРУЕМ X и y ===")
    y_raw = train[TARGET_COL]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("Классы:", list(le.classes_))

    X = train.drop(columns=[TARGET_COL, ID_COL])
    X_test = test.drop(columns=[ID_COL])

    print("Форма X:", X.shape)
    print("Форма X_test:", X_test.shape)

    # --- категориальные признаки для CatBoost ---
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print("\nКатегориальные признаки:", cat_cols)

    for col in cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # --- train/val split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)

    # --- модель ---
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        depth=8,
        learning_rate=0.05,
        iterations=1000,
        random_state=RANDOM_STATE,
        verbose=100
    )

    print("\n=== ОБУЧАЕМ CatBoostClassifier (train/val) ===")
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_cols,
        use_best_model=True
    )

    # --- оценка на валидации ---
    y_val_pred = model.predict(X_val).reshape(-1)
    acc = accuracy_score(y_val, y_val_pred)
    f1w = f1_score(y_val, y_val_pred, average="weighted")
    print(f"\nAccuracy:    {acc:.6f}")
    print(f"Weighted F1: {f1w:.6f}")

    # --- дообучаем на всех данных ---
    print("\n=== ОБУЧАЕМ НА ВСЁМ train И ПРЕДСКАЗЫВАЕМ ДЛЯ test ===")
    model.fit(
        X, y,
        cat_features=cat_cols,
        verbose=100
    )

    test_pred = model.predict(X_test).reshape(-1)
    test_labels = le.inverse_transform(test_pred)

    # --- submission ---
    submission = sample_sub.copy()
    if TARGET_COL not in submission.columns:
        submission[TARGET_COL] = test_labels
    else:
        submission[TARGET_COL] = test_labels

    out_path = "submission_catboost_classification_taskX.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nГотово! Сохранён {out_path}")


if __name__ == "__main__":
    main()
