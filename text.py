"""
task_text_pairs_tfidf_logreg.py

Шаблон под задачу текстовых ПАР (как B):

Есть:
- items.parquet: item_id, title, text
- train.csv: pair_id, left_id, right_id, label
- test.csv:  pair_id, left_id, right_id
- sample_submission.csv: pair_id, label

Идея:
- смержить train/test с items,
- собрать текст пары: left_title+left_text [SEP] right_title+right_text,
- TF-IDF → LogisticRegression,
- метрика: weighted F1.

ПЕРЕД ЗАПУСКОМ:
- Проверь пути и названия колонок.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ITEMS_PATH = "items.parquet"
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

ID_COL = "pair_id"
LABEL_COL = "label"

RANDOM_STATE = 42
VAL_SIZE = 0.2


def build_pair_text(df: pd.DataFrame) -> pd.Series:
    """
    Собирает текст пары из заголовков и текстов:

    title_left + text_left + [SEP] + title_right + text_right
    """
    return (
        df["title_left"].fillna("") + " " +
        df["text_left"].fillna("") + " [SEP] " +
        df["title_right"].fillna("") + " " +
        df["text_right"].fillna("")
    )


def main():
    print("=== ЗАГРУЗКА ДАННЫХ ===")
    items = pd.read_parquet(ITEMS_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print("\nitems.head():")
    print(items.head())
    print("\ntrain.head():")
    print(train.head())

    # --- готовим items_left / items_right ---
    items_left = items.rename(columns={
        "item_id": "left_id",
        "title": "title_left",
        "text": "text_left",
    })
    items_right = items.rename(columns={
        "item_id": "right_id",
        "title": "title_right",
        "text": "text_right",
    })

    # --- merge для train/test ---
    train_merged = (
        train
        .merge(items_left, on="left_id", how="left")
        .merge(items_right, on="right_id", how="left")
    )

    test_merged = (
        test
        .merge(items_left, on="left_id", how="left")
        .merge(items_right, on="right_id", how="left")
    )

    print("\ntrain_merged.head():")
    print(train_merged.head())

    # --- формируем тексты пар ---
    train_texts = build_pair_text(train_merged)
    test_texts = build_pair_text(test_merged)

    print("\nПример текста пары:")
    print(train_texts.iloc[0])

    # --- кодируем классы ---
    le = LabelEncoder()
    y = le.fit_transform(train_merged[LABEL_COL])
    print("Классы:", list(le.classes_))

    # --- TF-IDF ---
    vectorizer = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=3
    )

    print("\n=== Обучаем TF-IDF ===")
    X = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print("X shape:", X.shape)
    print("X_test shape:", X_test.shape)

    # --- train/val split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --- модель ---
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1
    )

    print("\n=== Обучаем LogisticRegression (train/val) ===")
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    f1w = f1_score(y_val, y_val_pred, average="weighted")
    print(f"Validation weighted F1: {f1w:.6f}")

    # --- дообучаем на всех данных ---
    print("\n=== Обучаем на всём train и предсказываем test ===")
    clf.fit(X, y)
    test_pred = clf.predict(X_test)
    test_labels = le.inverse_transform(test_pred)

    # --- submission ---
    submission = sample_sub.copy()
    if LABEL_COL not in submission.columns:
        submission[LABEL_COL] = test_labels
    else:
        submission[LABEL_COL] = test_labels

    out_path = "submission_text_pairs_tfidf_logreg.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nГотово! Сохранён {out_path}")


if __name__ == "__main__":
    main()
