"""
SUPER ML OLYMP CHEATSHEET
=========================

Файл-шпаргалка для олимпиадных ML-задач.

СТРУКТУРА:
0. Импорты (и комментарии, зачем каждый)
1. Часто используемые функции (с объяснением)
2. Памятка по базовым командам/операциям (pandas, sklearn)
3. БУСТИНГИ (CatBoost) — что это и как использовать
4. ШАБЛОНЫ ЗАДАЧ:
   4.1. Табличная регрессия (A-подобная задача) с CatBoost
   4.2. Табличная классификация с CatBoost
   4.3. Merge нескольких таблиц (пример)
   4.4. Текстовые пары (B-подобная задача) — TF-IDF + LogisticRegression
   4.5. Генеративная задача (локальная LLM)
   4.6. Аудио baseline (как из аудио сделать табличку)

Идея:
- Открываешь этот файл.
- Берёшь нужный шаблон (функцию), копируешь в своё решение.
- Меняешь пути к файлам и имена колонок.
- Запускаешь.
"""

# ============================================================
# 0. ИМПОРТЫ (и для чего они нужны)
# ============================================================

import os  # работа с путями к файлам/папкам
import numpy as np  # массивы, математика
import pandas as pd  # табличные данные (DataFrame)

# Разбиение данных на train/val
from sklearn.model_selection import train_test_split  # train_test_split

# Модели для табличных задач
from sklearn.ensemble import (
    RandomForestRegressor,    # ансамбль деревьев для регрессии
    RandomForestClassifier,   # ансамбль деревьев для классификации
)
from sklearn.linear_model import (
    LinearRegression,         # линейная регрессия (baseline)
    LogisticRegression,       # логистическая регрессия (часто с TF-IDF)
)
from sklearn.tree import (
    DecisionTreeRegressor,    # одно дерево для регрессии
    DecisionTreeClassifier,   # одно дерево для классификации
)

# Метрики качества
from sklearn.metrics import (
    mean_absolute_error,      # MAE для регрессии
    r2_score,                 # R² для регрессии
    accuracy_score,           # accuracy для классификации
    f1_score,                 # F1 (например weighted F1 для B-задачи)
)

# Преобразование меток (строк → числа)
from sklearn.preprocessing import LabelEncoder  # кодирование классов

# Для текстовых задач (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer  # превращает текст в признаки

# Попытка импортировать CatBoost (буст по деревьям)
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:
    CatBoostRegressor = None
    CatBoostClassifier = None

# Попытка импортировать всё для генеративных задач (LLM)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


# ============================================================
# 1. ЧАСТО ИСПОЛЬЗУЕМЫЕ ФУНКЦИИ (утилиты)
# ============================================================

def print_basic_info(df: pd.DataFrame, name: str = "df"):
    """
    Вывести базовую инфу о таблице:
    - название (name)
    - размер (shape)
    - первые строки (head)
    - типы столбцов и кол-во значений (info)
    """
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    print(df.head())
    print(df.info())


def mae_log10(y_true, y_pred):
    """
    MAE на логарифмах (как в задаче A):

    mae_log10 = MAE(log10(y_true + 1), log10(y_pred + 1))

    Используется, когда метрика в условии через log10.
    """
    return mean_absolute_error(
        np.log10(y_true + 1),
        np.log10(y_pred + 1)
    )


def encode_categoricals_get_dummies(X: pd.DataFrame, X_test: pd.DataFrame):
    """
    Кодирование категориальных признаков через one-hot (pd.get_dummies).

    Возвращает:
    - X_model      : DataFrame с dummy-признаками для train
    - X_test_model : DataFrame с dummy-признаками для test
    - cat_cols     : список категориальных колонок
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print("\nКатегориальные признаки (get_dummies):", cat_cols)

    X_model = pd.get_dummies(X, columns=cat_cols)
    X_test_model = pd.get_dummies(X_test, columns=cat_cols)

    # выравниваем столбцы test под train
    X_test_model = X_test_model.reindex(columns=X_model.columns, fill_value=0)

    print("X_model shape:", X_model.shape)
    print("X_test_model shape:", X_test_model.shape)

    return X_model, X_test_model, cat_cols


def split_train_val(
    X,
    y,
    is_classification: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Обёртка над train_test_split.

    Если is_classification=True — используем stratify=y,
    чтобы баланс классов в train/val был похожий.
    """
    kwargs = dict(test_size=test_size, random_state=random_state)
    if is_classification:
        kwargs["stratify"] = y

    X_train, X_val, y_train, y_val = train_test_split(X, y, **kwargs)
    print("\nX_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    return X_train, X_val, y_train, y_val


def evaluate_regression(y_val, y_pred, use_log_mae: bool = True):
    """
    Печатаем метрики для регрессии:
    - MAE
    - R²
    - MAE_LOG10 и score = 1/(1 + MAE_LOG10), если use_log_mae=True
    """
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"\n[REG] MAE: {mae:.6f}")
    print(f"[REG] R² : {r2:.6f}")
    if use_log_mae:
        mae_l = mae_log10(y_val, y_pred)
        score = 1 / (1 + mae_l)
        print(f"[REG] MAE_LOG10: {mae_l:.6f}")
        print(f"[REG] Score (1/(1+MAE_LOG10)): {score:.6f}")


def evaluate_classification(y_val, y_pred):
    """
    Печатаем метрики для классификации:
    - accuracy
    - weighted F1
    """
    acc = accuracy_score(y_val, y_pred)
    f1w = f1_score(y_val, y_pred, average="weighted")
    print(f"\n[CLS] Accuracy:    {acc:.6f}")
    print(f"[CLS] Weighted F1: {f1w:.6f}")


# ============================================================
# 2. ПАМЯТКА: БАЗОВЫЕ ОПЕРАЦИИ / КОМАНДЫ
# ============================================================

BASIC_COMMANDS_HELP = """
PANDAS (таблицы)
----------------
pd.read_csv(path)           - прочитать csv в DataFrame
pd.read_parquet(path)       - прочитать parquet

df.head(n)                  - первые n строк
df.info()                   - типы столбцов, кол-во значений
df.describe()               - статистика по числам

df["col"]                   - взять один столбец
df[["c1","c2"]]             - несколько столбцов

df.drop(columns=[...])      - удалить столбцы
df.merge(other, on="key", how="left")
                            - соединить две таблицы как SQL LEFT JOIN

pd.get_dummies(df, columns=[...])
                            - one-hot кодирование категориальных признаков

SKLEARN (общая схема)
---------------------
1) X, y
    y = train[TARGET_COL]
    X = train.drop(columns=[TARGET_COL, ...])

2) train/val split:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

3) модель:
    model = RandomForestRegressor(...)
    model.fit(X_train, y_train)

4) предсказать:
    y_val_pred = model.predict(X_val)

5) метрика:
    mae = mean_absolute_error(y_val, y_val_pred)

6) обучение на всех данных:
    model.fit(X, y)
    test_pred = model.predict(X_test)

7) сабмит:
    submission[target_col] = test_pred
    submission.to_csv("submission.csv", index=False)
"""

# ============================================================
# 3. БУСТИНГИ (CATBOOST) — что это и как использовать
# ============================================================

CATBOOST_HELP = """
CATBOOST (градиентный бустинг по деревьям)
-----------------------------------------
- Главная идея: ансамбль деревьев, обучаемых последовательно.
- Главное отличие от RandomForest:
    * у RandomForest деревья независимые (bagging),
    * у CatBoost деревья строятся по очереди (boosting) и учитывают ошибки прошлых.

Плюсы CatBoost:
- Обычно даёт лучшее качество на табличных данных.
- Умеет работать с категориальными признаками напрямую (без get_dummies).

Как использовать (регрессия):
-----------------------------
1) X, y как обычно.
2) cat_cols = список категориальных признаков (тип object/category).
3) Привести X[col].astype("category").
4) model = CatBoostRegressor(... параметры ...)
5) model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_cols)

Как использовать (классификация):
---------------------------------
1) y преобразовать через LabelEncoder (строка -> число).
2) CatBoostClassifier( loss_function="MultiClass" / "Logloss" ... ).
3) Остальное аналогично.
"""


# ============================================================
# 4. ШАБЛОНЫ ЗАДАЧ
# ============================================================

# ------------------------------------------------------------
# 4.1. Табличная РЕГРЕССИЯ (A-подобная) с CatBoost
# ------------------------------------------------------------

def template_tabular_regression_catboost():
    """
    Шаблон для A-подобной задачи (регрессия) с CatBoostRegressor.

    Ожидаем файлы:
    - train.csv
    - test.csv
    - sample_submission.csv

    ПЕРЕД ЗАПУСКОМ:
    - Поменять TRAIN_PATH / TEST_PATH / SAMPLE_SUB_PATH.
    - Поставить правильные TARGET_COL и ID_COL.
    - Убедиться, что catboost установлен.
    """

    if CatBoostRegressor is None:
        print("CatBoostRegressor не установлен. Установи catboost или используй RandomForest.")
        return

    print("\n=== TEMPLATE: TABULAR REGRESSION (CatBoost) ===")

    # --- настройки (менять под задачу) ---
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    SAMPLE_SUB_PATH = "sample_submission.csv"

    TARGET_COL = "transaction_sum"  # имя столбца-таргета
    ID_COL = "location_id"          # идентификатор объекта

    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # --- загрузка данных ---
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print_basic_info(train, "train")
    print_basic_info(test, "test")

    # --- разделяем на X и y ---
    y = train[TARGET_COL]
    X = train.drop(columns=[TARGET_COL, ID_COL])
    X_test = test.drop(columns=[ID_COL])

    # --- пример добавления своих фич (если нужны) ---
    # if "customers_count" in X.columns and "objects_count" in X.columns:
    #     X["customers_per_object"] = X["customers_count"] / (X["objects_count"] + 1)
    #     X_test["customers_per_object"] = X_test["customers_count"] / (X_test["objects_count"] + 1)

    # --- категориальные признаки (без get_dummies) ---
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print("\nCatBoost, категориальные признаки:", cat_cols)

    for col in cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # --- train/val split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )

    # --- модель CatBoostRegressor ---
    model = CatBoostRegressor(
        loss_function="MAE",   # подходит, если метрика связана с MAE
        eval_metric="MAE",
        depth=8,
        learning_rate=0.05,
        iterations=1000,
        random_state=RANDOM_STATE,
        verbose=100
    )

    print("\n>> Обучаем CatBoostRegressor...")
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_cols,
        use_best_model=True
    )

    # --- оценка на валидации ---
    y_val_pred = model.predict(X_val)
    evaluate_regression(y_val, y_val_pred, use_log_mae=True)

    # --- обучение на всех train-данных ---
    print("\n>> Обучаем модель на всех train-данных и предсказываем для test...")
    model.fit(
        X, y,
        cat_features=cat_cols,
        verbose=100
    )

    test_pred = model.predict(X_test)

    # --- формируем submission ---
    submission = sample_sub.copy()
    if TARGET_COL not in submission.columns:
        submission[TARGET_COL] = test_pred
    else:
        submission[TARGET_COL] = test_pred

    out_path = "submission_catboost_regression.csv"
    submission.to_csv(out_path, index=False)
    print(f"Сохранён файл {out_path}")


# ------------------------------------------------------------
# 4.2. Табличная КЛАССИФИКАЦИЯ с CatBoost
# ------------------------------------------------------------

def template_tabular_classification_catboost():
    """
    Шаблон для табличной классификации с CatBoostClassifier.

    Ожидаем файлы:
    - train.csv
    - test.csv
    - sample_submission.csv

    ПЕРЕД ЗАПУСКОМ:
    - Поменять пути и имена колонок.
    """

    if CatBoostClassifier is None:
        print("CatBoostClassifier не установлен. Установи catboost или используй RandomForestClassifier.")
        return

    print("\n=== TEMPLATE: TABULAR CLASSIFICATION (CatBoost) ===")

    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    SAMPLE_SUB_PATH = "sample_submission.csv"

    TARGET_COL = "target_class"  # имя столбца с классом
    ID_COL = "id"

    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print_basic_info(train, "train")
    print_basic_info(test, "test")

    # target (строки → числа)
    y_raw = train[TARGET_COL]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("Классы:", list(le.classes_))

    X = train.drop(columns=[TARGET_COL, ID_COL])
    X_test = test.drop(columns=[ID_COL])

    # категориальные признаки
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print("\nCatBoost, категориальные признаки:", cat_cols)

    for col in cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = CatBoostClassifier(
        loss_function="MultiClass",    # многоклассовая классификация
        eval_metric="TotalF1",         # можно менять под задачу
        depth=8,
        learning_rate=0.05,
        iterations=1000,
        random_state=RANDOM_STATE,
        verbose=100
    )

    print("\n>> Обучаем CatBoostClassifier...")
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_cols,
        use_best_model=True
    )

    y_val_pred = model.predict(X_val).reshape(-1)
    evaluate_classification(y_val, y_val_pred)

    print("\n>> Обучаемся на всех данных и делаем предсказания для test...")
    model.fit(
        X, y,
        cat_features=cat_cols,
        verbose=100
    )

    test_pred = model.predict(X_test).reshape(-1)
    test_labels = le.inverse_transform(test_pred)

    submission = sample_sub.copy()
    if TARGET_COL not in submission.columns:
        submission[TARGET_COL] = test_labels
    else:
        submission[TARGET_COL] = test_labels

    out_path = "submission_catboost_classification.csv"
    submission.to_csv(out_path, index=False)
    print(f"Сохранён файл {out_path}")


# ------------------------------------------------------------
# 4.3. Пример: MERGE нескольких таблиц по ключу
# ------------------------------------------------------------

def template_merge_example():
    """
    Пример, как объединить train/test с дополнительной таблицей (например geo_data).

    Ожидаем:
    - train.csv, test.csv
    - geo_data.parquet (или csv), есть общий ключ (например h3_index)

    После merge получаем train_merged / test_merged и дальше используем
    один из шаблонов регрессии/классификации.
    """

    print("\n=== TEMPLATE: MERGE EXAMPLE ===")

    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    GEO_PATH = "geo_data.parquet"  # или .csv

    KEY_COL = "h3_index"  # по какому ключу мержим

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if GEO_PATH.endswith(".parquet"):
        geo = pd.read_parquet(GEO_PATH)
    else:
        geo = pd.read_csv(GEO_PATH)

    print_basic_info(train, "train BEFORE merge")
    print_basic_info(geo, "geo")

    train_merged = train.merge(geo, on=KEY_COL, how="left")
    test_merged = test.merge(geo, on=KEY_COL, how="left")

    print_basic_info(train_merged, "train_merged")
    print_basic_info(test_merged, "test_merged")

    # дальше:
    # y = train_merged[TARGET_COL]
    # X = train_merged.drop(columns=[TARGET_COL, ID_COL, KEY_COL])
    # X_test = test_merged.drop(columns=[ID_COL, KEY_COL])
    # ... использовать один из шаблонов выше.


# ------------------------------------------------------------
# 4.4. TEXT PAIRS (B-подобная) — TF-IDF + LogisticRegression
# ------------------------------------------------------------

def template_text_pairs_tfidf():
    """
    Шаблон под задачу, похожую на B:

    Дано:
    - items.parquet: item_id, title, text
    - train.csv: pair_id, left_id, right_id, label
    - test.csv: pair_id, left_id, right_id
    - sample_submission.csv: pair_id, label

    Идея:
    - смержить train/test с items (left/right)
    - собрать текст пары: left_title + left_text + [SEP] + right_title + right_text
    - TF-IDF → LogisticRegression
    """

    print("\n=== TEMPLATE: TEXT PAIRS TF-IDF ===")

    ITEMS_PATH = "items.parquet"
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    SAMPLE_SUB_PATH = "sample_submission.csv"

    ID_COL = "pair_id"
    LABEL_COL = "label"

    items = pd.read_parquet(ITEMS_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print_basic_info(items, "items")
    print_basic_info(train, "train")
    print_basic_info(test, "test")

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

    def build_pair_text(df: pd.DataFrame) -> pd.Series:
        return (
            df["title_left"].fillna("") + " " +
            df["text_left"].fillna("") + " [SEP] " +
            df["title_right"].fillna("") + " " +
            df["text_right"].fillna("")
        )

    train_texts = build_pair_text(train_merged)
    test_texts = build_pair_text(test_merged)

    print("\nПример текста пары:\n", train_texts.iloc[0])

    le = LabelEncoder()
    y = le.fit_transform(train_merged[LABEL_COL])
    print("Классы:", list(le.classes_))

    vectorizer = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=3
    )

    X = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print("X shape:", X.shape)
    print("X_test shape:", X_test.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1
    )

    print("\n>> Обучаем LogisticRegression (TF-IDF)...")
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    f1w = f1_score(y_val, y_val_pred, average="weighted")
    print("Validation weighted F1:", f1w)

    clf.fit(X, y)
    test_pred = clf.predict(X_test)
    test_labels = le.inverse_transform(test_pred)

    submission = sample_sub.copy()
    if LABEL_COL not in submission.columns:
        submission[LABEL_COL] = test_labels
    else:
        submission[LABEL_COL] = test_labels

    out_path = "submission_text_pairs.csv"
    submission.to_csv(out_path, index=False)
    print(f"Сохранён файл {out_path}")


# ------------------------------------------------------------
# 4.5. Генеративная задача (локальная LLM)
# ------------------------------------------------------------

def template_generative_llm():
    """
    Генеративная задача с локальной LLM:

    Ожидаем:
    - test.csv, в нём колонка INPUT_COL (например, "input_text")
    - sample_submission.csv, в нём колонка OUTPUT_COL (например, "output_text")
    - модель лежит в MODEL_PATH (в Shared)

    ВАЖНО:
    - Подстроить MODEL_PATH, INPUT_COL, OUTPUT_COL и build_prompt под реальную задачу.
    """

    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        print("transformers/torch не установлены — генеративный шаблон недоступен.")
        return

    print("\n=== TEMPLATE: GENERATIVE LLM ===")

    MODEL_PATH = "/path/to/local/model"  # поменять на путь в Shared
    TEST_PATH = "test.csv"
    SAMPLE_SUB_PATH = "sample_submission.csv"

    INPUT_COL = "input_text"
    OUTPUT_COL = "output_text"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_NEW_TOKENS = 128
    BATCH_SIZE = 4

    print("Загружаем модель из:", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    inputs = test[INPUT_COL].fillna("").tolist()
    outputs = []

    def build_prompt(x: str) -> str:
        """
        Шаблон промпта — подстраивать под условие задачи.
        """
        return f"Входной текст:\n{x}\n\nОтветь кратко и понятно на русском языке:"

    print("Генерация ответов...")
    for i in range(0, len(inputs), BATCH_SIZE):
        batch_texts = inputs[i:i + BATCH_SIZE]
        prompts = [build_prompt(t) for t in batch_texts]

        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for prompt, full in zip(prompts, decoded):
            if full.startswith(prompt):
                answer = full[len(prompt):].strip()
            else:
                answer = full.strip()
            outputs.append(answer)

        print(f"Генерировано {min(i + BATCH_SIZE, len(inputs))}/{len(inputs)}")

    outputs = outputs[:len(test)]

    submission = sample_sub.copy()
    if OUTPUT_COL not in submission.columns:
        submission[OUTPUT_COL] = outputs
    else:
        submission[OUTPUT_COL] = outputs

    out_path = "submission_generative.csv"
    submission.to_csv(out_path, index=False)
    print(f"Сохранён файл {out_path}")
    print(submission.head())


# ------------------------------------------------------------
# 4.6. Аудио baseline — как из аудио сделать табличку
# ------------------------------------------------------------

def template_audio_baseline():
    """
    Базовый шаблон для аудио-задачи:

    Ожидаем:
    - train.csv: file_name, target
    - test.csv: file_name
    - папка AUDIO_DIR с wav-файлами

    Идея:
    - прочитать аудио,
    - посчитать простые признаки (mean, std, mfcc_mean, mfcc_std),
    - получить табличку и дальше использовать шаблон CatBoost/RandomForest.

    НУЖНО:
    - библиотека librosa (если её не будет на олимпиаде, будет пример только как идея).
    """

    try:
        import librosa
    except ImportError:
        print("librosa не установлена — аудио baseline как код может не работать, но логика показана.")
        return

    print("\n=== TEMPLATE: AUDIO BASELINE ===")

    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    AUDIO_DIR = "audio"  # папка с wav

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    print_basic_info(train, "train")
    print_basic_info(test, "test")

    def extract_features(path, sr=16000):
        """
        Вычислить простые признаки для одного аудио:
        - mean, std по сигналу
        - mfcc_mean, mfcc_std по MFCC
        """
        y, sr = librosa.load(path, sr=sr)
        feats = {}
        feats["signal_mean"] = float(np.mean(y))
        feats["signal_std"] = float(np.std(y))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        feats["mfcc_mean"] = float(mfcc.mean())
        feats["mfcc_std"] = float(mfcc.std())
        return feats

    def build_features(df):
        rows = []
        for fname in df["file_name"]:
            full_path = os.path.join(AUDIO_DIR, fname)
            feats = extract_features(full_path)
            feats["file_name"] = fname
            rows.append(feats)
        return pd.DataFrame(rows)

    print("Извлекаем признаки для train...")
    train_feats = build_features(train)
    print("Извлекаем признаки для test...")
    test_feats = build_features(test)

    train_full = train.merge(train_feats, on="file_name")
    test_full = test.merge(test_feats, on="file_name")

    print_basic_info(train_full, "train_full")
    print_basic_info(test_full, "test_full")

    # Дальше:
    # y = train_full["target"]
    # X = train_full.drop(columns=["target", "file_name"])
    # X_test = test_full.drop(columns=["file_name"])
    # ... и можно применить template_tabular_regression_catboost / RandomForest.
    print("Дальше можно использовать любой табличный шаблон (CatBoost / RandomForest).")


# ============================================================
# ФИНАЛ: что делать при прямом запуске файла
# ============================================================

if __name__ == "__main__":
    print("SUPER ML OLYMP CHEATSHEET загружен.")
    print("Открой файл, выбери нужный шаблон (template_...) и копируй в своё решение.")
    print("Также можно вызвать функции прямо отсюда, если указать правильные пути к данным.")
