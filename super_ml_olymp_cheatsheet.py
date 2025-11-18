"""
SUPER_ML_OLYMP_CHEATSHEET.PY

Большая шпаргалка для олимпиадных ML-задач.

ЦЕЛЬ:
- Чтобы за 1 файл можно было:
  * вспомнить хронологию решения любой задачи,
  * быстро собрать код для новой задачи (A-подобной или B-подобной),
  * вспомнить, что делает каждая важная функция.

ТИПЫ ЗАДАЧ, КОТОРЫЕ ПОКРЫВАЕМ:
1) Табличная регрессия (как задача A: предсказать число, MAE / log-MAE).
2) Табличная классификация (класс по табличным фичам).
3) Несколько таблиц (merge по ключу: geo_data, users, items и т.д.).
4) Текстовые задачи / пары текстов (TF-IDF + LogisticRegression).
5) Генеративные задачи с локальной LLM (модель лежит в Shared, без интернета).

СТРУКТУРА ФАЙЛА:
0. Импорты
1. Общий чек-лист для любой ML-задачи
2. Утилиты (функции-кирпичики: метрики, работа с фичами)
3. Шаблон: табличная РЕГРЕССИЯ (A-подобные задачи)
4. Шаблон: табличная КЛАССИФИКАЦИЯ
5. Шаблон: merge нескольких таблиц по ключу
6. Шаблон: текстовая задача / пары текстов (TF-IDF + LogisticRegression)
7. Шаблон: генеративная задача с локальной LLM
8. Мини-справочник по функциям и классам
"""

# ============================================================
# 0. ИМПОРТЫ
# ============================================================

# Работа с таблицами
import pandas as pd              # главный инструмент для табличных данных
import numpy as np               # математические операции, массивы

# Разбиение выборки на train/val
from sklearn.model_selection import train_test_split

# Модели для табличных данных
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Метрики
from sklearn.metrics import (
    mean_absolute_error,         # MAE для регрессии
    r2_score,                    # R² для регрессии
    accuracy_score,              # accuracy для классификации
    f1_score,                    # F1 для классификации
)

# Для кодирования категориальных таргетов и текстовых меток
from sklearn.preprocessing import LabelEncoder

# Для текстов (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

# Опционально: бустинг по деревьям (если установлен)
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:
    CatBoostRegressor = None
    CatBoostClassifier = None

# (для генеративных задач)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


# ============================================================
# 1. ОБЩИЙ ЧЕК-ЛИСТ ДЛЯ ЛЮБОЙ ML-ЗАДАЧИ
# ============================================================

"""
УНИВЕРСАЛЬНЫЙ ПЛАН:

1) ПРОЧИТАТЬ УСЛОВИЕ:
   - Что хотим предсказать?
       * число    → регрессия
       * класс    → классификация
       * текст    → генерация / NLP
   - Какая МЕТРИКА?
       * MAE, RMSE, log-MAE, R²
       * accuracy, F1, ROC-AUC
       * weighted F1, macro F1 и т.п.

2) ЗАГРУЗИТЬ ДАННЫЕ:
   - train: признаки + target
   - test: только признаки
   - sample_submission: формат сабмита
   - дополнительные таблицы: geo_data, items, users и т.п.

3) ПОСМОТРЕТЬ НА ДАННЫЕ:
   - df.head()   → первые строки
   - df.info()   → типы столбцов, пропуски
   - df.describe() → статистика по числам

4) ОПРЕДЕЛИТЬ КОЛОНКИ:
   - TARGET_COL: как называется таргет в train
   - ID_COL: ID объекта (для сабмита)

5) СОБРАТЬ X, y:
   - y = train[TARGET_COL]
   - X = train.drop(columns=[TARGET_COL, ID_COL, ...])
   - X_test = test.drop(columns=[ID_COL, ...])

6) ЕСЛИ ЕСТЬ НЕСКОЛЬКО ТАБЛИЦ:
   - сделать merge по ключу:
       train = train.merge(geo_data, on="h3_index", how="left")
       test  = test.merge(geo_data, on="h3_index", how="left")

7) ОБРАБОТАТЬ ПРИЗНАКИ:
   - заполнить пропуски (если нужно),
   - придумать простые фичи (деления, суммы, логарифмы).

8) КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ:
   - через get_dummies (one-hot)  ИЛИ
   - через CatBoost с cat_features.

9) train_test_split:
   - X_train, X_val, y_train, y_val = train_test_split(...)
   - для классификации → stratify=y.

10) ВЫБРАТЬ МОДЕЛЬ:
   - регрессия: RandomForestRegressor / CatBoostRegressor / LinearRegression
   - классификация: RandomForestClassifier / CatBoostClassifier / LogisticRegression
   - тексты: TF-IDF + LogisticRegression
   - генерация: локальная LLM (AutoModelForCausalLM).

11) ОБУЧИТЬ МОДЕЛЬ И ОЦЕНИТЬ НА ВАЛИДАЦИИ.

12) ОБУЧИТЬ НА ВСЕХ train-ДАННЫХ.

13) ПРЕДСКАЗАТЬ ДЛЯ test.

14) СДЕЛАТЬ submission.csv и сохранить:
   submission.to_csv("submission.csv", index=False)
"""


# ============================================================
# 2. УТИЛИТЫ (ФУНКЦИИ, КОТОРЫЕ ПОЛЕЗНО ЗНАТЬ)
# ============================================================

def print_basic_info(df: pd.DataFrame, name: str = "df"):
    """
    Печатает базовую информацию о таблице:
    - shape
    - head()
    - info()
    """
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    print(df.head())
    print(df.info())


def mae_log10(y_true, y_pred):
    """
    MAE на логарифмах, как в задаче A:
      mae_log10 = MAE(log10(y_true+1), log10(y_pred+1))
    """
    return mean_absolute_error(
        np.log10(y_true + 1),
        np.log10(y_pred + 1)
    )


def encode_categoricals_get_dummies(X: pd.DataFrame, X_test: pd.DataFrame):
    """
    Кодирует категориальные признаки через pd.get_dummies() (one-hot).
    Возвращает:
      X_model, X_test_model, cat_cols
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print("\nКатегориальные признаки:", cat_cols)

    X_model = pd.get_dummies(X, columns=cat_cols)
    X_test_model = pd.get_dummies(X_test, columns=cat_cols)

    # Выравниваем столбцы test под train (очень важно!)
    X_test_model = X_test_model.reindex(columns=X_model.columns, fill_value=0)

    print("X_model shape:", X_model.shape)
    print("X_test_model shape:", X_test_model.shape)

    return X_model, X_test_model, cat_cols


def split_train_val(X, y, is_classification: bool = False, test_size: float = 0.2, random_state: int = 42):
    """
    Обертка над train_test_split.
    Если is_classification = True, делаем stratify=y.
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
    Печатает метрики для регрессии:
    - MAE
    - R²
    - MAE_LOG10 и score = 1/(1+MAE_LOG10), если use_log_mae=True
    """
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"\n[REG] MAE: {mae:.6f}")
    print(f"[REG] R2:  {r2:.6f}")

    if use_log_mae:
        mae_l = mae_log10(y_val, y_pred)
        score = 1 / (1 + mae_l)
        print(f"[REG] MAE_LOG10: {mae_l:.6f}")
        print(f"[REG] Score (1/(1+MAE_LOG10)): {score:.6f}")


def evaluate_classification(y_val, y_pred):
    """
    Печатает метрики для классификации:
    - accuracy
    - weighted F1
    """
    acc = accuracy_score(y_val, y_pred)
    f1w = f1_score(y_val, y_pred, average="weighted")
    print(f"\n[CLS] Accuracy:     {acc:.6f}")
    print(f"[CLS] Weighted F1:  {f1w:.6f}")


# ============================================================
# 3. ШАБЛОН: ТАБЛИЧНАЯ РЕГРЕССИЯ (A-ПОДОБНЫЕ ЗАДАЧИ)
# ============================================================

def tabular_regression_template():
    """
    ШАБЛОН ДЛЯ РЕГРЕССИИ (например, задача A):
    - train.csv
    - test.csv
    - sample_submission.csv

    ПЕРЕД ЗАПУСКОМ:
    1) Поменять пути TRAIN_PATH / TEST_PATH / SAMPLE_SUB_PATH.
    2) Поставить правильные TARGET_COL и ID_COL.
    3) (по желанию) добавить свои фичи.
    """

    # ---------- 3.1 КОНФИГ ----------
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    SAMPLE_SUB_PATH = "sample_submission.csv"

    TARGET_COL = "transaction_sum"   # ИМЯ таргета
    ID_COL = "location_id"           # ИМЯ ID

    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    print("\n=== TABULAR REGRESSION TEMPLATE ===")

    # ---------- 3.2 ЗАГРУЗКА ----------
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print_basic_info(train, "train")
    print_basic_info(test, "test")

    # ---------- 3.3 X, y ----------
    y = train[TARGET_COL]
    X = train.drop(columns=[TARGET_COL, ID_COL])
    X_test = test.drop(columns=[ID_COL])

    # МЕСТО ДЛЯ СВОИХ ФИЧ:
    # Примеры — писать ТОЛЬКО если такие колонки реально есть
    # if "customers_count" in X.columns and "objects_count" in X.columns:
    #     X["customers_per_object"] = X["customers_count"] / (X["objects_count"] + 1)
    #     X_test["customers_per_object"] = X_test["customers_count"] / (X_test["objects_count"] + 1)

    # ---------- 3.4 КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ----------
    X_model, X_test_model, cat_cols = encode_categoricals_get_dummies(X, X_test)

    # ---------- 3.5 TRAIN/VAL SPLIT ----------
    X_train, X_val, y_train, y_val = split_train_val(
        X_model, y, is_classification=False,
        test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # ---------- 3.6 МОДЕЛЬ RandomForestRegressor ----------
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    print("\n>> Обучение RandomForestRegressor...")
    model.fit(X_train, y_train)

    # ---------- 3.7 ОЦЕНКА НА ВАЛИДАЦИИ ----------
    y_val_pred = model.predict(X_val)
    evaluate_regression(y_val, y_val_pred, use_log_mae=True)

    # ---------- 3.8 ОБУЧЕНИЕ НА ВСЁМ TRAIN И SUBMISSION ----------
    print("\n>> Обучение на всех данных и предсказание для test...")
    model.fit(X_model, y)

    test_pred = model.predict(X_test_model)

    submission = sample_sub.copy()
    if TARGET_COL not in submission.columns:
        submission[TARGET_COL] = test_pred
    else:
        submission[TARGET_COL] = test_pred

    submission.to_csv("submission_regression.csv", index=False)
    print("Сохранён файл submission_regression.csv")


# ============================================================
# 4. ШАБЛОН: ТАБЛИЧНАЯ КЛАССИФИКАЦИЯ
# ============================================================

def tabular_classification_template():
    """
    Шаблон для КЛАССИФИКАЦИИ:
    - таргет — класс (строка/число),
    - метрика — accuracy/F1.

    ПЕРЕД ЗАПУСКОМ:
    - указать TRAIN_PATH / TEST_PATH / SAMPLE_SUB_PATH
    - поставить TARGET_COL и ID_COL.
    """

    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    SAMPLE_SUB_PATH = "sample_submission.csv"

    TARGET_COL = "target_class"
    ID_COL = "id"

    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    print("\n=== TABULAR CLASSIFICATION TEMPLATE ===")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print_basic_info(train, "train")
    print_basic_info(test, "test")

    # y_raw могут быть строками (названия классов)
    y_raw = train[TARGET_COL]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("Классы:", list(le.classes_))

    X = train.drop(columns=[TARGET_COL, ID_COL])
    X_test = test.drop(columns=[ID_COL])

    # Кодируем категориальные
    X_model, X_test_model, cat_cols = encode_categoricals_get_dummies(X, X_test)

    # TRAIN/VAL с стратификацией
    X_train, X_val, y_train, y_val = split_train_val(
        X_model, y, is_classification=True,
        test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # Модель
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    print("\n>> Обучение RandomForestClassifier...")
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    evaluate_classification(y_val, y_val_pred)

    # Обучаемся на всех данных
    print("\n>> Обучение на всех данных и предсказание на test...")
    model.fit(X_model, y)
    test_pred = model.predict(X_test_model)
    test_labels = le.inverse_transform(test_pred)

    submission = sample_sub.copy()
    if TARGET_COL not in submission.columns:
        submission[TARGET_COL] = test_labels
    else:
        submission[TARGET_COL] = test_labels

    submission.to_csv("submission_classification.csv", index=False)
    print("Сохранён файл submission_classification.csv")


# ============================================================
# 5. ШАБЛОН: MERGE НЕСКОЛЬКИХ ТАБЛИЦ ПО КЛЮЧУ
# ============================================================

def merge_example_template():
    """
    Шаблон, как объединять несколько таблиц по ключу (например, h3_index):
    - train.csv, test.csv
    - geo_data.parquet

    Дальше — обычный пайплайн регрессии/классификации.
    """

    print("\n=== MERGE EXAMPLE TEMPLATE ===")

    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    GEO_PATH = "geo_data.parquet"

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    geo = pd.read_parquet(GEO_PATH)

    print_basic_info(train, "train BEFORE merge")
    print_basic_info(geo, "geo")

    # Предположим, ключ — h3_index
    train_merged = train.merge(geo, on="h3_index", how="left")
    test_merged = test.merge(geo, on="h3_index", how="left")

    print_basic_info(train_merged, "train_merged")
    print_basic_info(test_merged, "test_merged")

    # Дальше с train_merged и test_merged можно делать всё то же,
    # что в tabular_regression_template / tabular_classification_template:
    # y = train_merged[TARGET_COL]
    # X = train_merged.drop(columns=[TARGET_COL, ID_COL, "h3_index"])
    # X_test = test_merged.drop(columns=[ID_COL, "h3_index"])
    # ...


# ============================================================
# 6. ШАБЛОН: ТЕКСТЫ / ПАРЫ ТЕКСТОВ (TF-IDF + LOGISTIC REGRESSION)
# ============================================================

def text_pairs_tfidf_template():
    """
    Шаблон под B-подобные задачи:
    - items.parquet: item_id, title, text
    - train.csv: pair_id, left_id, right_id, label
    - test.csv: pair_id, left_id, right_id
    - sample_submission.csv: pair_id, label

    Идея:
    1) Мержим train/test с items по left_id/right_id.
    2) Собираем текст пары: [left_title+text] [SEP] [right_title+text].
    3) TF-IDF → матрица признаков.
    4) LogisticRegression → weighted F1.
    """

    print("\n=== TEXT PAIRS TF-IDF TEMPLATE ===")

    ITEMS_PATH = "items.parquet"
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    SAMPLE_SUB_PATH = "sample_submission.csv"

    # Названия колонок
    ID_COL = "pair_id"
    LABEL_COL = "label"

    items = pd.read_parquet(ITEMS_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print_basic_info(items, "items")
    print_basic_info(train, "train")
    print_basic_info(test, "test")

    # Готовим левую и правую части
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

    print_basic_info(train_merged[[ID_COL, "left_id", "right_id", LABEL_COL,
                                   "title_left", "title_right"]], "train_merged small")

    # Собираем текст пары
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

    # Кодируем метки в числа
    le = LabelEncoder()
    y = le.fit_transform(train_merged[LABEL_COL])
    print("Классы:", list(le.classes_))

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=3
    )

    X = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print("X shape:", X.shape)
    print("X_test shape:", X_test.shape)

    # train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Модель
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1
    )

    print("\n>> Обучение LogisticRegression...")
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    f1w = f1_score(y_val, y_val_pred, average="weighted")
    print("Validation weighted F1:", f1w)

    # Обучаем на всех данных + сабмит
    clf.fit(X, y)
    test_pred = clf.predict(X_test)
    test_labels = le.inverse_transform(test_pred)

    submission = sample_sub.copy()
    if LABEL_COL not in submission.columns:
        submission[LABEL_COL] = test_labels
    else:
        submission[LABEL_COL] = test_labels

    submission.to_csv("submission_text_pairs.csv", index=False)
    print("Сохранён файл submission_text_pairs.csv")


# ============================================================
# 7. ШАБЛОН: ГЕНЕРАТИВНАЯ ЗАДАЧА С ЛОКАЛЬНОЙ LLM
# ============================================================

def generative_llm_template():
    """
    Шаблон для генеративной задачи:

    test.csv:
        - колонка INPUT_COL, например "input_text" / "question"
    sample_submission.csv:
        - колонка OUTPUT_COL, например "output_text"

    LLM:
        - лежит локально в MODEL_PATH (например, в Shared).

    ПЕРЕД ЗАПУСКОМ:
    - поменять MODEL_PATH, TEST_PATH, SAMPLE_SUB_PATH
    - указать INPUT_COL, OUTPUT_COL
    - подстроить build_prompt() под условие задачи.
    """

    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        print("transformers / torch не установлены. Генеративный шаблон недоступен.")
        return

    print("\n=== GENERATIVE LLM TEMPLATE ===")

    MODEL_PATH = "/path/to/local/model"   # ПУТЬ К МОДЕЛИ (из Shared)
    TEST_PATH = "test.csv"
    SAMPLE_SUB_PATH = "sample_submission.csv"

    INPUT_COL = "input_text"              # имя колонки с входным текстом
    OUTPUT_COL = "output_text"            # имя колонки для ответа

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_NEW_TOKENS = 128
    BATCH_SIZE = 4

    print("Загрузка модели из:", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    print("Читаем данные...")
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    inputs = test[INPUT_COL].fillna("").tolist()
    outputs = []

    def build_prompt(x: str) -> str:
        """
        Настройка промпта под задачу.
        Здесь — универсальный пример.
        В реальной задаче подстраивать под условие.
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
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

        # Обрезаем промпт из начала, оставляем только "ответ"
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

    OUT_PATH = "submission_generative.csv"
    submission.to_csv(OUT_PATH, index=False)
    print(f"Сохранён файл {OUT_PATH}")
    print(submission.head())


# ============================================================
# 8. МИНИ-СПРАВОЧНИК ПО ФУНКЦИЯМ/КЛАССАМ (как напоминалка)
# ============================================================

"""
pandas:
- pd.read_csv(path), pd.read_parquet(path)
    → прочитать таблицу.

- df.head(n)
    → первые n строк.

- df.info()
    → типы столбцов + количество non-null.

- df.describe()
    → статистика по числовым колонкам.

- df["col"], df[["col1", "col2"]]
    → доступ к столбцам.

- df.drop(columns=[...])
    → удалить столбцы.

- df.merge(other, on="key", how="left")
    → SQL-левый join: "присоединить фичи из другой таблицы".

- pd.get_dummies(df, columns=[...])
    → one-hot кодирование категориальных признаков.

sklearn.model_selection:
- train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    → разбить данные на train и val.

sklearn.preprocessing:
- LabelEncoder:
    le = LabelEncoder()
    y = le.fit_transform(y_raw)        # строки → числа
    y_back = le.inverse_transform(y)   # числа → строки

sklearn.feature_extraction.text:
- TfidfVectorizer:
    v = TfidfVectorizer(...)
    X = v.fit_transform(list_of_texts)
    X_test = v.transform(list_of_texts_test)

Модели:
- RandomForestRegressor / RandomForestClassifier:
    ансамбль деревьев; хорошо работает "из коробки" на табличных данных.

- LinearRegression:
    классическая линейная регрессия.

- LogisticRegression:
    линейная модель для классификации; очень часто — TF-IDF + LogisticRegression.

- DecisionTreeRegressor / DecisionTreeClassifier:
    одно дерево; можно использовать как baseline или для анализа.

- CatBoostRegressor / CatBoostClassifier:
    градиентный бустинг по деревьям, дружит с категориальными признаками.
    Важный параметр: cat_features = [индексы_категориальных_столбцов].

Метрики:
- mean_absolute_error(y_true, y_pred)       → MAE.
- r2_score(y_true, y_pred)                  → R².
- accuracy_score(y_true, y_pred)            → доля правильных ответов.
- f1_score(y_true, y_pred, average="weighted") → F1 с учётом долей классов.

ГЛАВНЫЙ ПАТТЕРН ДЛЯ 90% ЗАДАЧ:
X, y → train_test_split → model.fit → predict → метрика →
обучение на всех данных → predict на test → submission.csv.
"""


if __name__ == "__main__":
    # При прямом запуске файла ничего автоматически не делаем,
    # чтобы случайно не запустить тяжёлые вещи.
    print("SUPER ML OLYMP CHEATSHEET загружен.")
    print("Открой этот файл и используй нужные шаблоны-функции.")
