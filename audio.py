"""
task_audio_to_tabular_baseline.py

Базовый шаблон под АУДИО-задачу:

Ожидаем:
- train.csv: file_name, target
- test.csv: file_name
- папка AUDIO_DIR с .wav файлами

Идея:
- Прочитать каждое аудио,
- Посчитать простые признаки (mean, std, mfcc_mean, mfcc_std),
- Получить табличные X и X_test,
- Потом можно подключить любой табличный шаблон (CatBoost/RandomForest).
"""

import os
import numpy as np
import pandas as pd

try:
    import librosa
except ImportError:
    librosa = None

TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
AUDIO_DIR = "audio"    # папка с wav-файлами, лежит рядом


def extract_features(path, sr=16000):
    """
    Вычислить простые признаки для одного аудио:
    - среднее значение сигнала
    - стандартное отклонение
    - среднее и std по MFCC
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


def main():
    if librosa is None:
        print("librosa не установлена — код не запустится, но это хороший шаблон логики для аудио-задачи.")
        return

    print("=== ЗАГРУЗКА train/test ===")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    print("\ntrain.head():")
    print(train.head())

    print("\n=== ИЗВЛЕКАЕМ ПРИЗНАКИ ИЗ АУДИО ===")
    print("Train аудио...")
    train_feats = build_features(train)
    print("Test аудио...")
    test_feats = build_features(test)

    train_full = train.merge(train_feats, on="file_name")
    test_full = test.merge(test_feats, on="file_name")

    print("\ntrain_full.head():")
    print(train_full.head())
    print("\ntrain_full.info():")
    print(train_full.info())

    # Дальше:
    # y = train_full["target"]
    # X = train_full.drop(columns=["target", "file_name"])
    # X_test = test_full.drop(columns=["file_name"])
    # И можно использовать CatBoost / RandomForest из других шаблонов.
    print("\nТеперь можно взять X, y, X_test и подключить табличный шаблон (CatBoost/RandomForest).")


if __name__ == "__main__":
    main()
