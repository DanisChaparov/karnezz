"""
llm_local_template.py

Шаблон для задач, где нужно использовать LLM ЛОКАЛЬНО:

Пример формата:
- test.csv: содержит колонку с входным текстом (например, "input_text" или "question")
- sample_submission.csv: содержит ID и колонку для ответа (например, "answer")

Цель:
- взять локальную модель (из Shared),
- прогнать по всем строкам test,
- сохранить ответы в submission.csv.

Важно:
- Модель должна лежать на диске (Shared/...),
- Никаких внешних API, интернета и облачных сервисов.
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ==============================
# 1. НАСТРОЙКИ ПОД КОНКРЕТНУЮ ЗАДАЧУ
# ==============================

# Путь к локальной модели (ПОМЕНЯТЬ под окружение олимпиады!)
# Например: "/workspace/Shared/models/llama-small" или что-то, что дадут организаторы
MODEL_PATH = "/path/to/local/model"  # <<< ПОМЕНЯЙ

# Пути к данным
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

# Названия колонок в данных
INPUT_COL = "input_text"     # <<< Колонка с входным текстом в test.csv
OUTPUT_COL = "answer"        # <<< Колонка с ответом в submission (из sample_submission.csv)

# Настройки генерации
MAX_NEW_TOKENS = 128
BATCH_SIZE = 4      # Можно увеличить / уменьшить по VRAM
DO_SAMPLE = False   # Если True — сэмплирование; если False — жёсткий greedy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
# 2. ЗАГРУЗКА МОДЕЛИ
# ==============================

print(">> Загрузка модели из:", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# Если у модели нет pad_token_id — можно приравнять к eos_token_id
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# ==============================
# 3. ЗАГРУЗКА ДАННЫХ
# ==============================

print(">> Чтение test и sample_submission...")
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("test shape:", test.shape)
print("test head:")
print(test.head())

if INPUT_COL not in test.columns:
    raise ValueError(f"В test.csv нет колонки {INPUT_COL}. Проверь имя столбца.")

if OUTPUT_COL not in sample_sub.columns:
    # если в sample_submission другое имя колонки — можно создать свою
    print(f"В sample_submission нет колонки {OUTPUT_COL}, создаю новую.")
    sample_sub[OUTPUT_COL] = ""


# ==============================
# 4. ПОДГОТОВКА ВХОДОВ
# ==============================

inputs = test[INPUT_COL].fillna("").tolist()
outputs = []

print(">> Начинаем генерацию...")
for start in range(0, len(inputs), BATCH_SIZE):
    batch_texts = inputs[start:start + BATCH_SIZE]

    # Здесь можно при необходимости добавить системный промпт:
    # Например:
    # batch_prompts = [
    #     f"Ты помощник по анализу текста. Ответь кратко.\nВопрос: {t}\nОтвет:"
    #     for t in batch_texts
    # ]
    batch_prompts = batch_texts  # базовый вариант: без доп. обёртки

    enc = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            # если DO_SAMPLE=True, можно добавить:
            # top_p=0.9,
            # temperature=0.7,
        )

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

    # Внимание: многие модели возвращают "Вход + Ответ".
    # Часто проще всего взять:
    #   - либо весь decoded[i],
    #   - либо вырезать всё после исходного входа.
    #
    # Базовый вариант: просто берём последнюю строку/часть.
    for inp, full_out in zip(batch_prompts, decoded):
        # Пытаемся вырезать ответ после входа, если модель дописывает дальше
        if full_out.startswith(inp):
            answer = full_out[len(inp):].strip()
        else:
            answer = full_out.strip()
        outputs.append(answer)

    print(f"  Сгенерировано: {min(start + BATCH_SIZE, len(inputs))}/{len(inputs)}")


# ==============================
# 5. СОБИРАЕМ SUBMISSION
# ==============================

# На всякий случай обрезаем до длины test
outputs = outputs[:len(test)]

submission = sample_sub.copy()
submission[OUTPUT_COL] = outputs

OUT_PATH = "submission_llm.csv"
submission.to_csv(OUT_PATH, index=False)

print(f"\nГотово! Сабмит сохранён в {OUT_PATH}")
print("Пример строк сабмита:")
print(submission.head())
