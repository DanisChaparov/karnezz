"""
task_generative_llm.py

Шаблон под ГЕНЕРАТИВНУЮ задачу с локальной LLM.

Ожидаем:
- test.csv: есть колонка INPUT_COL (например "input_text")
- sample_submission.csv: есть колонка OUTPUT_COL (например "output_text")
- локальная модель лежит по пути MODEL_PATH (например в Shared).

Надо будет:
- поменять MODEL_PATH, INPUT_COL, OUTPUT_COL
- при необходимости подправить build_prompt под условие задачи.
"""

import pandas as pd

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

# Путь к модели в файловой системе кластера (НАДО ПОДСТАВИТЬ)
MODEL_PATH = "/path/to/local/model"

INPUT_COL = "input_text"
OUTPUT_COL = "output_text"

DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
MAX_NEW_TOKENS = 128
BATCH_SIZE = 4


def build_prompt(x: str) -> str:
    """
    Как формируется промпт для модели.
    ЭТО НУЖНО ПОДСТРАИВАТЬ ПОД КОНКРЕТНУЮ ЗАДАЧУ.
    """
    return f"Входной текст:\n{x}\n\nОтветь кратко и понятно на русском языке:"


def main():
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        print("transformers/torch не установлены — генеративный шаблон недоступен.")
        return

    print("=== ЗАГРУЗКА ДАННЫХ ===")
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print(test.head())

    print("\n=== ЗАГРУЗКА МОДЕЛИ ===")
    print("MODEL_PATH:", MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    inputs = test[INPUT_COL].fillna("").tolist()
    outputs = []

    print("\n=== ГЕНЕРАЦИЯ ОТВЕТОВ ===")
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

        print(f"Готово: {min(i + BATCH_SIZE, len(inputs))}/{len(inputs)}")

    outputs = outputs[:len(test)]

    submission = sample_sub.copy()
    if OUTPUT_COL not in submission.columns:
        submission[OUTPUT_COL] = outputs
    else:
        submission[OUTPUT_COL] = outputs

    out_path = "submission_generative_llm.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nГотово! Сохранён {out_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
