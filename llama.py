from llama_cpp import Llama

llm = Llama(
    model_path="/Shared/models/llama-7b/llama-7b.gguf",
    n_ctx=2048,
    n_gpu_layers=40
)

print(llm("Напиши краткое объяснение градиентного бустинга.")["choices"][0]["text"])

pip install llama-cpp-pythonp
