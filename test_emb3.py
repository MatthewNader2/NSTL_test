from llama_cpp import Llama
llm = Llama(model_path="model_2/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", embedding=True, verbose=False)
result = llm.create_embedding("hello")
print(result)
