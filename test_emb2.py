from llama_cpp import Llama

llm = Llama(model_path="model_2/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", embedding=True, verbose=False)
result = llm.create_embedding("hello world")
print(type(result['data'][0]['embedding']))
if isinstance(result['data'][0]['embedding'], list):
    print(len(result['data'][0]['embedding']))
    if len(result['data']) > 1:
        print(len(result['data'][1]['embedding']))
    print("Num tokens/items in data:", len(result['data']))
