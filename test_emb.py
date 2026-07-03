from llama_cpp import Llama

llm = Llama(model_path="model_2/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", embedding=True, verbose=False)
emb1 = llm.create_embedding("hello")['data'][0]['embedding']
emb2 = llm.create_embedding("hello world, this is a longer sentence to test dimension")['data'][0]['embedding']

print("emb1 length:", len(emb1))
print("emb2 length:", len(emb2))
