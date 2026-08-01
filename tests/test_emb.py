import os
from llama_cpp import Llama
from inference import ModelManager

print("Loading model...")
llm = Llama(
    model_path="models/llms/qwen2.5-coder-1.5b-instruct/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    n_ctx=512,
    embedding=True,
    verbose=False
)
print("Embedding...")
t = "ID: numpy_seed_1 | Keywords: math array | Flow: any[any] -> any[any]"
raw = llm.create_embedding(t)
emb = raw['data'][0]['embedding']
print("Type of emb:", type(emb))
print("Len of emb:", len(emb))
if isinstance(emb, list) and len(emb) > 0:
    print("Type of emb[0]:", type(emb[0]))
    if isinstance(emb[0], list):
        print("Len of emb[0]:", len(emb[0]))
        print("Type of emb[0][0]:", type(emb[0][0]))
