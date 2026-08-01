import os
import time
from llama_cpp import Llama

model_dir = "models/llms/qwen2.5-coder-1.5b-instruct"
gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
model_path = os.path.join(model_dir, gguf_files[0])

print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_ctx=512,
    mmap=True,
    verbose=False,
    n_gpu_layers=-1,
    embedding=True,
    n_threads=1,
    n_batch=512
)

# Generate dummy text
dummy_text = "ID: some_id | Keywords: some keywords | Flow: string[source] -> int[result]"
texts = [dummy_text for _ in range(506)]

print(f"Embedding {len(texts)} texts using native batch API...")
t0 = time.time()
try:
    results = llm.create_embedding(texts)
    print(f"Success! {len(results['data'])} embeddings generated.")
except Exception as e:
    print(f"Failed: {e}")
t1 = time.time()
print(f"Time taken: {t1-t0:.2f}s")
