# NSTL Evaluation System Report

**Total Runs**: 28
**Overall Success Rate**: 22/28 (78.6%)

## Success Rate by Profile
| Profile | Success Rate |
|---|---|
| A | 4/4 (100.0%) |
| C | 9/12 (75.0%) |
| D | 9/12 (75.0%) |


## Success Rate by Embedder
| Embedder | Success Rate |
|---|---|
| embeddinggemma-300m | 11/14 (78.6%) |
| jina-embeddings-v5-text-nano | 11/14 (78.6%) |


## Success Rate by LLM (Profiles C/D)
| LLM | Success Rate |
|---|---|
| Qwen2.5-Coder-7B-Instruct-GGUF | 7/8 (87.5%) |
| qwen2.5-coder-0.5b-instruct | 8/8 (100.0%) |
| qwen2.5-coder-1.5b-instruct | 3/8 (37.5%) |


## Success Rate by Task
| Task ID | Success Rate |
|---|---|
| math_add_function | 13/14 (92.9%) |
| pandas_csv_clean | 9/14 (64.3%) |


## Detailed Failures
### pandas_csv_clean (Profile C, Emb: jina-embeddings-v5-text-nano, LLM: qwen2.5-coder-1.5b-instruct)
**Error**:
```
Validation failed: Ages are not sorted in descending order
```
**Generated Code**:
```python
import pandas

input_file = 'data.csv'
data = pandas.read_csv(input_file)
cleaned_data = data.dropna()
sorted_data = cleaned_data.sort_values('age', ascending=False)
cleaned_data.to_csv('cleaned_data.csv', index=False)
```

### math_add_function (Profile C, Emb: jina-embeddings-v5-text-nano, LLM: qwen2.5-coder-1.5b-instruct)
**Error**:
```
Execution failed with return code 1:
Traceback (most recent call last):
  File "/media/matthew/New Volume/grad_test/nstl_prototype/temp_eval_run.py", line 2, in <module>
    core_computation_eval_ndframe_replace = input_source.replace()
                                            ^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'replace'

```
**Generated Code**:
```python
input_source = None
core_computation_eval_ndframe_replace = input_source.replace()
_config_config_deprecatedoption_rkey_default = core_computation_eval_ndframe_replace.rkey
```

### pandas_csv_clean (Profile D, Emb: jina-embeddings-v5-text-nano, LLM: qwen2.5-coder-1.5b-instruct)
**Error**:
```
Validation failed: Ages are not sorted in descending order
```
**Generated Code**:
```python
import pandas

input_file = 'data.csv'
data = pandas.read_csv(input_file)
cleaned_data = data.dropna()
sorted_data = cleaned_data.sort_values('age', ascending=False)
cleaned_data.to_csv('cleaned_data.csv', index=False)
```

### pandas_csv_clean (Profile C, Emb: embeddinggemma-300m, LLM: qwen2.5-coder-1.5b-instruct)
**Error**:
```
Validation failed: Ages are not sorted in descending order
```
**Generated Code**:
```python
import pandas

input_file = 'data.csv'
data = pandas.read_csv(input_file)
cleaned_data = data.dropna()
sorted_data = cleaned_data.sort_values('age', ascending=False)
cleaned_data.to_csv('cleaned_data.csv', index=False)
```

### pandas_csv_clean (Profile D, Emb: embeddinggemma-300m, LLM: qwen2.5-coder-1.5b-instruct)
**Error**:
```
Validation failed: Ages are not sorted in descending order
```
**Generated Code**:
```python
import pandas

input_file = 'data.csv'
data = pandas.read_csv(input_file)
cleaned_data = data.dropna()
sorted_data = cleaned_data.sort_values('age', ascending=False)
cleaned_data.to_csv('cleaned_data.csv', index=False)
```

### pandas_csv_clean (Profile D, Emb: embeddinggemma-300m, LLM: Qwen2.5-Coder-7B-Instruct-GGUF)
**Error**:
```
Validation failed: Ages are not sorted in descending order
```
**Generated Code**:
```python
import pandas as pd

input_file = 'data.csv'
df = pd.read_csv(input_file)
df_cleaned = df.dropna()
df_sorted = df_cleaned.sort_values('age', ascending=False)
df_cleaned.to_csv('cleaned_data.csv', index=False)
```
