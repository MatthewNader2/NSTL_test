# NSTL Prototype Benchmarking Report
**Prompt**: `Read a CSV file named data.csv into a pandas dataframe, drop any rows with missing values, sort it by the 'age' column in descending order, and then save the cleaned dataframe to a new CSV file named cleaned_data.csv.`

## Embedding Model: jina-embeddings-v5-text-nano
### Profile A | LLM: auto
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile C | LLM: qwen2.5-coder-0.5b-instruct
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile D | LLM: qwen2.5-coder-0.5b-instruct
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile C | LLM: qwen2.5-coder-1.5b-instruct
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile D | LLM: qwen2.5-coder-1.5b-instruct
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile C | LLM: Qwen2.5-Coder-7B-Instruct-GGUF
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile D | LLM: Qwen2.5-Coder-7B-Instruct-GGUF
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

## Embedding Model: embeddinggemma-300m
### Profile A | LLM: auto
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile C | LLM: qwen2.5-coder-0.5b-instruct
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile D | LLM: qwen2.5-coder-0.5b-instruct
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile C | LLM: qwen2.5-coder-1.5b-instruct
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile D | LLM: qwen2.5-coder-1.5b-instruct
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile C | LLM: Qwen2.5-Coder-7B-Instruct-GGUF
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

### Profile D | LLM: Qwen2.5-Coder-7B-Instruct-GGUF
```python
import pandas as pd

var_1 = pd.read_csv('data.csv')
var_2 = var_1.sort_values(by='age', ascending=True)
var_3 = var_1.sort_values(by='column', ascending=True)
var_4 = var_3.select_dtypes(include=['number']).fillna(value=0.0)
var_5 = var_4.dropna(axis=0, how='any')
var_5.to_csv('cleaned_data.csv', index=False)
```

