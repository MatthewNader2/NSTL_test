# NSTL Prototype Benchmarking Report
**Prompt**: `Read a CSV file named data.csv into a pandas dataframe, drop any rows with missing values, sort it by the 'age' column in descending order, and then save the cleaned dataframe to a new CSV file named cleaned_data.csv.`

## Embedding Model: jina-embeddings-v5-text-nano
### Profile A | LLM: auto
```python
input_source = 'age'
read_csv = pandas.read_csv(input_source)
dataframe_boxplot = read_csv.boxplot()
series_drop = dataframe_boxplot.drop()
```

### Profile C | LLM: qwen2.5-coder-0.5b-instruct
```python
# Planner Error: LLM output was not valid JSON.
```

### Profile D | LLM: qwen2.5-coder-0.5b-instruct
```python
input_source = 'age'
read_csv = pandas.read_csv(input_source)
dataframe_boxplot = read_csv.boxplot()
series_drop = dataframe_boxplot.drop()
```

### Profile C | LLM: qwen2.5-coder-1.5b-instruct
```python
input_source = 'age'
dataframe = pandas.read_csv(input_source)
boxplot = dataframe.boxplot()
sorted_dataframe = boxplot.sort_values()
```

### Profile D | LLM: qwen2.5-coder-1.5b-instruct
```python
input_source = 'age'
dataframe = pandas.read_csv(input_source)
boxplot = dataframe.boxplot()
drop_series = boxplot.drop()
```

### Profile C | LLM: Qwen2.5-Coder-7B-Instruct-GGUF
```python
input_source = 'age'
df = pandas.read_csv(input_source)
boxplot_chart = df.boxplot()
```

### Profile D | LLM: Qwen2.5-Coder-7B-Instruct-GGUF
```python
input_source = 'age'
df = pandas.read_csv(input_source)
boxplot_result = df.boxplot()
cleaned_series = boxplot_result.drop()
```

## Embedding Model: embeddinggemma-300m
### Profile A | LLM: auto
```python
input_source = 'age'
read_csv = pandas.read_csv(input_source)
dataframe_boxplot = read_csv.boxplot()
numpy_any = numpy.any(dataframe_boxplot)
numpy_sort = numpy.sort(numpy_any)
```

### Profile C | LLM: qwen2.5-coder-0.5b-instruct
```python
input_source = 'age'
read_csv = pandas.read_csv(input_source)
```

### Profile D | LLM: qwen2.5-coder-0.5b-instruct
```python
input_source = 'age'
read_csv = pandas.read_csv(input_source)
dataframe_boxplot = read_csv.boxplot()
numpy_any = numpy.any(dataframe_boxplot)
numpy_sort = numpy.sort(numpy_any)
```

### Profile C | LLM: qwen2.5-coder-1.5b-instruct
```python
df = pandas.read_csv('age')
```

### Profile D | LLM: qwen2.5-coder-1.5b-instruct
```python
input_source = 'age'
dataframe = pandas.read_csv(input_source)
boxplot = dataframe.boxplot()
any_value = numpy.any(boxplot)
sorted_values = numpy.sort(any_value)
```

### Profile C | LLM: Qwen2.5-Coder-7B-Instruct-GGUF
```python
input_source = 'age'
df = pandas.read_csv(input_source)
boxplot = df.boxplot()
df_clean = boxplot.drop()
df_sorted = df_clean.sort_values()
```

### Profile D | LLM: Qwen2.5-Coder-7B-Instruct-GGUF
```python
input_source = 'age'
df = pandas.read_csv(input_source)
boxplot_result = df.boxplot()
any_result = numpy.any(boxplot_result)
sorted_result = numpy.sort(any_result)
```

