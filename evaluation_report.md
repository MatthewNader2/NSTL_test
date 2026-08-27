# NSTL Comprehensive Evaluation Report

## Profile A

**Pass Rate:** 2/5 (40.0%)

| Task | Status | Latency | AST Nodes | Error |
|------|--------|---------|-----------|-------|
| pandas_csv_clean | FAILED | 0.66s | 17 | Traceback (most recent call last):   File "/tmp/nstl_eval_pa |
| opencv_gray_convert | FAILED | 0.46s | 12 | Traceback (most recent call last):   File "/tmp/nstl_eval_op |
| vague_data_transform | FAILED | 0.58s | 12 | Traceback (most recent call last):   File "/tmp/nstl_eval_va |
| long_ml_pipeline | PASSED | 0.75s | 14 |  |
| dijkstra_algorithm | PASSED | 0.10s | 1 |  |

## Profile C

**Pass Rate:** 0/5 (0.0%)

| Task | Status | Latency | AST Nodes | Error |
|------|--------|---------|-----------|-------|
| pandas_csv_clean | FAILED | 6.29s | 49 | rs.py", line 1620, in __init__     self._engine = self._make |
| opencv_gray_convert | FAILED | 5.13s | 40 | [ WARN:0@0.010] global loadsave.cpp:275 findDecoder imread_( |
| vague_data_transform | FAILED | 3.90s | 25 | rs.py", line 1620, in __init__     self._engine = self._make |
| long_ml_pipeline | FAILED | 10.33s | 70 | readers.py", line 1620, in __init__     self._engine = self. |
| dijkstra_algorithm | FAILED | 2.88s | 119 | Traceback (most recent call last):   File "/tmp/nstl_eval_di |

## Profile D

**Pass Rate:** 3/5 (60.0%)

| Task | Status | Latency | AST Nodes | Error |
|------|--------|---------|-----------|-------|
| pandas_csv_clean | FAILED | 1.10s | 14 |  |
| opencv_gray_convert | FAILED | 0.72s | 13 |  |
| vague_data_transform | PASSED | 0.89s | 13 |  |
| long_ml_pipeline | PASSED | 0.57s | 14 |  |
| dijkstra_algorithm | PASSED | 0.10s | 1 |  |

## Profile E

**Pass Rate:** 0/5 (0.0%)

| Task | Status | Latency | AST Nodes | Error |
|------|--------|---------|-----------|-------|
| pandas_csv_clean | FAILED | 5.36s | 49 | rs.py", line 1620, in __init__     self._engine = self._make |
| opencv_gray_convert | FAILED | 4.34s | 40 | [ WARN:0@0.010] global loadsave.cpp:275 findDecoder imread_( |
| vague_data_transform | FAILED | 3.30s | 25 | rs.py", line 1620, in __init__     self._engine = self._make |
| long_ml_pipeline | FAILED | 9.14s | 70 | readers.py", line 1620, in __init__     self._engine = self. |
| dijkstra_algorithm | FAILED | 2.64s | 119 | Traceback (most recent call last):   File "/tmp/nstl_eval_di |

