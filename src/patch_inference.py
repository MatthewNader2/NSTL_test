import sys

with open("inference.py", "r", encoding="utf-8") as f:
    code = f.read()

# Add the ONNX mock to the top
mock_code = """import os
import time
import threading
import psutil
import logging
from abc import ABC, abstractmethod
from typing import List

# Mock transformers.onnx for jinaai embedding models compatibility
import sys
if 'transformers.onnx' not in sys.modules:
    import types
    transformers_onnx = types.ModuleType('transformers.onnx')
    transformers_onnx.OnnxConfig = type('OnnxConfig', (object,), {})
    sys.modules['transformers.onnx'] = transformers_onnx

import transformers.pytorch_utils
if not hasattr(transformers.pytorch_utils, 'find_pruneable_heads_and_indices'):
    def find_pruneable_heads_and_indices(*args, **kwargs):
        return set(), []
    transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

import transformers.configuration_utils
if not hasattr(transformers.configuration_utils.PretrainedConfig, 'is_decoder'):
    transformers.configuration_utils.PretrainedConfig.is_decoder = False
if not hasattr(transformers.configuration_utils.PretrainedConfig, 'add_cross_attention'):
    transformers.configuration_utils.PretrainedConfig.add_cross_attention = False"""

target = """import os
import time
import threading
import psutil
import logging
from abc import ABC, abstractmethod
from typing import List"""
code = code.replace(target, mock_code)

target2 = "self.embedder = SentenceTransformer(emb_path, device=emb_device, trust_remote_code=True)"
replacement2 = "self.embedder = SentenceTransformer(emb_path, device=emb_device, trust_remote_code=True, model_kwargs={'low_cpu_mem_usage': False})"
code = code.replace(target2, replacement2)

with open("inference.py", "w", encoding="utf-8") as f:
    f.write(code)
