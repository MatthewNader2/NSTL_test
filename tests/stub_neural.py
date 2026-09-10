"""
Lightweight stubs for the neural stack (torch / faiss / sentence_transformers /
transformers / llama_cpp) so the pure-symbolic engine paths (Profile 0, planner,
unification, synthesis, sandbox) can be exercised in a CPU-only CI environment.

These stubs are ONLY injected by test harnesses; the real engine never imports
this module. Neural profiles (A/C/D/E) require the genuine libraries and model
weights and are not covered by these stubs.
"""
import sys
import types
import numpy as np


def _mk_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def install_neural_stubs() -> None:
    if "torch" in sys.modules and hasattr(sys.modules["torch"], "__nstl_stub__"):
        return

    # ---------------- torch ----------------
    torch = _mk_module("torch")
    torch.__nstl_stub__ = True

    class _Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            pass

        @staticmethod
        def ipc_collect():
            pass

        @staticmethod
        def mem_get_info():
            return (0, 0)

        class OutOfMemoryError(RuntimeError):
            pass

        @staticmethod
        def get_device_properties(_i=0):
            class _P:
                total_memory = 0
            return _P()

    class _Mps:
        @staticmethod
        def is_available():
            return False

    torch.cuda = _Cuda
    torch.backends = types.SimpleNamespace(mps=_Mps)
    torch.Tensor = object
    torch.device = lambda *a, **k: None

    # ---------------- faiss ----------------
    faiss = _mk_module("faiss")
    faiss.__nstl_stub__ = True

    class _IndexFlatIP:
        def __init__(self, dim: int):
            self.dim = dim
            self._vecs = np.zeros((0, dim), dtype=np.float32)
            self.ntotal = 0

        def add(self, matrix):
            self._vecs = np.vstack([self._vecs, np.asarray(matrix, dtype=np.float32)])
            self.ntotal = self._vecs.shape[0]

        def search(self, query, k):
            q = np.asarray(query, dtype=np.float32).reshape(1, -1)
            if self.ntotal == 0:
                return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)
            k = min(k, self.ntotal)
            sims = self._vecs @ q.T
            idx = np.argsort(-sims, axis=0)[:k].flatten()
            return sims[idx].T.reshape(1, -1), idx.reshape(1, -1).astype(np.int64)

    def _write_index(index, path):
        np.save(path, index._vecs)

    def _read_index(path):
        vecs = np.load(path)
        idx = _IndexFlatIP(vecs.shape[1])
        idx.add(vecs)
        return idx

    faiss.IndexFlatIP = _IndexFlatIP
    faiss.write_index = _write_index
    faiss.read_index = _read_index

    # sentence_transformers / transformers / llama_cpp: lazy imports inside
    # model-loading methods; they must fail loudly there, so leave them absent.
