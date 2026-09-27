# tests/fixtures/mock_audio_lib.py
import numpy as np

def load_audio(filepath: str) -> np.ndarray:
    """Load audio signal from audio file."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

def resample_audio(data: np.ndarray) -> np.ndarray:
    """Resample audio signal."""
    return data[::2]

def save_audio(data: np.ndarray, dest_path: str) -> str:
    """Save audio signal to destination file path."""
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write("AUDIO_WAV_BYTES_MOCK")
    return dest_path
