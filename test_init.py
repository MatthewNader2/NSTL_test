import traceback

from main import initialize_async

try:
    initialize_async()
except Exception as e:
    traceback.print_exc()
