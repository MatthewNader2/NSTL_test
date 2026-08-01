import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from router import LatticeRouter
from lattice import LatticeOrchestrator
from unification import UnificationGate, ExecutionContext

def main():
    print("Initializing Lattice Orchestrator (SQLite Backend)...")
    try:
        orchestrator = LatticeOrchestrator()
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        sys.exit(1)

    print("Initializing Router and FAISS (Excluding constants)...")
    try:
        router = LatticeRouter(orchestrator)
        print("Router initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize router: {e}")
        sys.exit(1)

    print("Testing AST Unification Logic...")
    try:
        # Dummy snippet without parameters
        snippet_1 = "output_var = cv2.cvtColor(input_var)"
        params = ["cv2.COLOR_BGR2GRAY"]
        result = UnificationGate.inject_parameters(snippet_1, params)
        print(f"Injection Result: {result}")
        assert "COLOR_BGR2GRAY" in result
        
        # Test with kwargs or multiple params
        snippet_2 = "model.compile(optimizer='adam')"
        params_2 = ["loss='categorical_crossentropy'", "metrics=['accuracy']"]
        result_2 = UnificationGate.inject_parameters(snippet_2, params_2)
        print(f"Injection Result 2: {result_2}")
    except Exception as e:
        print(f"Failed AST Injection test: {e}")
        sys.exit(1)

    print("All tests passed.")

if __name__ == "__main__":
    main()
