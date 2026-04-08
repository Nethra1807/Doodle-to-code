import os
import subprocess
import sys

if __name__ == "__main__":
    # Convenience script to start the Flask backend
    backend_script = os.path.join(os.path.abspath(os.path.dirname(__file__)), "backend", "app.py")
    subprocess.run([sys.executable, backend_script])
