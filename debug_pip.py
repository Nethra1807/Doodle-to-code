import subprocess
import sys

try:
    print("Starting pip install...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "flask-cors", "flask-sqlalchemy", "--user"], capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Exit code:", result.returncode)
except Exception as e:
    print("Exception:", str(e))
