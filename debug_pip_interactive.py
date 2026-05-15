import subprocess
import sys

def run_pip():
    print("Starting verbose pip install...")
    process = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", "flask-cors", "flask-sqlalchemy", "--user", "-vvv"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        
    process.stdout.close()
    return_code = process.wait()
    print(f"\nFinished with code {return_code}")

if __name__ == "__main__":
    run_pip()
