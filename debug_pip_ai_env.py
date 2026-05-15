import subprocess
import os
import sys

# We want to use the ai_env python
python_exe = os.path.join(os.getcwd(), "ai_env", "Scripts", "python.exe")

def run_pip():
    print(f"Starting verbose pip install using {python_exe}...")
    process = subprocess.Popen(
        [python_exe, "-m", "pip", "install", "flask", "python-dotenv", "flask-cors", "flask-sqlalchemy", "--progress-bar", "off"],
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
