import subprocess, sys
from pathlib import Path

SCRIPTS = [
    "preprocess_covid.py",
    "preprocess_ecg.py",
    "preprocess_electricity.py",
    "preprocess_metr-la.py",
    "preprocess_solar.py",
    "preprocess_traffic.py",
    "preprocess_wiki.py",
]

def run_script(script):
    script_path = Path(script)
    if not script_path.exists():
        raise FileNotFoundError(f"Script {script} not found.")
    
    print(f"\n Running {script}...")
    result = subprocess.run(
        [sys.executable, script],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script}")
    print(f" Finished {script}.")
    
def main():
    for script in SCRIPTS:
        run_script(script)
    print("\n All Data Preprocessing Completed")
    
if __name__ == "__main__":
    main()