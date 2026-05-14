"""
run_all.py
==========

Master orchestrator for the full eligibility triage pipeline.

Steps:
1. Generate synthetic data
2. Normalize payer names
3. Apply business rules
4. Launch Streamlit UI

Usage:
    python src/run_all.py
"""

import os
import subprocess
import sys
import time

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "data")

SIMULATE_SCRIPT   = os.path.join(BASE, "simulate_data.py")
NORMALIZE_SCRIPT  = os.path.join(BASE, "normalize.py")
RULES_SCRIPT      = os.path.join(BASE, "rules.py")
STREAMLIT_APP     = os.path.join(BASE, "app.py")
STREAMLIT_BIN = os.path.join(BASE, "..", "venv", "Scripts", "streamlit.exe")



def run_step(name, script):
    print(f"\n===== STEP: {name} =====")
    print(f"Running: {script}\n")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n❌ ERROR in step: {name}")
        sys.exit(1)
    print(f"✅ Completed: {name}\n")
    time.sleep(1)


def main():
    print("\n========================================")
    print(" FULL ELIGIBILITY TRIAGE PIPELINE START ")
    print("========================================\n")

    # 1. Simulate data
    run_step("Simulating synthetic data", SIMULATE_SCRIPT)

    # 2. Normalize payer names
    run_step("Normalizing payer names", NORMALIZE_SCRIPT)

    # 3. Apply business rules
    run_step("Applying eligibility rules", RULES_SCRIPT)

    # 4. Launch Streamlit UI
    print("\n===== STEP: Launching Streamlit App =====")
    print("Opening dashboard...\n")
    subprocess.run([STREAMLIT_BIN, "run", STREAMLIT_APP])


if __name__ == "__main__":
    main()
