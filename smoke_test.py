import subprocess
import sys
from pathlib import Path

required_files = ["predict_har.py","models/sensor_scaler.joblib",  "results/frozen_final_configurations.json", "data/demo/test_sample_000.npy","data/demo/invalid_sample.npy"]

# Verify essential files
for filename in required_files:
    assert Path(filename).exists(), ( f"Missing file: {filename}")
print("Required files: PASS")

# Run valid prediction
valid = subprocess.run([sys.executable,"predict_har.py", "--input", "data/demo/test_sample_000.npy" ],
    capture_output=True, text=True)

assert valid.returncode == 0, valid.stderr
assert "Predicted activity:" in valid.stdout

print("Valid prediction: PASS")

# Run invalid prediction
invalid = subprocess.run([sys.executable,"predict_har.py", "--input","data/demo/invalid_sample.npy"],
    capture_output=True, text=True)

assert invalid.returncode != 0
assert "Invalid shape" in invalid.stderr

print("Invalid input handling: PASS")
print("All smoke tests passed!")
