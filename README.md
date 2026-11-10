
# Human Activity Recognition Using Smartphone Sensors

## Module
Artificial Neural Networks - NB6057CEM

## Project
Human Activity Recognition Using Artificial Neural Networks

## Dataset
UCI Human Activity Recognition Using Smartphones

Source:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The dataset contains 10,299 sensor windows from 30 participants.
There are six human activity classes.

## Models
1. Logistic Regression - Non-neural baseline
2. Manual NumPy MLP - Backpropagation implementation
3. TensorFlow MLP - Feedforward neural network
4. Elman RNN - Recurrent neural network

## Development Environment
Python 3.13
TensorFlow/Keras
NumPy
Pandas
Scikit-learn
Matplotlib
VS Code and Jupyter Notebook

Exact dependency versions are in requirements.txt.

## Project Files
main.ipynb - Practical experiments
predict_har.py - Activity prediction interface
smoke_test.py - Basic software verification
models/ - Trained checkpoints and preprocessing
results/ - Experimental measurements and configurations
figures/ - Generated charts
data/splits/ - Saved participant-separated split indices
data/demo/ - Example inference inputs

## Data Preparation
The original UCI HAR train/test partition was preserved.

A participant-separated validation set was created from
the original training partition using seed 42.

The nine sensor channels were:
- Body acceleration XYZ
- Estimated gravity acceleration XYZ
- Gyroscope XYZ

Each observation contains 128 time steps.

The training-fitted scaler is saved in:
models/sensor_scaler.joblib

## Reproducing the Environment
On Windows, create a Python 3.13 environment:

py -V:3.13 -m venv .venv313

Install dependencies using:

.venv313\Scripts\python.exe -m pip install -r requirements.txt

Download the original dataset from the UCI source above
and extract it into data/UCI HAR Dataset.

The notebook contains data preparation, manual training,
framework models, and experiments. Follow its recorded
execution order to reproduce the experiments.

## Run a Prediction

.venv313\Scripts\python.exe predict_har.py --input data\demo\test_sample_000.npy

## Invalid Input Example

.venv313\Scripts\python.exe predict_har.py --input data\demo\invalid_sample.npy

## Smoke Test

.venv313\Scripts\python.exe smoke_test.py

## Experimental Evidence
See results/step18_repeated_runs.csv
See results/step18_ablation_comparison.csv
See results/step18_robustness.csv
See results/step19_final_test_results.csv
See results/step19_inference_timing.csv

The final model configuration was frozen using validation
evidence before official test evaluation.

## Limitations
The dataset contains recordings from 30 participants
under controlled collection conditions.

Results do not establish production deployment readiness.
Different smartphone placements, devices and environments
may affect prediction quality.
