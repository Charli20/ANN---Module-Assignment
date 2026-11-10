
# UCI HAR - Smartphone Activity Prediction
# Loads the saved model and preprocessing
import argparse
import json
import numpy as np
import joblib
import tensorflow as tf

from pathlib import Path

ACTIVITIES = ["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING","LAYING"]

def predict_activity(input_path):

    # Load user input
    sample = np.load( input_path,allow_pickle=False)

    # Validate input shape
    if sample.shape != (128, 9):
        raise ValueError(f"Invalid shape: {sample.shape}. "
            "Expected exactly (128, 9)." )

    # Validate numerical values
    if not np.isfinite(sample).all():
        raise ValueError("Input contains NaN or infinite values.")

    # Load frozen model configuration
    with open("results/frozen_final_configurations.json","r", encoding="utf-8") as file:
        config = json.load(file)

    architecture = config["validation_leader"]

    model_path = config["architectures"][architecture]["checkpoints"]["42"]


    scaler = joblib.load("models/sensor_scaler.joblib" )

    model = tf.keras.models.load_model(model_path)

    # Apply training-fitted preprocessing
    scaled = scaler.transform(sample.astype(np.float32)).astype(np.float32)

    # Match architecture input shape
    if architecture == "MLP":
        model_input = scaled.reshape(1, 1152)
    else:
        model_input = scaled.reshape(1, 128, 9)

    # Generate prediction
    probabilities = model.predict(model_input,verbose=0)[0]

    predicted_id = int(np.argmax(probabilities))

    confidence = float(probabilities[predicted_id])

    print("\nUCI HAR Activity Prediction")
    print("---------------------------")
    print("Model:", architecture)
    print("Input shape:", sample.shape)
    print("Predicted activity:", ACTIVITIES[predicted_id])
    print("Confidence:", f"{confidence * 100:.2f}%")

    return predicted_id


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict smartphone activity" )

    parser.add_argument("--input", required=True, help="Path to a 128x9 NumPy sensor file")

    args = parser.parse_args()

    try:
        predict_activity(args.input)

    except (ValueError, FileNotFoundError, OSError) as error:
        parser.error(str(error))
