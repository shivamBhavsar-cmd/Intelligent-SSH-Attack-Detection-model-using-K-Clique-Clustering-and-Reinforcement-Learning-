import numpy as np
import pandas as pd
import os

# Function to load the saved Q-table
def load_q_table(file_path):
    try:
        q_table = np.load(file_path)
        print(f"Q-table loaded successfully from {file_path}")
        return q_table
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return None

# Function to predict using the Q-table
def predict_with_q_table(q_table, X):
    print("Making predictions using Q-table...")
    
    # Initialize an empty list for predictions
    predictions = []

    # Iterate over each data point (state) in the input
    for state in range(X.shape[0]):
        action = np.argmax(q_table[state, :])  # Choose the best action (prediction)
        predictions.append(action)

    return np.array(predictions)

# Function to evaluate predictions (optional, for comparison with true labels)
def evaluate_predictions(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Load and process new data (or the same data)
def load_new_data(file_path):
    print(f"Loading new data from {file_path}...")
    df = pd.read_csv(file_path)

    # Detect the label column and preprocess the data (same as training)
    label_column = detect_label_column(df)
    if label_column:
        print(f"Detected label column: {label_column}")
        X, y = preprocess_data(df, label_column)
        return X, y
    else:
        print(f"No label column detected in {file_path}")
        return None, None

# Main Execution
q_table_file = 'q_table.npy'  # The saved Q-table file
new_data_file = 'MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'  # Path to the new data CSV file

# Load the Q-table
q_table = load_q_table(q_table_file)

# Load and preprocess new data (or the same data used for training)
X_new, y_new = load_new_data(new_data_file)

if q_table is not None and X_new is not None:
    # Make predictions using the Q-table
    y_pred = predict_with_q_table(q_table, X_new)

    # Optionally evaluate the predictions (if you have true labels to compare with)
    if y_new is not None:
        evaluate_predictions(y_new, y_pred)

    # Display the predictions
    print("Predicted labels:", y_pred)
else:
    print("Could not load Q-table or data for predictions.")
