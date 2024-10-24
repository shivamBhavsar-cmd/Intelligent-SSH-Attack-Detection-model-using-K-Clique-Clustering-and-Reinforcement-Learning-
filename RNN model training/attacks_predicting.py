import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Load Q-table from the saved file
def load_q_table(file_path):
    try:
        q_table = np.load(file_path)
        print("Q-table loaded successfully from", file_path)
        return q_table
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return None

# Function to load and process new CSV file for prediction
def load_new_data(file_path):
    print(f"Loading new data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        label_column = detect_label_column(df)
        if label_column:
            print(f"Detected label column: {label_column}")
            X_new, y_new = preprocess_data(df, label_column)
            return X_new, y_new
        else:
            print("No label column detected in new data.")
            return None, None
    except Exception as e:
        print(f"Error loading new data: {e}")
        return None, None

# Detect the label column dynamically
def detect_label_column(df):
    common_label_names = ['Label', 'label', 'Target', 'target', 'Class', 'class']
    
    for col in df.columns:
        if col in common_label_names:
            return col

    for col in df.columns:
        if df[col].nunique() <= 2:  # Assuming binary classification
            return col
    
    return None

# Preprocess data
def preprocess_data(df, label_column):
    print("Starting data preprocessing...")
    
    df = df.fillna(0)
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        df[column], label_encoders[column] = pd.factorize(df[column])

    X = df.drop(label_column, axis=1).values
    y = df[label_column].values
    
    print("Data preprocessing completed.")
    return X, y

# Make predictions using the Q-table
def make_predictions(q_table, X_new):
    print("Making predictions using Q-table...")
    
    n_states = q_table.shape[0]
    y_pred = []

    for i in range(X_new.shape[0]):
        state = min(i, n_states - 1)  # Prevent state index out of bounds
        action = np.argmax(q_table[state, :])
        y_pred.append(action)

    y_pred = np.array(y_pred)
    return y_pred

# Calculate accuracy
def calculate_accuracy(y_true, y_pred):
    if y_true is not None:
        accuracy = np.mean(y_true == y_pred) * 100
        print(f"Prediction Accuracy: {accuracy:.2f}%")
    else:
        accuracy = None
    return accuracy

# Analyze misclassified instances
def analyze_misclassifications(y_true, y_pred):
    if y_true is not None:
        misclassified_indices = np.where(y_true != y_pred)[0]
        print(f"Number of misclassified instances: {len(misclassified_indices)}")
        if len(misclassified_indices) > 0:
            print("Misclassified instance indices:", misclassified_indices)
    else:
        print("No true labels available for misclassification analysis.")

# Check class distribution
def check_class_distribution(y_true, y_pred):
    if y_true is not None:
        true_label_counts = pd.Series(y_true).value_counts()
        print("True label distribution:\n", true_label_counts)

    pred_label_counts = pd.Series(y_pred).value_counts()
    print("Predicted label distribution:\n", pred_label_counts)

# Main execution
q_table_file = 'q_table.npy'  # Path to your saved Q-table
csv_file = 'MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'  # Example CSV file

# Load Q-table
q_table = load_q_table(q_table_file)

if q_table is not None:
    # Load new data
    X_new, y_new = load_new_data(csv_file)

    if X_new is not None:
        # Make predictions
        y_pred = make_predictions(q_table, X_new)

        # Calculate accuracy (if true labels are available)
        calculate_accuracy(y_new, y_pred)

        # Check class distribution
        check_class_distribution(y_new, y_pred)

        # Analyze misclassified instances
        analyze_misclassifications(y_new, y_pred)

        # Save predicted labels (optional)
        np.save('predicted_labels.npy', y_pred)
        print("Predicted labels saved to predicted_labels.npy.")
    else:
        print("No new data available for prediction.")
else:
    print("Q-table loading failed. Exiting.")
