import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model (in this case, the Q-table)
def load_model(model_path='q_table.npy'):
    try:
        q_table = np.load(model_path, allow_pickle=True).item()
        print("Model loaded successfully.")
        return q_table
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to preprocess the input CSV file
def preprocess_input_data(file_path):
    print(f"Processing input file: {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Fill NaN values with 0 (same preprocessing step as the training phase)
    df = df.fillna(0)
    
    # Convert categorical variables to numerical using LabelEncoder (same as training phase)
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    
    # Drop unnecessary columns (such as labels, if present)
    if ' Label' in df.columns:
        df.drop(' Label', axis=1, inplace=True)
    
    return df

# Function to predict attacks using the model
def predict_attacks(model, data):
    print("Predicting attacks...")
    
    # For simplicity, let's assume the model here is just a Q-table lookup
    predictions = {}
    for i in range(len(data)):
        # Simulate prediction using the Q-table (adjust as per your real model logic)
        attack_type = np.random.choice(['DDoS', 'PortScan', 'Infiltration', 'WebAttack', 'Benign'])  # Dummy choices
        predictions[i] = attack_type

    return predictions

# Function to visualize results
def visualize_results(predictions):
    attack_counts = pd.Series(predictions).value_counts()

    print("Visualizing results...")
    
    # Plot bar chart for attack types
    plt.figure(figsize=(10, 6))
    sns.barplot(x=attack_counts.index, y=attack_counts.values, palette='viridis')
    plt.title('Attack Types Distribution')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.show()
    
    # Plot pie chart for attack types
    plt.figure(figsize=(8, 8))
    plt.pie(attack_counts.values, labels=attack_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
    plt.title('Attack Types Breakdown')
    plt.show()

# Function to provide mitigation steps for each attack type
def provide_mitigation_steps(predictions):
    attack_type_steps = {
        'DDoS': [
            "1. Implement rate limiting.",
            "2. Use firewalls or intrusion detection systems (IDS).",
            "3. Block malicious IPs or patterns.",
            "4. Employ anti-DDoS protection services."
        ],
        'PortScan': [
            "1. Use firewalls to block unused ports.",
            "2. Set up intrusion detection and prevention systems (IDS/IPS).",
            "3. Regularly monitor network traffic for unusual activities."
        ],
        'Infiltration': [
            "1. Update all software to patch vulnerabilities.",
            "2. Use strong authentication and encryption protocols.",
            "3. Monitor for suspicious activity."
        ],
        'WebAttack': [
            "1. Use Web Application Firewalls (WAF).",
            "2. Validate and sanitize user inputs to prevent SQL injection, XSS.",
            "3. Use HTTPS to secure data transmission."
        ],
        'Benign': ["No malicious activity detected."]
    }
    
    # Print mitigation steps based on predictions
    print("Providing mitigation steps...")
    for index, attack_type in predictions.items():
        print(f"Data row {index}: Predicted Attack Type: {attack_type}")
        steps = attack_type_steps.get(attack_type, ["No mitigation steps available for this attack type."])
        for step in steps:
            print(step)
        print("-" * 50)

# Main execution function
def main():
    # Load the pre-trained model
    model = load_model()

    if model is not None:
        # Take user input for raw CSV file path
        file_path = input("Please enter the path to the raw CSV file: ")
        
        # Preprocess the input data
        data = preprocess_input_data(file_path)
        
        # Predict attacks using the loaded model
        predictions = predict_attacks(model, data)
        
        # Visualize the predictions with graphs
        visualize_results(predictions)
        
        # Provide mitigation steps based on predictions
        provide_mitigation_steps(predictions)
    else:
        print("Model not found. Please ensure the model is available.")

if __name__ == "__main__":
    main()
