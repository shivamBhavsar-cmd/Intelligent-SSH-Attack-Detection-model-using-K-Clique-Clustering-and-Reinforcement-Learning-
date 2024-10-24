import os
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Function to load data from the given directory
def load_data(directory):
    print(f"Loading data from directory: {directory}...")
    all_dataframes = []

    try:
        # Loop through each file in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                print(f"Successfully loaded {filename}.")
                
                # Read CSV using Dask for large datasets, assume missing for integer columns
                df = dd.read_csv(file_path, assume_missing=True)
                all_dataframes.append(df)
    except Exception as e:
        print(f"Error loading files: {e}")
        return None

    # Concatenate all dataframes into one
    print("Concatenating data frames...")
    combined_df = dd.concat(all_dataframes, axis=0)
    
    return combined_df

# Preprocess Data
def preprocess_data(df):
    print("Starting data preprocessing...")
    
    # Print available columns
    print("Available columns in the dataset:", df.columns)

    # Fill NaN values with 0
    df = df.fillna(0)
    
    # Convert categorical variables to numerical using pandas for better control
    df = df.compute()  # Convert Dask dataframe to Pandas for processing
    label_encoders = {}
    
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    
    # Replace 'Label' with the actual name of your target column
    X = df.drop(' Label', axis=1)  # Ensure the space in column name is handled
    y = df[' Label']
    
    print("Data preprocessing completed.")
    
    return X, y

# Dummy function to represent training (replace with actual training code)
class Agent:
    def __init__(self):
        self.q_table = {}

    def train(self, X, y):
        print("Training the agent...")
        # Simulate training
        self.q_table = {i: np.random.random() for i in range(len(X))}
        print("Training completed.")
        return self

def train_agent(X, y):
    agent = Agent()
    return agent.train(X, y)

# Main Execution
directory = 'MachineLearningCVE'
df = load_data(directory)

if df is not None:
    X, y = preprocess_data(df)
    trained_agent = train_agent(X, y)

    # Save the Q-table
    np.save('q_table.npy', trained_agent.q_table)
    print("Q-table saved successfully.")
else:
    print("No data available for training.")
