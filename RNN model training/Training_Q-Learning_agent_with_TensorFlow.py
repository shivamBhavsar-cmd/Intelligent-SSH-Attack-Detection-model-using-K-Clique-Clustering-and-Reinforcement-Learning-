import os
import pandas as pd
import numpy as np
import tensorflow as tf

# Function to load and process each CSV file
def load_and_process_data(directory):
    print(f"Loading and processing data from directory: {directory}...")
    all_dataframes = []

    try:
        # Loop through each file in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                print(f"Processing {filename}...")
                
                # Load CSV using Pandas
                df = pd.read_csv(file_path)
                label_column = detect_label_column(df)
                if label_column:
                    print(f"Detected label column: '{label_column}' in {filename}")
                    X, y = preprocess_data(df, label_column)
                    all_dataframes.append((X, y))
                else:
                    print(f"No label column found in {filename}. Skipping.")
    
    except Exception as e:
        print(f"Error processing files: {e}")
        return None

    return all_dataframes

# Function to detect the label column dynamically
def detect_label_column(df):
    common_label_names = ['Label', 'label', 'Target', 'target', 'Class', 'class']
    
    # Check for columns with commonly used label names
    for col in df.columns:
        if col in common_label_names:
            return col
    
    # If no common label found, infer the target column by the number of unique values (binary classification assumption)
    for col in df.columns:
        if df[col].nunique() <= 2:  # Assuming binary classification (2 unique values)
            return col
    
    return None

# Preprocess Data using Pandas
def preprocess_data(df, label_column):
    print("Starting data preprocessing...")
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # Convert categorical variables to numerical using Pandas
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        df[column], label_encoders[column] = pd.factorize(df[column])

    # Split into features and label
    X = df.drop(label_column, axis=1).values
    y = df[label_column].values
    
    print("Data preprocessing completed.")
    
    return X, y

# Q-Learning Agent using TensorFlow for GPU acceleration
class QLearningAgentTF:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table as a TensorFlow variable for GPU acceleration
        self.q_table = tf.Variable(tf.zeros([n_states, n_actions], dtype=tf.float32))

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(self.n_actions)  # Explore
        else:
            action = tf.argmax(self.q_table[state, :]).numpy()  # Exploit
        return action

    def learn(self, state, action, reward, next_state):
        # Update Q-value using the Q-learning formula
        predict = self.q_table[state, action]
        target = reward + self.gamma * tf.reduce_max(self.q_table[next_state, :])
        new_value = predict + self.alpha * (target - predict)

        # Update the Q-table at the specific state-action pair
        self.q_table[state, action].assign(new_value)

# Simulate Environment (for illustration, modify for your task)
class Environment:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_states = X.shape[0]  # Each sample can be considered a state
        self.n_actions = 2  # Example: Binary classification (attack/no attack)
        self.current_state = 0  # Starting state index

    def reset(self):
        self.current_state = 0
        return self.current_state

    def step(self, action):
        reward = 0
        done = False

        # Example reward mechanism based on correct/incorrect classification
        if action == self.y[self.current_state]:
            reward = 1  # Reward for correct classification
        else:
            reward = -1  # Penalty for incorrect classification

        self.current_state += 1
        if self.current_state >= self.n_states:
            done = True
            self.current_state = 0  # Reset the environment

        next_state = self.current_state
        return next_state, reward, done

def train_agent(X, y, episodes=5):
    print("Training Q-Learning agent with TensorFlow (GPU)...")

    # Initialize environment and Q-learning agent
    env = Environment(X, y)
    agent = QLearningAgentTF(n_states=env.n_states, n_actions=env.n_actions)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            total_reward += reward
            state = next_state

            if done:
                break

        # Calculate and display the training progress percentage
        progress_percentage = ((episode + 1) / episodes) * 100
        print(f"Training Progress: {progress_percentage:.2f}%")

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

    print("Training completed.")
    return agent


# Main Execution
directory = 'MachineLearningCVE'
all_data = load_and_process_data(directory)

if all_data:
    for X, y in all_data:
        trained_agent = train_agent(X, y)
        
        # Save the Q-table (convert TensorFlow variable to NumPy)
        np.save('q_table.npy', trained_agent.q_table.numpy())
        print("Q-table saved successfully.")
else:
    print("No data available for training.")
