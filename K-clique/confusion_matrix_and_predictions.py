import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load your pretrained model
model = tf.keras.models.load_model('ssh_attack_model.keras')

# Load your raw data
raw_data_path = 'MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
raw_data = pd.read_csv(raw_data_path)

# Step 1: Check column names
print(raw_data.columns)

# Step 2: Drop the label column and preprocess the data
X_test = raw_data.drop(' Label', axis=1)  # Use the correct label column name (' Label')
y_test = raw_data[' Label']  # Use the correct label column name (' Label')

# Step 3: Use the model to predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert both y_test and y_pred_classes to strings to ensure consistency
y_test = y_test.astype(str)
y_pred_classes = y_pred_classes.astype(str)

# Step 4: Evaluate model performance
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Step 5: Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Step 6: Plot Bar Graph
# Assuming y_test contains the true labels and y_pred_classes contains predicted classes
label_counts = pd.Series(y_test).value_counts()
predicted_counts = pd.Series(y_pred_classes).value_counts()

plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar', color='blue', alpha=0.7, label='True Labels')
predicted_counts.plot(kind='bar', color='orange', alpha=0.7, label='Predicted Labels')
plt.title('Bar Graph of True vs Predicted Labels')
plt.legend()
plt.show()

# Step 7: Plot Histogram
plt.figure(figsize=(8, 6))
plt.hist(y_test, bins=10, alpha=0.7, label='True Labels', color='blue')
plt.hist(y_pred_classes, bins=10, alpha=0.7, label='Predicted Labels', color='orange')
plt.title('Histogram of True vs Predicted Labels')
plt.legend()
plt.show()

# Step 8: Optional - Use GPU to speed up (ensure you have TensorFlow-GPU installed)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f'Using GPU: {physical_devices[0]}')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass  # Invalid device or cannot modify memory growth settings
else:
    print("No GPU found, using CPU.")