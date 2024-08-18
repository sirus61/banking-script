import pymongo
import pandas as pd

# Connect to MongoDB
client = pymongo.MongoClient('mongodb+srv://banking:test@testbanking.4fqadcz.mongodb.net/')
db = client['test']
users_collection = db['Users']
accounts_collection = db['Accounts']

# Retrieve data
users_data = list(users_collection.find())
accounts_data = list(accounts_collection.find())

# Close the connection
client.close()

# Convert to DataFrame
users_df = pd.DataFrame(users_data)
accounts_df = pd.DataFrame(accounts_data)

import tensorflow as tf

# Set the number of threads
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

import pandas as pd
import numpy as np
import random as rd
# Number of samples
num_samples = 1000000

# Create dummy data
np.random.seed(42)
dummy_data = {
    'balance': np.random.randint(0, 500000, num_samples),
    'num_deposits': np.random.randint(0, 100, num_samples),
    'num_withdrawals': np.random.randint(0, 90, num_samples),
    'total_deposited': np.random.randint(0, 500000, num_samples),
    'total_withdrawn': np.random.randint(0, 500000, num_samples),
    'num_transfers_in': np.random.randint(0, 100, num_samples),
    'num_transfers_out': np.random.randint(0, 50, num_samples),
    'total_transferred_in': np.random.randint(0, 500000, num_samples),
    'total_transferred_out': np.random.randint(0, 500000, num_samples),
}

# Create a target credit score (0 to 100) favoring higher balances, more deposits, and fewer withdrawals
dummy_data['credit_score'] = (
    0.4 * (dummy_data['balance'] / 5000) +
    0.3 * (dummy_data['total_deposited'] / 50000) -
    0.2 * (dummy_data['total_withdrawn'] / 50000) +
    0.1 * (dummy_data['num_deposits'] - dummy_data['num_withdrawals'])
).astype(int) + rd.randint(15,20)

# Ensure some data points have low balance and equal withdrawals as deposits with low credit scores
for i in range(450):  # Adjust the number as needed
    dummy_data['balance'][i] = np.random.randint(0, 5000)
    dummy_data['total_deposited'][i] = dummy_data['total_withdrawn'][i]
    dummy_data['num_deposits'][i] = dummy_data['num_withdrawals'][i]
    dummy_data['credit_score'][i] = np.random.randint(0, 30)

dummy_data['credit_score'] = np.clip(dummy_data['credit_score'], 0, 100)

# Convert to DataFrame
dummy_df = pd.DataFrame(dummy_data)

# Save to CSV
dummy_df.to_csv('dummy_data.csv', index=False)

import tensorflow as tf
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dummy data
dummy_df = pd.read_csv('dummy_data.csv')

# Define features and target
features = ['balance', 'num_deposits', 'num_withdrawals', 'total_deposited', 'total_withdrawn', 'num_transfers_in', 'num_transfers_out', 'total_transferred_in', 'total_transferred_out']
target = 'credit_score'

# Scale the features
scaler = StandardScaler()
dummy_df[features] = scaler.fit_transform(dummy_df[features])

# Prepare input and output
X = dummy_df[features].values
y = dummy_df[target].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Batch and prefetch data for better performance
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(64).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Set the number of threads
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# Define the model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')  # Output the credit score directly
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# # Evaluate the model
# loss = model.evaluate(test_dataset)
# print(f"Test Loss: {loss}")
model.save('credit_score_model.h5')
model.save('credit_score_model.keras')
# Save the scaler
import joblib
joblib.dump(scaler, 'scaler.joblib')