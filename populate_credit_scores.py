import joblib
import tensorflow as tf
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler

# Load the model
loaded_model = tf.keras.models.load_model('credit_score_model.keras')

# Load the scaler
scaler = joblib.load('scaler.joblib')

# Connect to MongoDB
client = MongoClient('mongodb+srv://banking:test@testbanking.4fqadcz.mongodb.net/')
db = client['test']
accounts_collection = db['Accounts']

# Retrieve all accounts data
accounts_data = list(accounts_collection.find())

# Convert to DataFrame
accounts_df = pd.DataFrame(accounts_data)

# Extract and create features based on the available data
accounts_df['num_deposits'] = accounts_df['deposit_logs'].apply(len)
accounts_df['num_withdrawals'] = accounts_df['withdraw_logs'].apply(len)
accounts_df['total_deposited'] = accounts_df['deposit_logs'].apply(lambda x: sum(item.get('depositted_amount', 0) for item in x))
accounts_df['total_withdrawn'] = accounts_df['withdraw_logs'].apply(lambda x: sum(item.get('withdrawed_amount', 0) for item in x))
accounts_df['num_transfers_in'] = accounts_df['in'].apply(len)
accounts_df['num_transfers_out'] = accounts_df['out'].apply(len)
accounts_df['total_transferred_in'] = accounts_df['in'].apply(lambda x: sum(item.get('balance_transfered', 0) for item in x))
accounts_df['total_transferred_out'] = accounts_df['out'].apply(lambda x: sum(item.get('balance_transfered', 0) for item in x))

# Define features for prediction
features = ['balance', 'num_deposits', 'num_withdrawals', 'total_deposited', 'total_withdrawn', 'num_transfers_in', 'num_transfers_out', 'total_transferred_in', 'total_transferred_out']

# Scale the features
accounts_df[features] = scaler.transform(accounts_df[features])

# Predict credit scores
predicted_credit_scores = loaded_model.predict(accounts_df[features])

# Update the credit scores in the DataFrame
accounts_df['credit_score'] = predicted_credit_scores

# Update the credit scores in the database
for index, row in accounts_df.iterrows():
    accounts_collection.update_one(
        {'_id': row['_id']},
        {'$set': {'credit_score': float(row['credit_score'])}}
    )

print("Credit scores updated successfully.")
