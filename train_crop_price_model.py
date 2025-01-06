import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv('crop_prices.csv')

# Initialize separate LabelEncoders for each column
state_encoder = LabelEncoder()
district_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

# Preprocess the data
df['State'] = state_encoder.fit_transform(df['State'])
df['District'] = district_encoder.fit_transform(df['District'])
df['Crop Name'] = crop_encoder.fit_transform(df['Crop Name'])

# Features and target variable
X = df[['State', 'District', 'Crop Name']]  # Features
y = df['Price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoders
joblib.dump(model, 'crop_price_model.pkl')
joblib.dump(state_encoder, 'state_encoder.pkl')
joblib.dump(district_encoder, 'district_encoder.pkl')
joblib.dump(crop_encoder, 'crop_encoder.pkl')

print("Model and label encoders saved successfully.")
