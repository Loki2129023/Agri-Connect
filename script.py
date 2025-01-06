import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample data
data = {
    "crop": ["wheat", "rice", "corn", "barley"],
    "price": [200, 300, 250, 180],
}

# Encode crop names as integers for simplicity
crop_to_int = {crop: idx for idx, crop in enumerate(data["crop"])}
X = [[crop_to_int[crop]] for crop in data["crop"]]
y = data["price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and crop_to_int mapping
with open("crop.pkl", "wb") as f:
    pickle.dump((model, crop_to_int), f)
print(model)
print(crop_to_int)
print("Model saved successfully!")
