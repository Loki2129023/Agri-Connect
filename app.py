from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Enable CORS
app = Flask(__name__)
CORS(app)

# Load the trained model and label encoders
try:
    model = joblib.load('crop_price_model.pkl')
    state_encoder = joblib.load('state_encoder.pkl')
    district_encoder = joblib.load('district_encoder.pkl')
    crop_name_encoder = joblib.load('crop_encoder.pkl')
except Exception as e:
    print(f"Error loading .pkl files: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        state = request.form['state']
        district = request.form['district']
        crop_name = request.form['crop_name']
        
        print(f"Received data - State: {state}, District: {district}, Crop Name: {crop_name}")

        # Check if inputs are valid for each encoder
        if state not in state_encoder.classes_:
            return jsonify({'error': f"State '{state}' not recognized. Recognized values: {list(state_encoder.classes_)}"})
        if district not in district_encoder.classes_:
            return jsonify({'error': f"District '{district}' not recognized. Recognized values: {list(district_encoder.classes_)}"})
        if crop_name not in crop_name_encoder.classes_:
            return jsonify({'error': f"Crop Name '{crop_name}' not recognized. Recognized values: {list(crop_name_encoder.classes_)}"})

        # Encode inputs using the respective encoders
        state_encoded = state_encoder.transform([state])[0]
        district_encoded = district_encoder.transform([district])[0]
        crop_name_encoded = crop_name_encoder.transform([crop_name])[0]

        # Prepare input data for the model
        input_data = np.array([[state_encoded, district_encoded, crop_name_encoded]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return jsonify({'predicted_price': prediction})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Keep it running on port 5000
