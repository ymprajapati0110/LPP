from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and other components
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")  # List of feature column names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Create input vector
        input_vector = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        input_vector.at[0, 'Ram'] = int(data['ram'])
        input_vector.at[0, 'CPU Frequency'] = float(data['cpu_freq'])
        input_vector.at[0, 'Screen Width'] = int(data['screen_width'])
        input_vector.at[0, 'Screen Height'] = int(data['screen_height'])
        input_vector.at[0, 'Weight'] = float(data['weight'])
        input_vector.at[0, 'Memory Amount'] = float(data['memory_amount'])

        input_vector.at[0, f"{data['cpu_brand']}_CPU"] = 1
        input_vector.at[0, f"{data['gpu_brand']}_GPU"] = 1
        input_vector.at[0, data['opsys']] = 1
        input_vector.at[0, f"{data['memory_type']}_Memory"] = 1

        # Scale and predict
        input_scaled = scaler.transform(input_vector)
        prediction = model.predict(input_scaled)[0]
        final_price = float(round(prediction / 100, 2))  # Convert np.float32 to float

        return jsonify({'predicted_price': final_price})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
