from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Training the model (simple example - in production, you should load a pre-trained model)
data = {
    'heart_rate': [80, 120, 95, 130, 70, 150],
    'resp_rate': [18, 30, 22, 28, 16, 35],
    'temperature': [36.5, 39.0, 37.5, 40.0, 36.0, 41.0],
    'wbc_count': [7000, 17000, 10000, 20000, 6000, 25000],
    'sepsis_label': [0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
X = df[['heart_rate', 'resp_rate', 'temperature', 'wbc_count']]
y = df['sepsis_label']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    heart_rate = data['heart_rate']
    resp_rate = data['resp_rate']
    temperature = data['temperature']
    wbc_count = data['wbc_count']
    
    input_features = np.array([[heart_rate, resp_rate, temperature, wbc_count]])
    prediction = model.predict(input_features)
    
    return jsonify({'sepsis_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
