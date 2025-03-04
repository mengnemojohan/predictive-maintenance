from django.http import JsonResponse
import json
from pymongo import MongoClient
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), 'model_1_cnn_lstm.h5')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_model_1.pkl')
le_path = os.path.join(os.path.dirname(__file__), 'label_encoder_model_1.pkl')

model = keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
le = joblib.load(le_path)

mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['motor_db']
collection = db['motor_data']

def calculate_rul(fault_type, features):
    motor_temp, ambient_temp, vib_x, vib_y, vib_z, volt_a, volt_b, volt_c, curr_a, curr_b, curr_c = features
    fault_rul_map = {
        'Healthy': (40000, 0.0), 'Bearing Defects': (10000, 0.75), 'Radial Misalignment': (12000, 0.7),
        'Mechanical Looseness': (11000, 0.725), 'Rotor Imbalance': (10000, 0.75), 
        'Axial Shaft Misalignment': (13000, 0.675), 'Shaft Bending': (9000, 0.775), 
        'Thermal Expansion': (14000, 0.65), 'Loose Coupling': (11000, 0.725), 
        'Foundation Issues': (12000, 0.7), 'Structural Looseness': (11500, 0.7125), 
        'Resonance': (8000, 0.8), 'Overheating': (15000, 0.625), 'Overcurrent': (8000, 0.8),
        'Undervoltage': (10000, 0.75), 'Phase Imbalance': (9000, 0.775), 'Phase Loss': (2000, 0.95),
        'Phase Reversal': (5000, 0.875), 'Unbalanced Load': (8500, 0.7875)
    }
    base_rul, impact_factor = fault_rul_map.get(fault_type, (20000, 0.5))
    vib_severity = max(0, (max(vib_x, vib_y, vib_z) - 2.5) / 10)
    temp_severity = max(0, (motor_temp - 60) / 100)
    curr_severity = max(0, (max(curr_a, curr_b, curr_c) - 12) / 10)
    volt_severity = max(0, (218.5 - min(volt_a, volt_b, volt_c)) / 50) if min(volt_a, volt_b, volt_c) < 218.5 else 0
    total_severity = min(1.0, vib_severity + temp_severity + curr_severity + volt_severity)
    rul = base_rul * (1 - impact_factor) * (1 - total_severity)
    return max(0, int(rul))

def upload_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            collection.insert_one(data)
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=405)

def get_latest_data(request):
    if request.method == 'GET':
        try:
            data = list(collection.find().sort('_id', -1).limit(1))
            if not data:
                return JsonResponse({'data': [0]*11, 'fault_type': 'No Data', 'rul': 0})
            data = data[0]
            features = [
                data['motor_temp'], data['ambient_temp'],
                data['vib_x'], data['vib_y'], data['vib_z'],
                data['volt_a'], data['volt_b'], data['volt_c'],
                data['curr_a'], data['curr_b'], data['curr_c']
            ]
            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)
            X_3d = X_scaled.reshape((1, 12, 1))
            pred = model.predict(X_3d)
            fault_idx = np.argmax(pred)
            fault_type = le.classes_[fault_idx]
            rul = calculate_rul(fault_type, features)
            return JsonResponse({
                'data': features,
                'fault_type': fault_type,
                'rul': rul
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=405)