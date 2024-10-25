from flask import Flask, request, jsonify, render_template
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


data = pd.read_csv('data/Travel details dataset.csv')

def clean_cost_column(column):
    return data[column].replace({'\$': '', ' USD': '', ',': ''}, regex=True).astype(float)

data['Accommodation cost'] = clean_cost_column('Accommodation cost')
data['Transportation cost'] = clean_cost_column('Transportation cost')

def preprocess_data(data):
    label_encoders = {}
    categorical_columns = ['Destination', 'Traveler gender', 'Traveler nationality', 'Accommodation type', 'Transportation type']

    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le 

    return data, label_encoders

data, label_encoders = preprocess_data(data)

X = data[['Destination', 'Duration (days)', 'Traveler age', 'Traveler gender', 'Accommodation type', 'Transportation type']]
y_accommodation_cost = data['Accommodation cost']
y_transportation_cost = data['Transportation cost']

X_train, X_test, y_train_accommodation, y_test_accommodation = train_test_split(X, y_accommodation_cost, test_size=0.2, random_state=42)
X_train, X_test, y_train_transportation, y_test_transportation = train_test_split(X, y_transportation_cost, test_size=0.2, random_state=42)

accommodation_model = RandomForestRegressor()
transportation_model = RandomForestRegressor()

accommodation_model.fit(X_train, y_train_accommodation)
transportation_model.fit(X_train, y_train_transportation)

if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(accommodation_model, 'model/accommodation_model.pkl')
joblib.dump(transportation_model, 'model/transportation_model.pkl')

accommodation_model = joblib.load('model/accommodation_model.pkl')
transportation_model = joblib.load('model/transportation_model.pkl')

@app.route('/')
def home():
    return render_template('itinerary.html')

@app.route('/generate', methods=['POST'])
def generate_itinerary():
    user_input = request.json

    destination = label_encoders['Destination'].transform([user_input['Destination']])[0]
    accommodation_type = label_encoders['Accommodation type'].transform([user_input['Accommodation type']])[0]
    transportation_type = label_encoders['Transportation type'].transform([user_input['Transportation type']])[0]

    input_data = pd.DataFrame({
        'Destination': [destination],
        'Duration (days)': [user_input['Duration (days)']],
        'Traveler age': [user_input['Traveler age']],
        'Traveler gender': [user_input['Traveler gender']],
        'Accommodation type': [accommodation_type],
        'Transportation type': [transportation_type]
    })

    predicted_accommodation_cost = accommodation_model.predict(input_data)[0]
    predicted_transportation_cost = transportation_model.predict(input_data)[0]

    return jsonify({
        'predicted_accommodation_cost': predicted_accommodation_cost,
        'predicted_transportation_cost': predicted_transportation_cost
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
