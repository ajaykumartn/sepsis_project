from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the model and scaler
model = load_model('sepsis_model.h5')
scaler = joblib.load('scaler.pkl')  # Load the pre-fitted scaler

# Create users database if it doesn't exist
def init_db():
    with sqlite3.connect('users.db') as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE NOT NULL,
                            password TEXT NOT NULL
                        )''')

init_db()

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
            user = cursor.fetchone()
            if user:
                session['username'] = username
                return redirect(url_for('home'))
            else:
                msg = 'Invalid username or password'
    return render_template('login.html', msg=msg)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            with sqlite3.connect('users.db') as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                msg = 'User registered successfully!'
        except sqlite3.IntegrityError:
            msg = 'Username already exists!'
    return render_template('signup.html', msg=msg)

@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        if 'username' in session:
            return render_template('predict.html', username=session['username'])
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Get form data
            data = request.form.to_dict()

            # Convert form values to float and arrange in correct feature order
            input_data = np.array([[ 
                float(data['PRG']),
                float(data['PL']),
                float(data['PR']),
                float(data['SK']),
                float(data['TS']),
                float(data['M11']),
                float(data['BD2']),
                float(data['Age']),
                float(data['Insurance'])
            ]])

            # Scale input data
            input_data_scaled = scaler.transform(input_data)

            # Predict using the model
            prediction = model.predict(input_data_scaled)
            risk_score = prediction[0][0]

            # Calculate the risk percentage (multiply by 100 to get the percentage)
            risk_percentage = round(risk_score * 100, 2)

            # Positive or Negative feedback
            if risk_score >= 0.5:
                result = f'⚠️ High Risk of Sepsis ({risk_percentage}%)'
                treatment_suggestion = "Immediate medical intervention is necessary. Please consult a healthcare professional."
            else:
                result = f'✅ Low Risk of Sepsis ({risk_percentage}%)'
                treatment_suggestion = "Follow-up monitoring is recommended. Maintain a healthy lifestyle and schedule regular check-ups."

            return render_template('predict.html', 
                                   prediction=result, 
                                   treatment_suggestion=treatment_suggestion,
                                   risk_percentage=risk_percentage, 
                                   username=session.get('username'))

        except Exception as e:
            return render_template('predict.html', 
                                   prediction=f"Error: {str(e)}", 
                                   username=session.get('username'))

if __name__ == '__main__':
    app.run(debug=True)
