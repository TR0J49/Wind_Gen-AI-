from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  


def create_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    
    
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date_time TEXT,
                  wind_speed REAL,
                  wind_direction REAL,
                  hour INTEGER,
                  day INTEGER,
                  month INTEGER,
                  predicted_power REAL)''')
    
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  email TEXT NOT NULL UNIQUE,
                  password TEXT NOT NULL)''')
    
    conn.commit()
    conn.close()

create_db()


def load_and_preprocess_data():
    file_path = 'Cleaned Data (2).csv'
    data = pd.read_csv(file_path)
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])
    data['Hour'] = data['Date/Time'].dt.hour
    data['Day'] = data['Date/Time'].dt.day
    data['Month'] = data['Date/Time'].dt.month
    data.drop(columns=['Date/Time', 'Theoretical_Power_Curve (KWh)'], inplace=True)
    return data


def train_model():
    data = load_and_preprocess_data()
    X = data.drop(columns=['LV ActivePower (kW)'])
    y = data['LV ActivePower (kW)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    
    return model, scaler


model, scaler = train_model()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/model', methods=['GET', 'POST'])
def model_page():
    if 'user_id' not in session:
        flash('You need to login to access the Model page.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            data = request.form
            date_time_str = data['date_time']
            wind_speed = float(data['wind_speed'])
            wind_direction = float(data['wind_direction'])

            date_time = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
            hour = date_time.hour
            day = date_time.day
            month = date_time.month

            user_data = np.array([[wind_speed, wind_direction, hour, day, month]])
            user_data_scaled = scaler.transform(user_data)

            prediction = model.predict(user_data_scaled)
            output = prediction[0]

            print(f"Predicted Power: {output}")

            conn = sqlite3.connect('predictions.db')
            c = conn.cursor()
            try:
                c.execute('''INSERT INTO predictions 
                             (date_time, wind_speed, wind_direction, hour, day, month, predicted_power)
                             VALUES (?, ?, ?, ?, ?, ?, ?)''',
                          (date_time_str, wind_speed, wind_direction, hour, day, month, float(output)))  # Ensure output is a float
                conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite error: {e}")
            finally:
                conn.close()

            return render_template('model.html', prediction_text=f'Predicted LV ActivePower (kW): {output:.2f}')
        except Exception as e:
            # Log the error
            print(f"Error: {e}")
            return render_template('model.html', prediction_text=f'Error: {str(e)}')
    else:
        return render_template('model.html')

@app.route('/view_predictions')
def view_predictions():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM predictions')
    rows = c.fetchall()
    conn.close()
    return render_template('view_predictions.html', rows=rows)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):  # Check hashed password
            # Store user ID in session to mark them as logged in
            session['user_id'] = user[0]
            session['user_email'] = user[2]
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        # Hash the password using the default method (pbkdf2:sha256)
        hashed_password = generate_password_hash(password)
        
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                      (name, email, hashed_password))
            conn.commit()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists!', 'error')
            return redirect(url_for('signup'))
        finally:
            conn.close()
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    
    session.pop('user_id', None)
    session.pop('user_email', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/data/<date>')
def get_data(date):
    
    df = pd.read_csv("Cleaned Data (2).csv")
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    
    selected_date = datetime.strptime(date, '%Y-%m-%d').date()
    filtered_data = df[df['Date/Time'].dt.date == selected_date]

    
    data_json = filtered_data.to_dict(orient='records')
    return jsonify(data_json)

@app.route('/dashboard')
def dashboard():
   
    if 'user_id' not in session:
        flash('You need to login to access the Dashboard.', 'error')
        return redirect(url_for('login'))

    
    df = pd.read_csv("Cleaned Data (2).csv")
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    
    unique_dates = df['Date/Time'].dt.date.unique()
    unique_dates = [date.strftime('%Y-%m-%d') for date in unique_dates] 
    return render_template('dash.html', unique_dates=unique_dates)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)