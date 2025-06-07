from flask import Flask, request, jsonify
import joblib
import sqlite3
import os

app = Flask(__name__)

# Load the trained model with joblib
model = joblib.load('model/model.pkl')

# Create DB if not exists
DB_PATH = 'database/customers.db'
os.makedirs('database', exist_ok=True)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        phone TEXT PRIMARY KEY,
        purchase_frequency INTEGER
    )
''')
conn.commit()
conn.close()

# Loyalty logic
def infer_loyalty(purchase_frequency):
    if purchase_frequency >= 5:
        return 2  # High
    elif purchase_frequency >= 2:
        return 1  # Medium
    else:
        return 0  # Low

# Map satisfaction score to label and reward
def map_satisfaction(score):
    if score >= 8:
        return "Highly Satisfied", 50
    elif score >= 5:
        return "Moderately Satisfied", 30
    else:
        return "Not Satisfied", 10

# Encode feedback_score from string to int
def encode_feedback(score):
    mapping = {'low': 0, 'medium': 1, 'high': 2}
    if not score:
        return 1  # default to medium
    return mapping.get(score.lower(), 1)

# Store last results per phone in memory (dev only)
last_results = {}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Data received:", data)

    phone = data.get('phone')
    product_quality = int(data.get('product_quality', 5))
    service_quality = int(data.get('service_quality', 5))
    feedback_score_raw = data.get('feedback_score')
    feedback_score = encode_feedback(feedback_score_raw)

    # Update purchase frequency in DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT purchase_frequency FROM customers WHERE phone = ?", (phone,))
    row = cursor.fetchone()

    if row:
        purchase_frequency = row[0] + 1
        cursor.execute("UPDATE customers SET purchase_frequency = ? WHERE phone = ?", (purchase_frequency, phone))
    else:
        purchase_frequency = 1
        cursor.execute("INSERT INTO customers (phone, purchase_frequency) VALUES (?, ?)", (phone, 1))
    conn.commit()
    conn.close()

    # Infer loyalty
    loyalty_level = infer_loyalty(purchase_frequency)

    # Prepare features for model prediction
    features = [[
        product_quality,
        service_quality,
        purchase_frequency,
        feedback_score,
        loyalty_level
    ]]

    satisfaction_score = model.predict(features)[0]
    satisfaction_level, reward_points = map_satisfaction(satisfaction_score)

    # Save last result by phone for retrieval
    last_results[phone] = {
        "satisfaction_score": round(satisfaction_score, 2),
        "satisfaction_level": satisfaction_level,
        "reward_points": reward_points
    }

    return jsonify(last_results[phone])

@app.route('/last_result/<phone>', methods=['GET'])
def last_result(phone):
    result = last_results.get(phone)
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "No result found for this phone"}), 404

@app.route('/')
def index():
    return """
    <h2>Customer Feedback API is Running</h2>
    <p>Use POST /predict to test your model via script or Google Form</p>
    """
from flask import render_template

@app.route('/feedback_form')
def feedback_form():
    return render_template('feedback_form.html')

if __name__ == '__main__':
    app.run(debug=True)
