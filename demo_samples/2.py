from flask import Flask, request, jsonify
import os
import sqlite3
import boto3
import pickle

app = Flask(__name__)

# -------------------- Configurations --------------------

# Hardcoded AWS credentials (BAD PRACTICE)
AWS_ACCESS_KEY = "AKIAEXAMPLEACCESSKEY"
AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
BUCKET_NAME = "public-bucket-simulation"

# In-memory fake database (for SQL Injection)
def init_db():
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (username TEXT, password TEXT, role TEXT)")
    cursor.execute("INSERT INTO users VALUES ('admin', 'admin123', 'admin')")
    cursor.execute("INSERT INTO users VALUES ('user1', 'user123', 'user')")
    conn.commit()
    return conn

db_conn = init_db()

# -------------------- Endpoints --------------------

# Vulnerable Login (Broken Access Control + Cryptographic Failure)
@app.route('/login', methods=['GET', 'POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    cursor = db_conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE username='{username}'")
    user = cursor.fetchone()

    if user and password == user[1]:  # BAD: password compared directly without hashing
        return jsonify({"message": f"Welcome {username}. Role: {user[2]}", "users_data": users_exposed()})
    else:
        return jsonify({"message": "Invalid credentials"}), 401

def users_exposed():
    cursor = db_conn.cursor()
    cursor.execute("SELECT username, role FROM users")
    users = cursor.fetchall()
    return [{"username": u[0], "role": u[1]} for u in users]

# Vulnerable Query (SQL Injection)
@app.route('/query', methods=['GET'])
def query():
    username = request.args.get('username')
    cursor = db_conn.cursor()
    query = f"SELECT * FROM users WHERE username='{username}'"  # BAD: vulnerable to SQL Injection
    cursor.execute(query)
    result = cursor.fetchall()
    return jsonify({"result": result})

# Command Injection (Ping)
@app.route('/ping', methods=['GET'])
def ping():
    target = request.args.get('target')
    os.system(f"ping -c 1 {target}")  # BAD: user input directly in OS command
    return jsonify({"message": f"Pinging {target}"})

# Cloud Misconfiguration Simulation
@app.route('/upload', methods=['POST'])
def upload():
    filename = request.form.get('filename', 'example.txt')
    print(f"Simulated upload of {filename} to bucket {BUCKET_NAME} with keys {AWS_ACCESS_KEY}:{AWS_SECRET_KEY}")
    return jsonify({"message": f"File {filename} uploaded publicly (SIMULATED)", "bucket": BUCKET_NAME})

# Insecure Deserialization (Using pickle)
@app.route('/deserialize', methods=['POST'])
def deserialize():
    try:
        data = request.form.get('payload')  # Get form field
        obj = eval(data)  # CAUTION: Only for demo/testing
        return jsonify({"message": f"Deserialized object: {str(obj)}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400



# Simulated Use-After-Free (CVE-2025-29824)
objects = {}

@app.route('/allocate', methods=['POST'])
def allocate():
    obj_id = request.form.get('id')
    objects[obj_id] = {"data": "Important Secure Data"}
    return jsonify({"message": f"Object {obj_id} allocated."})

@app.route('/free', methods=['POST'])
def free():
    obj_id = request.form.get('id')
    if obj_id in objects:
        del objects[obj_id]
        return jsonify({"message": f"Object {obj_id} freed."})
    else:
        return jsonify({"message": "Object not found."}), 404

@app.route('/use', methods=['GET'])
def use():
    obj_id = request.args.get('id')
    obj = objects.get(obj_id)
    if obj:
        return jsonify({"data": obj['data']})
    else:
        return jsonify({"error": "Use-After-Free: Object no longer exists."}), 400

# -------------------- Main App Run --------------------

if __name__ == '__main__':
    app.run(debug=True)