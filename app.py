from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pickle

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cardio_model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print("Model load error:", e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    risk = None
    error = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            gender = int(request.form.get("gender", 1))
            height = float(request.form["height"])
            weight = float(request.form["weight"])
            ap_hi = float(request.form["ap_hi"])
            ap_lo = float(request.form["ap_lo"])
            cholesterol = int(request.form["cholesterol"])
            smoke = int(request.form["smoke"])
            alco = int(request.form["alco"])
            active = int(request.form["active"])

            X = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, smoke, alco, active]])

            if model is None:
                error = "Model not loaded"
            else:
                risk = round(model.predict_proba(X)[0][1] * 100, 2)

        except Exception as e:
            error = str(e)

    return render_template("predict.html", risk=risk, error=error)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    required = ["age","gender","height","weight","ap_hi","ap_lo","cholesterol","smoke","alco","active"]

    if not all(k in data for k in required):
        return jsonify({"error": "missing fields"}), 400

    X = np.array([[data[k] for k in required]])

    if model is None:
        return jsonify({"error": "model not available"}), 500

    risk = model.predict_proba(X)[0][1]
    return jsonify({"risk": float(risk)})
