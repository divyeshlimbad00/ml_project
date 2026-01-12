from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pickle

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "cardio_model.pkl")

# Load model (pipeline expected)
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print("Failed to load model:", e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    risk = None
    error = None

    if request.method == "POST":
        # required fields
        required = ["age", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "smoke", "alco", "active"]
        # check presence
        if not all(request.form.get(k, "") != "" for k in required):
            error = "All fields are required. Please fill out the form."
            return render_template("predict.html", risk=risk, error=error)

        try:
            age = float(request.form.get("age"))
            # convert age (years) directly to age_years used in training
            age_years = age
            gender = int(request.form.get("gender", 1))  # default to 1 when not provided

            height = float(request.form.get("height"))
            weight = float(request.form.get("weight"))
            ap_hi = float(request.form.get("ap_hi"))
            ap_lo = float(request.form.get("ap_lo"))
            cholesterol = int(request.form.get("cholesterol"))
            smoke = int(request.form.get("smoke"))
            alco = int(request.form.get("alco"))
            active = int(request.form.get("active"))

            # Build feature vector in the same order used when training
            X = np.array([[age_years, gender, height, weight, ap_hi, ap_lo, cholesterol, smoke, alco, active]])

            if model is None:
                error = "Model not available. Please run the training script to create the model."
            else:
                proba = model.predict_proba(X)[:, 1][0]
                risk = round(float(proba) * 100, 2)
        except Exception as exc:
            error = "Invalid input: " + str(exc)

    return render_template("predict.html", risk=risk, error=error)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    required = ["height","weight","ap_hi","ap_lo","cholesterol","smoke","alco","active"]
    if not all(k in data for k in required):
        return jsonify({"error": "missing fields"}), 400
    X = np.array([[data[k] for k in required]])
    if model is None:
        return jsonify({"error": "model not available"}), 500
    proba = model.predict_proba(X)[:, 1][0]
    return jsonify({"risk": float(proba)})

@app.route("/api/stats", methods=["GET"])
def api_stats():
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "project_week", "cardio_train.csv"), sep=';')
        cardio_counts = df['cardio'].value_counts().sort_index().to_dict()
        cholesterol_counts = df['cholesterol'].value_counts().sort_index().to_dict()
        return jsonify({"cardio": cardio_counts, "cholesterol": cholesterol_counts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
