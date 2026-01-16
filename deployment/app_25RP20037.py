from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("best_heart_disease_model.pkl")
features = joblib.load("model_features.pkl")
classes = joblib.load("class_names.pkl")

def validate_input(data):
    if not isinstance(data, dict):
        return False, "Input must be a JSON object."
    for f in features:
        if f not in data:
            return False, f"Missing feature: {f}"
    return True, None

@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    valid, error = validate_input(data)
    if not valid:
        return jsonify({"error": error}), 400
    try:
        df_input = pd.DataFrame([data], columns=features)
        pred_class = model.predict(df_input)[0]
        pred_proba = model.predict_proba(df_input)[0]
        confidence = float(pred_proba.max())
        prob_dict = dict(zip(classes, [float(f"{p:.4f}") for p in pred_proba]))
        return jsonify({"prediction": pred_class, "confidence": confidence, "probabilities": prob_dict})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    prob_dict = None
    if request.method == "POST":
        input_data = {}
        try:
            for f in features:
                value = request.form.get(f)
                if value is None or value == "":
                    return render_template("index_25RP20037.html", error=f"Missing input: {f}", features=features)
                if f in ["age","trestbps","chol","thalach","oldpeak","ca"]:
                    value = float(value)
                input_data[f] = value
            df_input = pd.DataFrame([input_data], columns=features)
            pred_class = model.predict(df_input)[0]
            pred_proba = model.predict_proba(df_input)[0]
            confidence = float(pred_proba.max())
            prob_dict = dict(zip(classes, [float(f"{p:.4f}") for p in pred_proba]))
            result = pred_class
        except Exception as e:
            return render_template("index_25RP20037.html", error=str(e), features=features)
    return render_template("index_25RP20037.html", result=result, confidence=confidence, probabilities=prob_dict, features=features)

if __name__ == "__main__":
    app.run(debug=True)
