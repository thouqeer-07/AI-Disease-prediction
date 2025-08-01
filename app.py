from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("disease_model.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("Index.html")  # Match uploaded filename

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    # Pad to 17 symptoms
    if len(symptoms) < 17:
        symptoms += ['none'] * (17 - len(symptoms))
    symptoms = symptoms[:17]  # In case user adds more than 17

    # Encode each symptom using the same encoder
    try:
        encoded_symptoms = [symptom_encoder.transform([s])[0] if s in symptom_encoder.classes_ 
                            else symptom_encoder.transform(["none"])[0]
                            for s in symptoms]
    except Exception as e:
        return jsonify({"result": f"Symptom encoding error: {str(e)}"})

    # Create dataframe with column names
    column_names = [f"Symptom_{i+1}" for i in range(17)]
    input_df = pd.DataFrame([encoded_symptoms], columns=column_names)

    try:
        proba = model.predict_proba(input_df)[0]
    except Exception as e:
        return jsonify({"result": f"Prediction error: {str(e)}"})

    # Top 3 predictions
    top_indices = proba.argsort()[-3:][::-1]
    top_diseases = [
        {
            "disease": disease_encoder.inverse_transform([idx])[0],
            "confidence": f"{proba[idx] * 100:.2f}%"
        }
        for idx in top_indices
    ]

    return jsonify({"result": top_diseases})

if __name__ == "__main__":
    app.run(debug=True)
