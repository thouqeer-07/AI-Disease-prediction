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
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data["symptoms"]

    # Debugging: Print entered and valid symptoms
    print("🟡 Symptoms entered by user:", symptoms)
    print("✅ Valid symptom options:", list(symptom_encoder.classes_))

    # Pad to 17 symptoms
    if len(symptoms) < 17:
        symptoms += ['None'] * (17 - len(symptoms))

    # Encode symptoms safely
    encoded_symptoms = [
        symptom_encoder.transform([s])[0] if s in symptom_encoder.classes_ 
        else symptom_encoder.transform(["None"])[0]
        for s in symptoms
    ]

    print("🔵 Encoded symptoms:", encoded_symptoms)

    # Predict
    input_df = pd.DataFrame([encoded_symptoms])
    prediction = model.predict(input_df)[0]
    disease = disease_encoder.inverse_transform([prediction])[0]

    return jsonify({"result": f"Predicted Disease: {disease}"})


if __name__ == "__main__":
    app.run(debug=True)
