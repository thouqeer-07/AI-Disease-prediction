import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("disease_dataset.csv")

# Separate features and target
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Encode disease (target) labels
disease_encoder = LabelEncoder()
y_encoded = disease_encoder.fit_transform(y)

# Encode all symptom values to numbers
symptom_encoder = LabelEncoder()

# Flatten all symptom values to fit the encoder
flat_symptoms = pd.unique(X.values.ravel())  # get unique symptom names
symptom_encoder.fit(flat_symptoms)

# Transform the entire dataframe (X) symptom-wise
X_encoded = X.apply(lambda col: symptom_encoder.transform(col))

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_encoded, y_encoded)

# Save everything
joblib.dump(model, "disease_model.pkl")
joblib.dump(symptom_encoder, "symptom_encoder.pkl")
joblib.dump(disease_encoder, "disease_encoder.pkl")

print("✅ Model and encoders saved successfully.")
