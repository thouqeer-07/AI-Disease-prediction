import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("disease_dataset.csv")
# Identify symptom columns
symptom_cols = [col for col in df.columns if col.startswith("Symptom")]

# Clean symptom text: remove spaces, lowercase, fix typos
for col in symptom_cols:
    df[col] = df[col].astype(str).str.strip().str.lower().str.replace(" ", "_").str.replace("__", "_")

# Get all symptom columns
symptom_cols = [col for col in df.columns if col.startswith("Symptom")]
df[symptom_cols] = df[symptom_cols].fillna("None")

# Encode symptoms
encoder = LabelEncoder()
for col in symptom_cols:
    df[col] = encoder.fit_transform(df[col])

# Encode disease labels
disease_encoder = LabelEncoder()
df["Disease"] = disease_encoder.fit_transform(df["Disease"])

# Train model
X = df[symptom_cols]
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "disease_model.pkl")
joblib.dump(encoder, "symptom_encoder.pkl")
joblib.dump(disease_encoder, "disease_encoder.pkl")

print("✅ Model trained and saved!")
