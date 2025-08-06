import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "diabetes_model.pkl")
print(" Model saved as diabetes_model.pkl")
