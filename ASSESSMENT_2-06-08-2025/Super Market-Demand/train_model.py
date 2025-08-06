import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("demand_data.csv")

# Encode categorical features
le_product = LabelEncoder()
le_store = LabelEncoder()
df['product'] = le_product.fit_transform(df['product'])
df['store'] = le_store.fit_transform(df['store'])
df['day'] = pd.to_datetime(df['date']).dt.dayofweek  # Monday=0

# Features and target
X = df[['product', 'store', 'promotion', 'day']]
y = df['demand']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model + encoders
joblib.dump(model, 'model.pkl')
joblib.dump(le_product, 'product_encoder.pkl')
joblib.dump(le_store, 'store_encoder.pkl')
print(" Model and encoders saved")
