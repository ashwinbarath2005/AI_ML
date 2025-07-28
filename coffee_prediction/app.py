import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# Dataset
data= {
    'Day': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7','D8', 'D9', 'D10'],
    'Weather' : ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Rainy'],
    'TimeOfDay': ['Morning', 'Morning', 'Afternoon', 'Afternoon', 'Evening', 'Morning', 'Morning', 'Afternoon', 'Evening', 'Morning'],
    'SleepQuality' : ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor'],
    'Mood': ['Tired', 'Fresh', 'Tired', 'Energetic', 'Tired', 'Fresh', 'Tired', 'Tired', 'Energetic', 'Tired'],
    'BuyCoffee': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

df = df.drop('Day', axis=1)

# Encoding
df_encoded = df.copy()
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df_encoded.drop('BuyCoffee', axis=1)
y = df_encoded['BuyCoffee']

# Train model
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Streamlit UI
st.title("Coffee Purchase Prediction (ID3 Decision Tree)")

st.sidebar.header("Input Your Conditions")
def user_input():
    weather = st.sidebar.selectbox("Weather", df['Weather'].unique())
    time = st.sidebar.selectbox("Time Of Day", df['TimeOfDay'].unique())
    sleep = st.sidebar.selectbox("Sleep Quality", df['SleepQuality'].unique())
    mood = st.sidebar.selectbox("Mood", df['Mood'].unique())
    return pd.DataFrame([[weather, time, sleep, mood]], columns=['Weather', 'TimeOfDay', 'SleepQuality', 'Mood'])

input_df = user_input()

# Encode input
input_encoded = input_df.copy()
for col in input_encoded.columns:
    input_encoded[col] = label_encoders[col].transform(input_encoded[col])

# Prediction
prediction = model.predict(input_encoded)[0]
prediction_label = label_encoders['BuyCoffee'].inverse_transform([prediction])[0]

st.subheader("Prediction:")
st.success(f"The model predicts: {prediction_label}")

st.subheader("Your Input:")
st.write(input_df)

st.subheader("Training Data:")
st.dataframe(df)

# Plot the decision tree
st.subheader("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
plot_tree(model, feature_names=X.columns, class_names=label_encoders['BuyCoffee'].classes_, filled=True)
st.pyplot(fig)