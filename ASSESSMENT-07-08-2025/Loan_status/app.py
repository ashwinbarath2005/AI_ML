import streamlit as st
import pickle

# Load the decision tree
with open("loan_status.pkl", "rb") as file:
    tree = pickle.load(file)

# Custom prediction function (copied from training code)
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    value = sample.get(root)
    subtree = tree[root].get(value)
    if isinstance(subtree, dict):
        return predict(subtree, sample)
    else:
        return subtree if subtree else "Unknown"

# Streamlit UI
st.title("Loan Status Prediction (ID3 Decision Tree)")

st.write("Enter applicant details to predict loan approval:")

credit_score = st.selectbox("Credit Score", ['Good', 'Average', 'Poor'])
employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed'])
income_level = st.selectbox("Income Level", ['High', 'Medium', 'Low'])
collateral = st.selectbox("Collateral", ['Yes', 'No'])

if st.button("Predict"):
    input_data = {
        'CreditScore': credit_score,
        'EmploymentType': employment_type,
        'IncomeLevel': income_level,
        'Collateral': collateral
    }

    prediction = predict(tree, input_data)
    
    st.subheader("🔍 Prediction Result:")
    if prediction == 'Approved':
        st.success("✅ Loan Approved")
    elif prediction == 'Rejected':
        st.error("❌ Loan Rejected")
    else:
        st.warning("⚠️ Unable to determine loan status.")
