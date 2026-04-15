import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# --- Load and prepare data ---
@st.cache_data
def load_data():
    data = pd.read_csv("Mystery Dataset.csv")
    data = data[['Outcome', 'Class', 'Sex', 'Age']]
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    # Impute missing values in Age
    imputer = SimpleImputer(strategy='median')
    data['Age'] = imputer.fit_transform(data[['Age']])
    return data

@st.cache_resource
def train_model(data):
    X = data[['Class', 'Sex', 'Age']]
    y = data['Outcome']
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Load data and train model
data = load_data()
model = train_model(data)

# --- App UI ---
st.title("Case Cracker")
st.write("Enter a profile to determine outcome likelihood.")

name = st.text_input("Name")
gender = st.selectbox("Gender", ["male", "female"])
pclass = st.selectbox("Class (1 = Highest, 3 = Lowest)", [1, 2, 3])
age = st.slider("Age", 0, 100, 25)

# --- Prediction ---
if st.button("Predict Outcome"):
    gender_encoded = 1 if gender == 'female' else 0
    input_data = pd.DataFrame([[pclass, gender_encoded, age]],
                              columns=['Class', 'Sex', 'Age'])
    # Handle missing values in input
    input_data['Age'] = input_data['Age'].fillna(data['Age'].median())
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    result = "1" if prediction == 1 else "0"
    
    st.subheader(f"Result for {name}:")
    st.write(f"Prediction: {result}")
    st.write(f"Outcome Probability: {probability:.2f}")

# --- Footer ---
st.write("---")
st.caption("This tool uses historical data patterns to generate predictions.")