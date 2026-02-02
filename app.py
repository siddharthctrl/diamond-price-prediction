import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎"
)

st.title("💎 Diamond Price Prediction App")
st.write("Predict diamond prices using a trained KNN Regression model")

@st.cache_resource
def load_model():
    with open("diamond_price_knn_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.header("🔹 Enter Diamond Details")

carat = st.number_input("Carat", 0.2, 5.0, 1.0, step=0.01)
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox(
    "Clarity",
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
)
depth = st.number_input("Depth", 50.0, 75.0, 61.0)
table = st.number_input("Table", 50.0, 75.0, 57.0)
x = st.number_input("Length (x)", 0.0, 10.0, 6.0)
y = st.number_input("Width (y)", 0.0, 10.0, 6.0)
z = st.number_input("Height (z)", 0.0, 10.0, 4.0)

if st.button("🔮 Predict Price"):
    input_df = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Diamond Price: ${prediction:,.2f}")
