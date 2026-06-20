import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


st.markdown("""
<style>
.title-box {
    background-color: #1f3c63;
    padding: 1.2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
}
.result-box {
    background-color: #25b376;
    padding: 1.2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.markdown('<div class="title-box"><h2>🏠 House Price Prediction</h2></div>', unsafe_allow_html=True)
st.write("Fill in the house details below and click Predict.")


gr_liv_area = st.number_input("Living Area (sq ft)", 200, 6000, 1500)
overall_qual = st.slider("Overall Quality (1=Poor, 10=Excellent)", 1, 10, 5)
year_built = st.number_input("Year Built", 1870, 2026, 2000)
total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 4000, 800)
garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
full_bath = st.slider("Full Bathrooms", 0, 4, 2)
bedroom_abv_gr = st.slider("Bedrooms", 0, 8, 3)
yr_sold = st.number_input("Year of Sale", 2000, 2026, 2026)


input_dict = {col: 0 for col in model_columns}

input_dict["GrLivArea"] = gr_liv_area
input_dict["OverallQual"] = overall_qual
input_dict["YearBuilt"] = year_built
input_dict["TotalBsmtSF"] = total_bsmt_sf
input_dict["GarageCars"] = garage_cars
input_dict["FullBath"] = full_bath
input_dict["BedroomAbvGr"] = bedroom_abv_gr
input_dict["YrSold"] = yr_sold
input_dict["HouseAge"] = yr_sold - year_built
input_dict["TotalBath"] = full_bath

input_df = pd.DataFrame([input_dict])[model_columns]

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.markdown(f'<div class="result-box">Estimated Price: ${prediction:,.2f}</div>', unsafe_allow_html=True)

st.caption("AIML Summer Internship 2026 - House Price Prediction System")
