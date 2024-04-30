import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import base64

loaded_model = joblib.load('pipe.joblib')

def predict(model, new_data):

    X_new = new_data.drop(columns=['Unnamed: 0'])
    y_new = new_data['median_house_value']
    y_pred_new = model.predict(X_new)
    new_data['predictions'] = y_pred_new
    return new_data, y_new, y_pred_new

def to_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_data.csv">Télécharger le fichier CSV avec prédictions</a>'
    return href

def make_prediction(input_df):
    prediction = loaded_model.predict(input_df)
    return prediction

st.title("Application de prédiction des prix des logements")

st.header("Télécharger le nouveau jeu de données")
uploaded_file = st.file_uploader("Téléchargez votre fichier CSV contenant les données", type="csv")

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)

    st.write("Les premières lignes du nouveau jeu de données :")
    st.write(new_data.head())

    if st.button("Faire des prédictions"):
        predicted_data, y_true, y_pred = predict(loaded_model, new_data)

        st.write("Les prédictions sont terminées!")
        st.write("Le jeu de données avec les prédictions :")
        st.write(predicted_data)

        mse_new = mean_squared_error(y_true, y_pred)
        r2_new = r2_score(y_true, y_pred)

        st.write("RMSE (Nouveau jeu de données):", np.sqrt(mse_new))
        st.write("R^2 Score (Nouveau jeu de données):", r2_new)

        st.markdown(to_csv(predicted_data), unsafe_allow_html=True)

st.title("Entrer les informations")

longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
housing_median_age = st.number_input("Housing Median Age")
total_rooms = st.number_input("Total Rooms")
total_bedrooms = st.number_input("Total Bedrooms")
population = st.number_input("Population")
households = st.number_input("Households")
median_income = st.number_input("Median Income")
ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

if st.button("Faire une prédiction"):
    input_data = [{
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }]

    input_df = pd.DataFrame.from_records(input_data)

    prediction = make_prediction(input_df)

    st.write("La prédiction est :", prediction[0])
