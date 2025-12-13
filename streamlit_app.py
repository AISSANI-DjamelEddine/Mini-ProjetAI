import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 1️⃣ Charger le modèle et le scaler sauvegardés
with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 2️⃣ Titre de l'application
st.title("Prédiction du prix d'une maison - California Housing")

# 3️⃣ Saisie des caractéristiques par l'utilisateur
longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=-122.0)
latitude = st.number_input("Latitude", min_value=29.0, max_value=42.0, value=37.0)
housing_median_age = st.number_input("Age médian des habitations", min_value=1, max_value=100, value=20)
total_rooms = st.number_input("Nombre total de pièces", min_value=1, value=1000)
total_bedrooms = st.number_input("Nombre total de chambres", min_value=1, value=200)
population = st.number_input("Population", min_value=1, value=500)
households = st.number_input("Nombre de ménages", min_value=1, value=150)
median_income = st.number_input("Revenu médian", min_value=0.0, value=5.0)

# 4️⃣ Créer un DataFrame avec le bon ordre de colonnes
columns = ['longitude', 'latitude', 'housing_median_age', 
           'total_rooms', 'total_bedrooms', 'population', 
           'households', 'median_income']

X_user = pd.DataFrame([[longitude, latitude, housing_median_age,
                        total_rooms, total_bedrooms, population,
                        households, median_income]], columns=columns)

# 5️⃣ Appliquer le scaler (déjà fit sur X_train)
X_user_scaled = scaler.transform(X_user)

# 6️⃣ Prédire le prix
if st.button("Prédire"):
    prediction = model.predict(X_user_scaled)
    st.success(f"Le prix prédit de la maison est : ${prediction[0]:,.2f}")
