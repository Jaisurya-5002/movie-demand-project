import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model/model.pkl", "rb"))

st.title("🎬 Movie Demand Prediction")

genre = st.selectbox("Genre", ["Action", "Drama", "Comedy", "Horror", "Romance"])
rating = st.slider("Rating", 1.0, 10.0, 7.0)
day = st.selectbox("Day", ["Weekday", "Weekend"])
holiday = st.selectbox("Holiday", [0, 1])
temperature = st.slider("Temperature", 20, 40, 30)
shows = st.slider("Shows", 1, 10, 5)

genre_map = {"Action":0,"Drama":1,"Comedy":2,"Horror":3,"Romance":4}
day_map = {"Weekday":0,"Weekend":1}

input_data = np.array([[ 
    genre_map[genre], 
    rating, 
    day_map[day], 
    holiday, 
    temperature, 
    shows
]])

if st.button("Predict"):
    result = model.predict(input_data)[0]
    st.success(f"Expected Bookings: {int(result)}")