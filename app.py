import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

model=pickle.load(open("/content/model.pkl","rb"))
scaler=pickle.load(open("/content/minmax.pkl","rb"))






st.markdown("<h1 style='text-align: center;'>Cancer Cell Classification </h1>", unsafe_allow_html=True)
st.markdown("## User Input Features")
radius_mean = st.slider("radius_mean", 1.0, 20.0, 10.0, step=0.1, key='radius_mean_slider')
texture_mean = st.slider("texture_mean", 1.0, 20.0, 10.0, step=0.1, key='texture_mean_slider')
perimeter_mean = st.slider("perimeter_mean", 1.0, 20.0, 10.0, step=0.1, key='perimeter_mean_slider')


area_mean = st.slider("area_mean", 1000.0, 5000.0, 2000.0, step=0.1, key='area_mean_slider')


smoothness_mean = st.slider("smoothness_mean", 0.0, 0.1634, 0.08, step=0.001, key='smoothness_mean_slider')


compactness_mean = st.slider("compactness_mean", 0.0, 0.3454, 0.2, step=0.001, key='compactness_mean_slider')


concavity_mean = st.slider("concavity_mean", 0.0, 0.4268, 0.3, step=0.001, key='concavity_mean_slider')


concave_points_mean = st.slider("concave_points_mean", 0.0, 0.2012, 0.1, step=0.001, key='concave_points_mean_slider')


symmetry_mean = st.slider("symmetry_mean", 0.0, 0.304, 0.15, step=0.001, key='symmetry_mean_slider')


fractal_dimension_mean = st.slider("fractal_dimension_mean", 0.0, 0.09744, 0.05, step=0.001, key='fractal_dimension_mean_slider')

radius_se = st.slider("radius_se", 0.0, 2.873, 1.5, step=0.01, key='radius_se_slider')

texture_se = st.slider("texture_se", 0.0, 4.885, 2.0, step=0.01, key='texture_se_slider')


perimeter_se = st.slider("perimeter_se", 0.0, 21.98, 10.0, step=0.1, key='perimeter_se_slider')

area_se = st.slider("area_se", 0.0, 542.2, 200.0, step=1.0, key='area_se_slider')


smoothness_se = st.slider("smoothness_se", 0.0, 0.03113, 0.015, step=0.0001, key='smoothness_se_slider')


compactness_se = st.slider("compactness_se", 0.0, 0.1354, 0.05, step=0.0001, key='compactness_se_slider')


concavity_se = st.slider("concavity_se", 0.0, 0.396, 0.2, step=0.001, key='concavity_se_slider')


concave_points_se = st.slider("concave_points_se", 0.0, 0.05279, 0.025, step=0.0001, key='concave_points_se_slider')

symmetry_se = st.slider("symmetry_se", 0.0, 0.07895, 0.04, step=0.0001, key='symmetry_se_slider')

fractal_dimension_se = st.slider("fractal_dimension_se", 0.0, 0.02984, 0.015, step=0.0001, key='fractal_dimension_se_slider')

radius_worst = st.slider("radius_worst", 0.0, 36.04, 18.0, step=0.1, key='radius_worst_slider')

texture_worst = st.slider("texture_worst", 0.0, 49.0, 25.0, step=0.1, key='texture_worst_slider')

perimeter_worst = st.slider("perimeter_worst", 0.0, 250.0, 125.0, step=0.1, key='perimeter_worst_slider')

area_worst = st.slider("area_worst", 0.0, 4200.0, 2000.0, step=10.0, key='area_worst_slider')

smoothness_worst = st.slider("smoothness_worst", 0.0, 0.2226, 0.1, step=0.0001, key='smoothness_worst_slider')

compactness_worst = st.slider("compactness_worst", 0.0, 1.058, 0.5, step=0.0001, key='compactness_worst_slider')

concavity_worst = st.slider("concavity_worst", 0.0, 1.252, 0.6, step=0.0001, key='concavity_worst_slider')

concave_points_worst = st.slider("concave_points_worst", 0.0, 0.291, 0.1, step=0.0001, key='concave_points_worst_slider')

symmetry_worst = st.slider("symmetry_worst", 0.0, 0.6638, 0.3, step=0.0001, key='symmetry_worst_slider')

fractal_dimension_worst = st.slider("fractal_dimension_worst", 0.0, 0.2075, 0.1, step=0.0001, key='fractal_dimension_worst_slider')

user_input=[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]





user_input = np.array(user_input).reshape(1, len(user_input))
user_input_scaled = scaler.transform(np.array(user_input).reshape(1, -1))

user_input_array = np.array(user_input)
print("Length of user_input:", len(user_input))
user_input_array = user_input_array.reshape(1, -1)

user_input_scaled = scaler.transform(user_input_array)



if st.button("predict"):
   prediction=model.predict(user_input)
   st.subheader("prediction")
   if prediction[0] == 1:
    st.write("cancer is Belign")
   else:
    st.write("cancer is malignant") 
