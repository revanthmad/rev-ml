import streamlit as st
import pandas as pd

st.title("🤖 Rev's ML App ")
st.info('⚡ This is an app built on a ML model using streamlit')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('penguins_cleaned.csv')
  df
  st.write('**X**')
  X = df.drop('species',axis=1)
  X
  st.write('**y**')
  y = df.species
  y

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species', use_container_width=True)

# Data Preparation
with st.sidebar:
  st.header('Input Features')
  island = st.selectbox('Island', ('Biscoe','Dream','Torgersen'))
  gender = st.selectbox('Gender',('male','female'))
  bill_length_mm = st.slider('Bill length (mm)', min_value=32.1, max_value=59.6, value=43.9, label_visibility="visible", width="stretch")
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2, label_visibility="visible")
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0, label_visibility="visible")
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0, label_visibility="visible")
  
  # Create a DataFrame for the input features
  data = {
    'island': island,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'gender': gender
  }
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X], axis=0)

with st.expander('Input Features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins




















