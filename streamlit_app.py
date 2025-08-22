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
  bill_length_mm = st.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
