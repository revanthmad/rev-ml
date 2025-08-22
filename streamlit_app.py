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
