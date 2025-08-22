import streamlit as st
import pandas as pd

st.title("🎈 Rev's ML App")

st.write('This is a ML app built with streamlit')

df = pd.read_csv('penguins_cleaned.csv')

df
