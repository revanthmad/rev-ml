import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("🐧 Penguin Species Prediction - Machine Learning app")
st.info('⚡ This is a streamlit-based ML app to make predctions on the trained penguin species based on their two main features i.e., "island" and "gender"')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('penguins_cleaned.csv')
  df
  
  st.write('**X**')
  X_raw = df.drop('species',axis=1)
  X_raw
  
  st.write('**y**')
  y_raw = df.species
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species', use_container_width=True)

# Input features
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
    'sex': gender
  }
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input Features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins
 

# Data preparation
# Encode X
encode = ['island','sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {
  'Adelie': 0,
  'Chinstrap': 1,
  'Gentoo': 2
}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data Preparation'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  y

# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf = clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(
  columns = {
    0: 'Adelie',  
    1: 'Chinstrap',  
    2: 'Gentoo'
  }
)


# Display predictied species
st.subheader('Predicted Species')
st.dataframe(
  df_prediction_proba,
  column_config = {
    'Adelie': st.column_config.ProgressColumn(
      label='Adelie', 
      format='%f', 
      width = 'medium',
      min_value=0, 
      max_value=1
    ),
    'Chinstrap': st.column_config.ProgressColumn(
      label='Chinstrap', 
      format='%f', 
      width = 'medium',
      min_value=0, 
      max_value=1
    ),
    'Gentoo': st.column_config.ProgressColumn(
      label='Gentoo', 
      format='%f', 
      width = 'medium',
      min_value=0, 
      max_value=1
    )
  }
)

penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.success(str(penguins_species[prediction][0]))














