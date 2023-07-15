#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[11]:


# Load the dataset
df = pd.read_csv(r"C:\Users\Raj Kumar Vij\Downloads\CAR DETAILS.csv")

X = df.drop('selling_price', axis=1)
y = df['selling_price']

categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner']

# Apply one-hot encoding to the categorical columns
preprocessor = ColumnTransformer(
transformers=[('encoder', OneHotEncoder(), categorical_columns)],
remainder='passthrough')

X_encoded = preprocessor.fit_transform(X)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=1)

# Initialize the Random Forest Regressor
gbr = GradientBoostingRegressor()

# Train the model
gbr.fit(X_train, y_train)

def main():
    st.set_page_config(
        page_title="CAR DEKHO",
        page_icon=":car:",
        layout="centered",
    )
    st.title('Car Selling Price Predictor')
    st.header('Fill the details to predict the Car price')


# In[12]:


import streamlit as st

# Create the Streamlit app
st.title('Car Selling Price Prediction')

# User input for feature values
user_input = {}
for column in X.columns:
    if column in categorical_columns:
        unique_values = df[column].unique()
        user_input[column] = st.selectbox(column, unique_values)
    else:
        user_input[column] = st.number_input(column, value=0)

# Transform user input to one-hot encoding
user_input_encoded = preprocessor.transform(pd.DataFrame(user_input, index=[0]))

# Make predictions using the trained model
prediction = gbr.predict(user_input_encoded)

# Display the prediction
st.subheader('Prediction')
st.write(f'The predicted selling price is: {prediction[0]}')

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Follow me on")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button('GitHub'):
        st.sidebar.markdown('[Visit my GitHub profile and check the codes](https://github.com/pranavcode29)')
with col2:
    if st.sidebar.button('LinkedIn'):
        st.sidebar.markdown('[Visit my LinkedIn profile](https://www.linkedin.com/in/pranavbansal0609/)')

