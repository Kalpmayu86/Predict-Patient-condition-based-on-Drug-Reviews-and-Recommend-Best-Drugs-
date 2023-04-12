#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import numpy as np
from pickle import dump
from pickle import load


# In[2]:


df=pd.read_csv("C:\\Users\\kalpana\\Downloads\\Drug name.csv")


# In[3]:


df.head()


# In[4]:


# Write a Streamlit app that allows the user to enter a review and receive recommendations
st.title("Patient's Condition based on Drug Reviews and Recommend Drugs")
review = st.text_input("Enter a patient review:")


# In[5]:


if st.button("Predict Condition"):
    condition = predict_condition(review)
    drugs = recommend_drugs(condition)
    st.write('The best 3 drugs for the "' +condition+ '" are:',drugs)
    for i, drug in enumerate(drugs):
        st.write(f'{i+1}. {drug}')


# In[6]:


# load the model from disk
model = load(open("Drug.pkl",'rb'))
st.subheader('Predicted Result')
st.subheader('Detected As')


# In[7]:


def predict_condition(review):
    review = vectorizer.transform([review])
    condition = model.predict(review)[0]
    return condition
def recommend_drugs(condition):
    # Filter the drug dataset for the given condition
    condition_data = df[df['condition'] == condition]

    # Group the data by drug and calculate the average rating for each drug
    drug_ratings = condition_data.groupby('drugName')['rating'].mean().reset_index()

    # Sort the drugs by rating in descending order and return the top 3 drugs
    top_drugs = drug_ratings.sort_values(by='rating', ascending=False)['drugName'][:3].tolist()

    return top_drugs

# Write a Streamlit app that allows the user to enter a review and receive recommendations
st.title("Patient's Condition based on Drug Reviews and Recommend Drugs")
review = st.text_input("Enter a patient review:")

if st.button("Predict Condition"):
    condition = predict_condition(review)
    drugs = recommend_drugs(condition)
    st.write('The best 3 drugs for the "' +condition+ '" are:',drugs)
    for i, drug in enumerate(drugs):
        st.write(f'{i+1}. {drug}')


# In[ ]:




