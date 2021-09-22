#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:28:35 2021

@author: sid
"""
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image



##Opening the file
pickle_in = open("rf.pkl","rb")
rf = pickle.load(pickle_in)

def welcome():
    return ("note forgery classifier")
    



def note_authenticator(variance,skewness,curtosis,entropy):
    prediction = rf.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction







def main():
    st.title("Note Authenticator")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Note Authenticator App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance = st.text_input("Variance","Type Here")
    skewness = st.text_input("skewness","Type Here")
    curtosis = st.text_input("curtosis","Type Here")
    entropy = st.text_input("entropy","Type Here")
    result = ""
    
    if st.button("predict"):
        result = note_authenticator(variance, skewness, curtosis, entropy)
    st.success('The final result is {}'.format(result))  
    if st.button("About"):
        st.text("Built by Siddhanth Bakshi")



if __name__=='__main__':
    main()