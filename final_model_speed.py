# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:06:36 2023

@author: Pratik
"""

import pickle
import numpy as np
import streamlit as st

loaded_model = pickle.load(open(r"C:\Users\Pratik\model_deployment\final_trained_model.sav",'rb'))

def RandomForest(input_data):
    input_data_asarray = np.asarray(input_data)
    input_data_reshaped = input_data_asarray.reshape(1,-1) 
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    st.title('Predicting the speed')
    
    ambient = st.text_input('Ambient Temperature')
    coolant = st.text_input('Coolant Temperature')
    u_d = st.text_input('Voltage D-Component')
    u_q = st.text_input('Voltage Q-Component')
    i_d = st.text_input('Current D-Component')
    i_q = st.text_input('Current Q-Component')
    pm = st.text_input('Permanent Magnet Temperature')
    stator_winding = st.text_input('Temperature by Thermal Sensor')
    
    empty = ''
    
    if st.button('Predicting the speed'):
        empty = RandomForest([ambient, coolant, u_d, u_q, i_d, i_q, pm,stator_winding ])
        
    st.success(empty)
    
if __name__ == '__main__':
    main()
    

