import streamlit as st
import pickle as pkl
from helper import query_preproces
model=pkl.load(open('model.pkl','rb'))

st.title('Find Duplicate')
q1=st.text_input('question 1')
q2=st.text_input('question 2')

model_input=query_preproces(q1, q2)

if st.button('Find'):
    pred=model.predict(model_input)[0]
    if pred:
        st.header('Duplicate')
    else :
        st.header('Not Duplicate')