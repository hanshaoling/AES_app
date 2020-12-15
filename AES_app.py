import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import LSTM, Embedding, Dense, GRU, Input
from tensorflow.keras import layers


st.title("Automated Essay Scoring Web App")
st.subheader('Shaoling Han | shaolinghan@hsph.harvard.edu')
st.subheader('https://github.com/hanshaoling')




def main():
    st.sidebar.title('Navigation')







add_selectbox = st.sidebar.radio(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

if __name__=='__main__':
    main()