import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import LSTM, Embedding, Dense, GRU, Input
from tensorflow.keras import layers
from zipfile import ZipFile

import os
import base64

st.title("Automated Essay Scoring Web App")
st.subheader('Shaoling Han | shaolinghan@hsph.harvard.edu')
st.subheader('https://github.com/hanshaoling')

# dict for essay prompts
prompt_dict={1: "Effects computers have on people",
             2: "Censorship in the Libraries",
             3: "ROUGH ROAD AHEAD: Do Not Exceed Posted Speed Limit, by Joe Kurmaskie",
             4: "Winter Hibiscus, by Minfong Ho",
             5: "Narciso Rodriguez, from Home: The Blueprints of Our Lives",
             6: "The Mooring Mast, by Marcia Amidon Lüsted",
             7: "Patience",
             8: "Benefits of laughter"}

prompt_inv_dict={}
for key, value in prompt_dict.items():
    prompt_inv_dict[value]=key

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

#@st.cache

@st.cache
def get_model(prompt_id):
    model_path=f'glo_gru_prompt_{prompt_id}'
    return tf.keras.models.load_model(model_path)
    
with ZipFile('glo_gru_prompt_1.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()

st.sidebar.title('Navigation')
add_selectbox = st.sidebar.radio(
    "Go to",
    ["Introduction", "Essay Grader", "Acknowledgement"]
    )

if add_selectbox=='Introduction':
    st.title('hello')


if add_selectbox == 'Essay Grader':
    prompt=st.selectbox('Please select Essay prompt:', [prompt_dict[x] for x in range(1,9)])
    prompt_id=prompt_inv_dict[prompt]
    filename=f'Essay Set #{prompt_id}--ReadMeFirst.docx'
    st.markdown(get_binary_file_downloader_html(filename, 'essay prompt description doc'), unsafe_allow_html=True)
    model=get_model(prompt_id)
    




