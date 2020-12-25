import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import LSTM, Embedding, Dense, GRU, Input
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from zipfile import ZipFile
import tempfile
import time

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

def create_model():
    '''
    function to create a template Glove + GRU model
    '''
    max_features = 15000
    sequence_length = 1200
    latent_dim=100
    text_vectorizer = TextVectorization(
        standardize='lower_and_strip_punctuation',
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    model=Sequential()
    model.add(layers.Input(shape = (1,),dtype=tf.string))
    model.add(text_vectorizer)
    model.add(Embedding(max_features, latent_dim, input_length=sequence_length, mask_zero=True))
    model.add(GRU(64, return_sequences=True))
    model.add(GRU(32))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    #model.summary()
    loss = tf.keras.losses.binary_crossentropy
    optimizer = optimizers.RMSprop(learning_rate= 0.0006)
    metrics=['mse']
    model.compile(loss= loss, optimizer = optimizer, metrics = metrics)
    return model

# cannot use @st.cache here because the models with text vectorization cannot be properly dumped with pickle
def get_model(path):
    '''
    load weights for models
    '''
    model=create_model()
    model.load_weights(path)
    return model

def main():
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


        input_essay=st.text_area('Essay') # randomly select a "typical" essay as default?


        if st.button('grading'):
            weight_path=f'./glo_gru_weights_prompt_{prompt_id}/glo_gru_weights_prompt_{prompt_id}'
            model=get_model(weight_path)
            score=model.predict([input_essay])[0][0]
            st.write(score)
            # rescale!
    
if __name__ == '__main__':
    main()



    




