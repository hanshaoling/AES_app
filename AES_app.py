import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
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
st.subheader('https://github.com/hanshaoling/AES_app')

# dict for essay prompts
prompt_dict={1: "Effects computers have on people",
             2: "Censorship in the Libraries",
             3: "ROUGH ROAD AHEAD: Do Not Exceed Posted Speed Limit, by Joe Kurmaskie",
             4: "Winter Hibiscus, by Minfong Ho",
             5: "Narciso Rodriguez, from Home: The Blueprints of Our Lives",
             6: "The Mooring Mast, by Marcia Amidon Lüsted",
             7: "Patience",
             8: "Benefits of laughter"}

# inverse dict for essay prompts
prompt_inv_dict={}
for key, value in prompt_dict.items():
    prompt_inv_dict[value]=key

# dict for each prompt's scale
low_scale={1:2,2:1,3:0,4:0,5:0,6:0,7:0,8:0}
high_scale={1:12,2:6,3:3,4:3,5:4,6:4,7:30,8:60}

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

@st.cache
def read_data():
    data=pd.read_excel('training_set_rel3.xls')
    data=data[['essay_set','essay']]
    return data

def main():
    st.sidebar.title('Navigation')
    add_selectbox = st.sidebar.radio(
        "Go to",
        ["Introduction", "Essay Grader", "Acknowledgement"]
        )

    if add_selectbox=='Introduction':
        st.header('Introduction')
        st.markdown(
            """
Unlike multiple choice questions in standardized tests that have clear right and wrong answers, 
essays are a great way to measure academic performance by allowing students to share their thoughts on open-ended questions. 
Essays promote diverse thinking and enable instructors to see how well students can think critically and apply the learned concepts in life. 
However, it is hard to have many essay questions in an exam especially in national exams 
since essays are time consuming for teachers to grade by hand as it usually requires several graders 
to grade one essay in order to produce a reliable score. 
Therefore, automated essay scoring (AES) systems can provide a potential mitigation to this problem. 
If carefully developed, having a few essay questions is feasible for many test organizations 
since the AES is able to score those essays in a fast and objective way. 
Thus, the goal is to build a model that predicts the essay scores that are consistent with human graders.
            """
        )
        st.header('Dataset overview')
        st.markdown(
            """
The dataset used to train this AES model is the [ASAP data](https://www.kaggle.com/c/asap-aes/data) from Kaggle. 
There are 8 essay prompts with over 12k essays in total and around 1500 essays for each prompt, except the 8th prompt. 
These essays are written by students from Grade 7 to Grade 10. The prompts are either persuasive or source dependent, 
and each prompt has its own scoring scale. More details for these prmopts are available in **Essay Grader** part.
            """
        )
        st.header('Model setting')
        st.markdown(
            """
A recurrent neural network (RNN) with 1 layer of **Glove** word embedding (100-D) and 2 layers of **GRU** (64 & 32 units, respectively)
is trained. **Sigmoid** activation for output is used to restrict the score within 0-1 range 
and rescaled to the original scale depending on specific prompt.   

To allow the model to tell difference of essays for different prompts, models are trained for each prompt separately. 
For purpose of data augmentation, essays from **other prompts** are sampled and added to each training sets, with score labeled as 0.
            """
        )
        st.header('Application')
        st.markdown(
            """
Now let's go to the **Essay Grader** from the sidebar navigation and try our model! 
            """
        )

    if add_selectbox == 'Essay Grader':
        prompt=st.selectbox('Please select Essay prompt:', [prompt_dict[x] for x in range(1,9)])
        prompt_id=prompt_inv_dict[prompt]
        filename=f'Essay Set #{prompt_id}--ReadMeFirst.docx'
        st.markdown(get_binary_file_downloader_html(filename, 'essay prompt description doc'), unsafe_allow_html=True)
        low=low_scale[prompt_id]
        high=high_scale[prompt_id]
        st.markdown(f"The score scale of this prompt is from {low} to {high}")

        data=read_data()
        data=data[data.essay_set==prompt_id]
        
        example=data.essay.sample(n=1).values[0]
        
        input_essay=st.text_area('Essay input', height=64) 

        weight_path=f'./glo_gru_weights_prompt_{prompt_id}/glo_gru_weights_prompt_{prompt_id}'
        model=get_model(weight_path)
        if st.button('Grade this essay'):             
            score=model.predict([input_essay])[0][0]
            score=score*(high-low)+low
            score=score.round(0).astype(int)
            st.markdown(f"The automatic score of this essay is {score}")

        if st.button('Try a random example of this prompt'):
            input_essay=st.text_area('Essay input', example, height=64) 
            score=model.predict([example])[0][0]
            score=score*(high-low)+low
            score=score.round(0).astype(int)
            st.markdown(f"The automatic score of this essay is {score}")

    if add_selectbox == 'Acknowledgement':
        st.header('Acknowledgement')
        st.markdown(
            """
This app was produced as part of the final project for [Harvard’s AC295 Fall 2020](https://harvard-iacs.github.io/2020F-AC295/) course.   
Thanks for team members **Duo Zhang, Erin Yang, Wenjie Gu**.   
For more details about our project design, EDA, etc., 
please refer to our previous [medium post](https://duozhang-75134.medium.com/automated-essay-grading-7bc6cb8ac0b5)
            """
        )

    
if __name__ == '__main__':
    main()



    




