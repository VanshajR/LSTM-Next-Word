from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Orthogonal


# Load the model and tokenizer
custom_objects = {'Orthogonal': Orthogonal}
model = load_model('next_word_lstm.h5',custom_objects=custom_objects, compile=False)
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_length=20):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) > max_sequence_length:
        token_list = token_list[-max_sequence_length:]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding='pre')
    pred = model.predict(token_list, verbose=0)
    pred_index = np.argmax(pred, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == pred_index:
            return word
    return None

# Streamlit App
st.set_page_config(page_title="Next Word Predictor", layout="centered", page_icon="⏭️")
st.title("Next Word Predictor ⏭️")
st.write("This app predicts the next word in a sentence based on a trained LSTM model using the text of *Hamlet*.")

# User input
input_text = st.text_input("Enter the beginning of a sentence:", "")

# Prediction
if input_text:
    next_word = predict_next_word(model, tokenizer, input_text)
    if next_word:
        st.write(f"**Prediction:** The next word is likely to be **'{next_word}'**.")
    else:
        st.write("Unable to predict the next word. Please try a different input.")
else:
    st.write("Type a sentence to get started!")

# Footer
st.markdown("---")
st.markdown("Model was trained on text from Hamlet, thus the predcited word is also in accordance to said work.")
