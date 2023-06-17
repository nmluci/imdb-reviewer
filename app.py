import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import numpy as np

model = load_model('model/sentichan.h5')

with open('model/tokenizer.json') as f:
    tokenizer = tokenizer_from_json(json.load(f))

emotions = ['Anger', 'Fear', 'Joy', 'Love', 'Sadness', 'Surprise']

def preprocess_text(text):
    text = text.lower()
    text = tokenizer.texts_to_sequences([text])
    text = np.array(text)
    text = tf.keras.preprocessing.sequence.pad_sequences(text, maxlen=40, padding='post', truncating='post')
    return text

def predict_emotion(text):
    text = preprocess_text(text)
    predictions = model.predict(text)[0]
    return emotions[np.argmax(predictions)], predictions

def main():
    st.title("Emotion Sentiment Analysis")
    st.write("Enter a sentence to analyze its emotion:")
    user_input = st.text_input("Input")
    
    if st.button("Analyze"):
        if user_input.strip() != "":
            emotion, probabilities = predict_emotion(user_input)
            st.write("Emotion:", emotion)
            st.write("Emotion Probabilities:")
            for i in range(len(emotions)):
                st.write(f"{emotions[i]}: {probabilities[i]*100:.2f}%")

if __name__ == "__main__":
    main()
