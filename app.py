import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# -----------------------------
# Load model and tokenizer
# -----------------------------

model = tf.keras.models.load_model("rnn_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -----------------------------
# Settings
# -----------------------------

max_length = 40

stop_words = set(stopwords.words('english'))

# -----------------------------
# Preprocessing Function
# -----------------------------

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = ' '.join(
        word for word in text.split()
        if word not in stop_words
    )

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post'
    )

    return padded

# -----------------------------
# Emotional Guidance
# -----------------------------

def emotional_guidance(emotion):

    guidance = {

        "Anxiety":
        "Take a short break, practice deep breathing, and focus on calming activities.",

        "Depression":
        "Consider talking to someone you trust and engage in small positive activities today.",

        "Stress":
        "Try relaxation exercises, short walks, or mindfulness meditation.",

        "Normal":
        "You seem emotionally balanced. Maintain healthy habits and self-care.",

        "Suicidal":
        "Please reach out to a trusted person or mental health professional immediately.",

        "Bipolar":
        "Maintain a healthy routine and monitor emotional changes carefully."
    }

    return guidance.get(
        emotion,
        "Take care of your emotional well-being."
    )

# -----------------------------
# SECTION 1 — Header
# -----------------------------

st.title(
    "AI-Based Mental Health Sentiment Monitoring System"
)

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

# -----------------------------
# SECTION 2 — About Project
# -----------------------------

st.header("About the Project")

st.write("""
This application uses Artificial Intelligence and Natural Language Processing (NLP)
to analyze emotional sentiment from user text messages.

The system uses a Simple Recurrent Neural Network (RNN) to learn
sequential emotional patterns from text data.

Applications of Emotional AI include:
- Mental health monitoring
- Early emotional risk detection
- AI-assisted counseling
- Emotional wellness support

RNN models are effective because they remember previous words
through hidden states and learn contextual emotional patterns.
""")

# -----------------------------
# SECTION 3 — User Input
# -----------------------------

st.header("Enter Your Thoughts")

st.write("Sample Inputs:")
st.write("- I feel emotionally exhausted and anxious lately")
st.write("- Today was peaceful and relaxing")
st.write("- I cannot sleep and my thoughts keep racing")

user_input = st.text_area(
    "User Text",
    placeholder="Enter your thoughts or feelings here..."
)

# -----------------------------
# SECTION 4 — Prediction Button
# -----------------------------

if st.button("Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        # Preprocess
        processed_input = preprocess_text(user_input)

        # Prediction
        prediction = model.predict(processed_input)

        predicted_index = np.argmax(prediction)

        predicted_emotion = encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = np.max(prediction) * 100

        # -----------------------------
        # SECTION 5 — Prediction Output
        # -----------------------------

        st.header("Prediction Result")

        st.success(
            f"Emotion Detected: {predicted_emotion}"
        )

        st.info(
            f"Confidence Score: {confidence:.2f}%"
        )

        if predicted_emotion in [
            "Anxiety",
            "Depression",
            "Stress",
            "Suicidal"
        ]:
            st.error(
                "Emotional Status: Negative emotional pattern detected."
            )
        else:
            st.success(
                "Emotional Status: Emotionally stable."
            )

        # -----------------------------
        # SECTION 6 — Visualization
        # -----------------------------

        st.header("Emotion Probability Distribution")

        emotions = encoder.classes_

        probabilities = prediction[0]

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(emotions, probabilities)

        ax.set_xlabel("Emotion")

        ax.set_ylabel("Probability")

        ax.set_title("Sentiment Confidence Graph")

        st.pyplot(fig)

        # -----------------------------
        # SECTION 7 — Emotional Guidance
        # -----------------------------

        st.header("Emotional Wellness Guidance")

        message = emotional_guidance(predicted_emotion)

        st.write(message)