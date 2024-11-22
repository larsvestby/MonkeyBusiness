import os
import numpy as np
import streamlit as st
import cv2
from PIL import Image
from keras.src.applications.densenet import preprocess_input
from keras.src.saving import load_model
from datetime import datetime

# Path to the trained model
model_path = "MonkeyBusiness/Tensorflow/monkey.keras"

# Streamlit page setup
st.set_page_config(page_title="Monkey Recognition", page_icon="🐒")
st.title("Monkey Recognition 🐒")
st.caption("by Dr. Lars Vestby")

# Load the trained model
model = load_model(model_path)

# Labels for each monkey species
species_labels = [
    "Mantled Howler", "Patas Monkey", "Bald Uakari", "Japanese Macaque",
    "Pygmy Marmoset", "White-Headed Capuchin", "Silvery Marmoset", "Common Squirrel Monkey",
    "Black-headed Night-Monkey", "Nilgiri Langur"
]

# Initialize session state for storing image history, feedback, and accuracy
if "history" not in st.session_state:
    st.session_state["history"] = []  # Stores (image, species, timestamp, feedback)
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None  # Stores the current prediction details
if "total_predictions" not in st.session_state:
    st.session_state["total_predictions"] = 0  # Total number of predictions
if "correct_predictions" not in st.session_state:
    st.session_state["correct_predictions"] = 0  # Total number of correct predictions

# Upload image
uploaded_image = st.file_uploader("Insert a picture of a monkey", type="png")

if uploaded_image is not None:
    # Convert the uploaded image to an array
    image_np = np.array(Image.open(uploaded_image))

    # Resize and preprocess the image for model prediction
    resized_img = cv2.resize(image_np, (150, 150))
    normalized_img = resized_img / 255.0
    model_input = np.expand_dims(normalized_img, axis=0)

    # Define column widths: make the middle column largest
    col1, col2, col3 = st.columns([1, 3, 1])  # Middle column is 3x larger than the side columns
    with col2:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict", use_container_width=True):
        # Predict using the model
        prediction = model.predict(model_input)
        predicted_species_index = np.argmax(prediction)
        predicted_species = species_labels[predicted_species_index]

        # Show prediction result
        st.subheader("Prediction Result:")
        st.success(f"The monkey is probably a {predicted_species}")

        # Display prediction probabilities as a bar chart
        st.bar_chart(prediction[0])

        # Add prediction to session state (last prediction for validation)
        st.session_state["last_prediction"] = {
            "image": uploaded_image,
            "species": predicted_species,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    # True/False buttons for feedback
    if st.session_state.get("last_prediction"):
        col_true, col_false = st.columns(2)

        with col_true:
            if st.button("True", use_container_width=True):
                # Mark prediction as correct and save to history
                st.session_state["correct_predictions"] += 1
                st.session_state["total_predictions"] += 1

                feedback = "True"
                st.session_state["history"].append(
                    (st.session_state["last_prediction"]["image"],
                     st.session_state["last_prediction"]["species"],
                     st.session_state["last_prediction"]["timestamp"],
                     feedback)
                )
                st.session_state["last_prediction"] = None  # Reset after feedback

        with col_false:
            if st.button("False", use_container_width=True):
                # Mark prediction as incorrect and save to history
                st.session_state["total_predictions"] += 1

                feedback = "False"
                st.session_state["history"].append(
                    (st.session_state["last_prediction"]["image"],
                     st.session_state["last_prediction"]["species"],
                     st.session_state["last_prediction"]["timestamp"],
                     feedback)
                )
                st.session_state["last_prediction"] = None  # Reset after feedback

# Sidebar to show prediction history
st.sidebar.title("Prediction History")
for i, (image, species, timestamp, feedback) in enumerate(st.session_state["history"]):
    st.sidebar.write(f"{feedback}")
    st.sidebar.image(image, caption=f"{species}", use_column_width=True)

# Calculate accuracy
if st.session_state["total_predictions"] > 0:
    accuracy = (st.session_state["correct_predictions"] / st.session_state["total_predictions"]) * 100
    st.sidebar.metric("Prediction Accuracy", f"{accuracy:.2f}%")
else:
    st.sidebar.metric("Prediction Accuracy", "N/A")