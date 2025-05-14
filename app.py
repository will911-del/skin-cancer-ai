import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("skin_cancer_model.h5")
class_names = ['nv', 'mel', 'bkl']

# Define class descriptions
class_descriptions = {
    'nv': 'Melanocytic nevi (benign moles)',
    'mel': 'Melanoma (malignant skin cancer)',
    'bkl': 'Benign keratosis-like lesions (non-cancerous growths)'
}

st.title("🔬 AI Skin Cancer Detector")
st.write("Upload a skin lesion image and let the AI classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    label = class_names[predicted_class]
    description = class_descriptions.get(label, "No description available.")

    st.success(f"🧠 Prediction: **{label}** ({confidence:.2%} confidence)")
    st.info(f"ℹ️ {description}")

import matplotlib.pyplot as plt

# Bar chart for confidence
st.subheader("📊 Model Confidence")
fig, ax = plt.subplots()
ax.barh(class_names, prediction[0], color="skyblue")
ax.set_xlim(0, 1)
ax.set_xlabel("Confidence")
ax.set_title("Confidence for Each Class")
for i, v in enumerate(prediction[0]):
    ax.text(v + 0.01, i, f"{v:.2%}", va='center')
st.pyplot(fig)

from fpdf import FPDF
import base64
import datetime

# Function to create PDF report
def create_pdf(pred_class, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="AI Skin Cancer Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {pred_class}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=True)

    filename = "skin_cancer_report.pdf"
    pdf.output(filename)
    return filename

# Generate report button
if st.button("📄 Download PDF Report"):
    pdf_file = create_pdf(predicted_class, prediction[0][class_names.index(predicted_class)])
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="{pdf_file}">Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)

