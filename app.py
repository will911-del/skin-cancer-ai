import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import base64
import datetime

# Load the model
model = tf.keras.models.load_model("skin_cancer_model.h5")

# Class labels
class_names = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Dermatofibroma",
    "Melanocytic Nevi",
    "Vascular Lesions",
    "Melanoma"
]

# Optional class descriptions
class_descriptions = {
    "Actinic Keratoses": "Precancerous skin condition caused by sun damage.",
    "Basal Cell Carcinoma": "Common type of skin cancer, usually not life-threatening.",
    "Benign Keratosis-like Lesions": "Non-cancerous skin growths.",
    "Dermatofibroma": "Benign fibrous skin nodule.",
    "Melanocytic Nevi": "Common moles.",
    "Vascular Lesions": "Blood vessel abnormalities.",
    "Melanoma": "Dangerous form of skin cancer that can spread quickly."
}

# Title
st.title("🔬 AI Skin Cancer Detector")
st.markdown("Upload a skin lesion image and let the AI classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))  # adjust size as per your model input
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    label = class_names[predicted_class]
    description = class_descriptions.get(label, "No description available.")

    # Show results
    st.success(f"🧠 Prediction: **{label}** ({confidence:.2%} confidence)")
    st.info(f"ℹ️ {description}")

    # Confidence chart
    st.subheader("📊 Model Confidence")
    fig, ax = plt.subplots()
    ax.barh(class_names, prediction[0], color="skyblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Confidence for Each Class")
    for i, v in enumerate(prediction[0]):
        ax.text(v + 0.01, i, f"{v:.2%}", va='center')
    st.pyplot(fig)

    # PDF Report Generation
    def create_pdf(pred_class, confidence, description):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Skin Cancer Detection Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Prediction: {pred_class}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=True)
        pdf.multi_cell(0, 10, txt=f"Description: {description}")
        pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        return pdf

    # Option to download PDF
    if st.button("📄 Download PDF Report"):
        pdf = create_pdf(label, confidence, description)
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        b64 = base64.b64encode(pdf_output.getvalue()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="report.pdf">Click here to download your PDF report</a>'
        st.markdown(href, unsafe_allow_html=True)

