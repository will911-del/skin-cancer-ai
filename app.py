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
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    help="Accepted formats: JPG, JPEG, PNG. Max 200MB."
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Image processing
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    label = class_names[predicted_class]
    description = class_descriptions.get(label, "No description available.")

    # Show result
    st.success(f"🧠 Prediction: **{label}** ({confidence:.2%} confidence)")
    st.info(f"ℹ️ {description}")

    # Confidence bar chart
    st.subheader("📊 Model Confidence")
    fig, ax = plt.subplots()
    ax.barh(class_names, prediction[0], color="skyblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Confidence for Each Class")
    for i, v in enumerate(prediction[0]):
        ax.text(v + 0.01, i, f"{v:.2%}", va='center')
    st.pyplot(fig)

    # Generate PDF report
    from fpdf import FPDF
    import base64
    import datetime

    def create_pdf(pred_class, confidence):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Skin Cancer AI Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Prediction: {pred_class}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=True)
        pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        return pdf.output(dest="S").encode("latin1")

    pdf_data = create_pdf(label, confidence)
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="skin_cancer_report.pdf">📄 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)

else:
    st.warning("👆 Please upload a skin lesion image to begin.")

# Streamlit download button
pdf_data = create_pdf(label, confidence)
b64 = base64.b64encode(pdf_data).decode()
href = f'<a href="data:application/octet-stream;base64,{b64}" download="skin_cancer_report.pdf">📄 Download Report</a>'
st.markdown(href, unsafe_allow_html=True)

# Function to create PDF report
def create_pdf(pred_class, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Skin Cancer AI Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction: {pred_class}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=True)
    pdf.output("report.pdf")
    return "report.pdf"

# Create and download the report
pdf_file = create_pdf(label, confidence)
with open(pdf_file, "rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="report.pdf">📄 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)
 
    # Option to download PDF
    if st.button("📄 Download PDF Report"):
        pdf = create_pdf(label, confidence, description)
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        b64 = base64.b64encode(pdf_output.getvalue()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="report.pdf">Click here to download your PDF report</a>'
        st.markdown(href, unsafe_allow_html=True)

