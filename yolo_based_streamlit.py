import streamlit as st
from ultralytics import YOLO
import pytesseract
import cv2
from pdf2image import convert_from_path
import numpy as np
import tempfile
from PIL import Image

# Load the YOLOv8 model (ensure 'best.pt' is your trained model path)
model = YOLO('best.pt')  # Replace with the path to your YOLOv8 weights

# Set up Tesseract path (update this if Tesseract is not in your PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Windows example

# Streamlit App
st.title("PDF Subpage Detection and Text Extraction")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Convert uploaded PDF to images
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    images = convert_from_path(temp_file_path, poppler_path=r"C:/Users/waqqa/Downloads/Release-24.07.0-0/poppler-24.07.0/Library/bin")  # Convert PDF to images at 300 DPI
    
    st.write("PDF converted to images successfully!")
    
    for page_num, image in enumerate(images):
        st.write(f"Processing page {page_num + 1}...")
        image_np = np.array(image)  # Convert PIL image to NumPy array
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        # Run YOLOv8 inference
        results = model(image_np)

        # Display detected subpages
        st.image(results[0].plot(), caption=f"Detected subpages on page {page_num + 1}")

        # Extract text from detected subpages using Tesseract
        subpage_count = len(results[0].boxes)  # Count the number of detected subpages

        if subpage_count > 1:  # Only process if more than 1 subpage is detected
            st.write(f"Detected {subpage_count} subpages.")
            cols = st.columns(subpage_count)  # Create columns based on the number of detected subpages

            for idx, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert bounding box tensor to list
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Check if the coordinates are within the image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)

                # Crop subpage image
                subpage_img = image_np[y1:y2, x1:x2]

                # Convert to PIL format for Tesseract if necessary
                subpage_img_pil = Image.fromarray(cv2.cvtColor(subpage_img, cv2.COLOR_BGR2RGB))

                # OCR on the cropped subpage image
                text = pytesseract.image_to_string(subpage_img_pil)
                
                # Display extracted text in respective column
                with cols[idx]:
                    st.write(f"**Subpage {idx + 1}:**")
                    st.text(text)