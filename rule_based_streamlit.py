import streamlit as st
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import os
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
# Function to convert PDF to images
def convert_pdf_to_images(pdf_file):
    return convert_from_path(pdf_file, poppler_path=r"C:/Users/waqqa/Downloads/Release-24.07.0-0/poppler-24.07.0/Library/bin")

# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

# Function to detect lines in an image
def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=1000, maxLineGap=10)
    return lines

# Function to merge close lines
def merge_close_lines(lines, distance_threshold=10):
    if lines is None or len(lines) == 0:
        return np.array([])

    def line_distance(line1, line2):
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]
        return min(
            np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2),
            np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2)
        )

    def merge_lines(group):
        x_coords = [line[0][0] for line in group] + [line[0][2] for line in group]
        y_coords = [line[0][1] for line in group] + [line[0][3] for line in group]
        return np.array([[min(x_coords), min(y_coords), max(x_coords), max(y_coords)]])

    merged_lines = []
    visited = [False] * len(lines)

    for i in range(len(lines)):
        if visited[i]:
            continue
        group = [lines[i]]
        visited[i] = True
        for j in range(i + 1, len(lines)):
            if not visited[j] and line_distance(lines[i], lines[j]) < distance_threshold:
                group.append(lines[j])
                visited[j] = True
        merged_lines.append(merge_lines(group))

    return np.array(merged_lines)

# Function to draw lines on an image
def draw_lines(image, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# Function to extract subpage coordinates
def extract_subpage_coordinates(image, vertical_lines, horizontal_lines):
    horizontal_lines = sorted(set((y1, x1) for x1, y1, _, _ in horizontal_lines))
    if len(vertical_lines) < 1 or len(horizontal_lines) < 2:
        raise ValueError("Not enough lines to define subpages")

    left = min(line[0] for line in horizontal_lines)
    right = max(line[0] for line in horizontal_lines)
    top = min(line[1] for line in vertical_lines)
    bottom = max(line[-1] for line in vertical_lines)

    subpage_coordinates = [
        (left, top, vertical_lines[0][0], horizontal_lines[1][0]),
        (vertical_lines[0][0], top, right, horizontal_lines[1][0]),
        (left, horizontal_lines[1][0], vertical_lines[0][0], bottom),
        (vertical_lines[0][0], horizontal_lines[1][0], right, bottom)
    ]

    subpages = []
    for x1, y1, x2, y2 in subpage_coordinates:
        subpage = image[y1:y2, x1:x2]
        subpages.append(subpage)

    return subpages

def find_quadruple_layout(lines, image_shape):
    vertical_lines = []
    horizontal_lines = []

    # Separate lines into vertical and horizontal based on their orientation
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 10:  # Vertical line
            vertical_lines.append((x1, y1, x2, y2))
        elif abs(y1 - y2) < 10:  # Horizontal line
            horizontal_lines.append((x1, y1, x2, y2))
    print(vertical_lines,horizontal_lines)
    # Check if we have the correct number of lines for a quadruple layout
    if len(vertical_lines) == 1 and len(horizontal_lines) == 3:
        return vertical_lines, horizontal_lines
    else:
        return None, None

# Function to extract text from subpages
def extract_text_from_subpages(subpages):
    texts = []
    for subpage in subpages:
        text = pytesseract.image_to_string(subpage, config='--psm 6')
        texts.append(text)
    return texts

# # Streamlit UI
# st.title("PDF Subpage Extractor")
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
# if uploaded_file is not None:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#         temp_pdf.write(uploaded_file.read())
#         temp_pdf_path = temp_pdf.name

#     images = convert_pdf_to_images(temp_pdf_path)

#     for i, image in enumerate(images):
#         st.image(image, caption=f"Page {i+1}", use_column_width=True)

#         image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         edges = preprocess_image(image_cv)
#         lines = detect_lines(edges)
#         lines = merge_close_lines(lines)

#         if lines is not None:
#             image_with_lines = draw_lines(image_cv.copy(), lines)

#             vertical_lines, horizontal_lines = find_quadruple_layout(lines, image_cv.shape)

#             if vertical_lines and horizontal_lines:
#                 subpages = extract_subpage_coordinates(image_with_lines, vertical_lines, horizontal_lines)
#                 texts = extract_text_from_subpages(subpages)

#                 st.write(f"Text from Page {i+1}:")

#                 for idx, (subpage, text) in enumerate(zip(subpages, texts)):
#                     st.write(f"Subpage {idx + 1}:")
#                     st.image(subpage, caption=f"Subpage {idx + 1}", use_column_width=True)
#                     st.text_area(f"Text from Subpage {idx + 1}", text, height=200)
#             else:
#                 st.write(f"No quadruple layout detected on page {i+1}.")
#         else:
#             st.write(f"No lines detected on page {i+1}.")

#     os.remove(temp_pdf_path)



# Streamlit UI
st.title("PDF Subpage Extractor")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    images = convert_pdf_to_images(temp_pdf_path)

    for i, image in enumerate(images):
        st.image(image, caption=f"Page {i+1}", use_column_width=True)

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        edges = preprocess_image(image_cv)
        lines = detect_lines(edges)
        lines = merge_close_lines(lines)

        if lines is not None:
            image_with_lines = draw_lines(image_cv.copy(), lines)

            vertical_lines, horizontal_lines = find_quadruple_layout(lines, image_cv.shape)

            if vertical_lines and horizontal_lines:
                subpages = extract_subpage_coordinates(image_with_lines, vertical_lines, horizontal_lines)
                texts = extract_text_from_subpages(subpages)

                st.write(f"Text from Page {i+1}:")

                # Horizontal scroll using columns
                cols = st.columns(len(subpages))
                for idx, (subpage, text) in enumerate(zip(subpages, texts)):
                    with cols[idx]:
                        st.image(subpage, caption=f"Subpage {idx + 1}", use_column_width=True)
                        st.text_area(f"Text from Subpage {idx + 1}", text, height=200)
            else:
                st.write(f"No quadruple layout detected on page {i+1}.")
        else:
            st.write(f"No lines detected on page {i+1}.")

    os.remove(temp_pdf_path)
