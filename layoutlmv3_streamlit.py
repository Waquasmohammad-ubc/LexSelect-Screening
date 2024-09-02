import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from pdf2image import convert_from_bytes
import io

# Initialize Streamlit app
st.title("PDF Entity Extraction with LayoutLMv3")

# Load the fine-tuned model and processor
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained("Waquas/layoutlmv3-lexselect4")

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def pdf_to_images(pdf_file):
    images = convert_from_bytes(pdf_file.read(), poppler_path=r"C:/Users/waqqa/Downloads/Release-24.07.0-0/poppler-24.07.0/Library/bin")
    return images

def perform_ocr_and_get_boxes(image):
    ocr_results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    boxes = []
    width, height = image.size

    for i in range(len(ocr_results['text'])):
        word = ocr_results['text'][i].strip()
        if word:
            words.append(word)
            x, y, w, h = (ocr_results['left'][i], ocr_results['top'][i], ocr_results['width'][i], ocr_results['height'][i])
            boxes.append(normalize_bbox([x, y, x + w, y + h], width, height))

    return words, boxes

def prepare_encoding(image, words, boxes):
    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        stride=128,
        padding="max_length",
        max_length=512,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )
    
    encoding.pop('offset_mapping', None)
    
    pixel_values = [torch.tensor(p) if not isinstance(p, torch.Tensor) else p for p in encoding['pixel_values']]
    encoding['pixel_values'] = torch.stack(pixel_values)
    
    overflowed_encodings = encoding.pop("overflow_to_sample_mapping")
    for k, v in encoding.items():
        if isinstance(v, list):
            if all(isinstance(item, torch.Tensor) for item in v):
                encoding[k] = torch.stack(v)
            else:
                try:
                    v_tensor = [torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in v]
                    encoding[k] = torch.stack(v_tensor)
                except Exception as e:
                    print(f"Could not convert and stack list for key '{k}': {e}")
        elif isinstance(v, torch.Tensor):
            pass

    return encoding

def run_inference(encoding):
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'], 
            attention_mask=encoding['attention_mask'],
            bbox=encoding['bbox'],
            pixel_values=encoding['pixel_values']
        )

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    return predictions

def post_process_and_visualize(image, predictions, encoding, words, boxes):
    labels = encoding['labels'].squeeze().tolist() if 'labels' in encoding else None
    if isinstance(predictions[0], list):
        new_pred = [pred for batch_preds in predictions for pred in batch_preds]
    else:
        new_pred = predictions

    new_labels = [label for batch_labels in labels for label in batch_labels] if labels and isinstance(labels[0], list) else labels

    def unnormalize_box(bbox, width, height):
        return [
            width * (bbox[0] / 1000),
            height * (bbox[1] / 1000),
            width * (bbox[2] / 1000),
            height * (bbox[3] / 1000),
        ]

    token_boxes = encoding['bbox'].squeeze().tolist()
    width, height = image.size

    if isinstance(token_boxes[0], list):
        new_token_boxes = [box for batch_boxes in token_boxes for box in batch_boxes]
    else:
        new_token_boxes = token_boxes

    true_predictions = [model.config.id2label[pred] for pred in new_pred if pred != -100]
    true_labels = [model.config.id2label[label] for label in new_labels if label != -100] if new_labels else []

    true_boxes = [
        unnormalize_box(box, width, height) for box in new_token_boxes 
        if isinstance(box, (list, tuple)) and len(box) == 4
    ]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    label2color = {'subpage_1': 'blue', 'subpage_2': 'green', 'subpage_3': 'orange', 'subpage_4': 'violet', 'other': 'grey'}

    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = prediction.lower()
        draw.rectangle(box, outline=label2color.get(predicted_label, 'grey'))
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color.get(predicted_label, 'grey'), font=font)

    return image, true_predictions

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    images = pdf_to_images(uploaded_file)
    all_predictions = []

    for idx, image in enumerate(images):
        words, boxes = perform_ocr_and_get_boxes(image)
        encoding = prepare_encoding(image, words, boxes)
        predictions = run_inference(encoding)
        annotated_image, page_predictions = post_process_and_visualize(image, predictions, encoding, words, boxes)

        st.image(annotated_image, caption=f"Page {idx + 1}", use_column_width=True)
        all_predictions.extend(page_predictions)

    st.subheader("Extracted Entities")
    st.write(all_predictions)
