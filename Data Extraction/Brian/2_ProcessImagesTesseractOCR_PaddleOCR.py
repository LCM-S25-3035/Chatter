'''
=============================================================================================
Create the output folder and process images for Tesseract OCR (via pytesseract)

Tesseract is an open-source OCR engine developed by HP and now maintained by Google.
It uses traditional image processing techniques (not deep learning).
It is language-flexible and lightweight, but not very good with:
    rotated text
    tabular data
    low-quality scans
    dense or complex layouts
=============================================================================================
'''

import os
from PIL import Image
import pytesseract

# Path to Tesseract executable (adjust this if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Paths
image_folder = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_images"
output_folder = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_text_Tesseract"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Loop through and process
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    output_txt = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.txt')

    try:
        # Open and OCR
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)

        # Save even if it's empty
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

        print(f"✅ Saved: {output_txt}")
    
    except Exception as e:
        # Log error but continue
        print(f"❌ Error processing {image_file}: {e}")
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(f"[Error reading this file: {e}]")



'''
=============================================================================================
Create the output folder and process images for PaddleOCR (via paddleocr)

PaddleOCR is a deep learning–based OCR system from Baidu.
It uses modern computer vision models (CNNs, transformers).
Built on PaddlePaddle, which is similar to PyTorch or TensorFlow.    
=============================================================================================
'''
import os
from paddleocr import PaddleOCR
from PIL import Image

# === Step 1: Initialize PaddleOCR ===
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# === Step 2: Define paths ===
image_folder = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_images"
output_folder = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_text_PaddleOCR2"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# === Step 3: Loop over each image and apply OCR ===
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)

        try:
            result = ocr.ocr(image_path, cls=True)

            # Extract text from the OCR result
            extracted_text = f"[Source image: {filename}]\n\n"
            has_text = False

            if result and result[0]:
                for line in result[0]:
                    box, (text, confidence) = line
                    if text.strip():
                        has_text = True
                        extracted_text += f"{text} (Confidence: {confidence:.2f})\n"

            if not has_text:
                extracted_text += "[NO TEXT DETECTED]"

        except Exception as e:
            extracted_text = f"[ERROR PROCESSING IMAGE: {filename}]\n{str(e)}"

        # === Paso 5: Save results in txt file ===
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

print("✅ OCR extraction and file generation completed for all images.")