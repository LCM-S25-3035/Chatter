# =============================================================================================
# Create a csv file with all the cleaned files Tesseract OCR (via pytesseract)
# =============================================================================================

import os
import pandas as pd

# Folder containing the cleaned .txt files
input_folder = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_text_Tesseract\cleaned_text_Tesseract"

# List to hold filename-text pairs
data = []

# Loop through all .txt files and collect content
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
            if text:  # Only include non-empty entries
                data.append({"filename": filename, "text": text})

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
output_csv = os.path.join(input_folder, "merged_cleaned_text_Tesseract.csv")
df.to_csv(output_csv, index=False)

print("✅ Merged CSV created at:", output_csv)



# =============================================================================================
# Create a csv file with all the cleaned files PaddleOCR (via paddleocr)
# =============================================================================================

import os
import pandas as pd

# Folder containing the cleaned .txt files
input_folder = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_text_PaddleOCR2\cleaned_text_PaddleOCR2"

# List to hold filename-text pairs
data = []

# Loop through all .txt files and collect content
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
            if text:  # Only include non-empty entries
                data.append({"filename": filename, "text": text})

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
output_csv = os.path.join(input_folder, "merged_cleaned_text_PaddleOCR2.csv")
df.to_csv(output_csv, index=False)

print("✅ Merged CSV created at:", output_csv)

