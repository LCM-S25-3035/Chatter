# =============================================================================================
# Cleaning Tesseract OCR (via pytesseract)
# =============================================================================================

# Cleaing Data import os
import re
import os

# Define input and output folder paths
input_folder = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_text_Tesseract"
output_folder = os.path.join(input_folder, "cleaned_text_Tesseract")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Patterns for deep cleaning
regex_patterns = [
    r'\[ERROR.*?\]',  # Remove error messages
    r'\[NO TEXT DETECTED\]',  # Remove boilerplate
    r'https?://\S+',  # Remove URLs
    r'page\s*\d+\s*(of\s*\d+)?',  # Remove page numbers like "Page 1 of 10"
    r'\b(source|references|footer)\b.*',  # Remove lines starting with certain headers
]

# Compile regex patterns
compiled_patterns = [re.compile(p, flags=re.IGNORECASE) for p in regex_patterns]

# Function to clean text
def clean_text(text):
    # Normalize whitespace
    text = text.replace('\xa0', ' ')  # Replace non-breaking space
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces/newlines
    # Apply all regex filters
    for pattern in compiled_patterns:
        text = pattern.sub('', text)
    # Remove non-ASCII symbols like �
    text = text.encode('ascii', errors='ignore').decode()
    # Strip outer whitespace and convert to lowercase
    return text.strip().lower()

# Process each text file
cleaned_count = 0
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile:
                raw_text = infile.read()
            cleaned = clean_text(raw_text)
            if cleaned:
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(cleaned)
                cleaned_count += 1
        except Exception as e:
            print(f"Error cleaning file {filename}: {e}")




# =============================================================================================
# Cleaning PaddleOCR (via paddleocr)
# =============================================================================================

import os
import re

# === Folder paths ===
input_folder = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_text_PaddleOCR2"
output_folder = os.path.join(input_folder, "cleaned_text_PaddleOCR2")
os.makedirs(output_folder, exist_ok=True)

# === Patterns to remove ===
regex_patterns = [
    r'\[ERROR.*?\]',                       # Remove error lines
    r'\[NO TEXT DETECTED\]',              # Placeholder lines
    r'https?://\S+',                      # URLs
    r'\b(source|references|footer)\b.*',  # Footer junk
    r'\(Confidence:\s*\d+(\.\d+)?\)',      # Confidence scores like (Confidence: 0.96)
    r'\[\s*Source image:.*?\]',           # Optional: remove image origin tag
]
compiled_patterns = [re.compile(p, flags=re.IGNORECASE) for p in regex_patterns]

# === Clean text ===
def clean_text(text):
    text = text.replace('\xa0', ' ')  # Replace non-breaking spaces
    for pattern in compiled_patterns:
        text = pattern.sub('', text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'\(\s*\)', '', text)  # Remove leftover empty parentheses
    text = text.encode('ascii', errors='ignore').decode()  # Remove non-ASCII chars
    return text.strip()

# === Apply cleaning ===
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()

            cleaned_text = clean_text(raw_text)

            # Only save non-empty content
            if cleaned_text:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                print(f"✅ Cleaned: {filename}")
            else:
                print(f"⚠️ Empty after cleaning: {filename}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

