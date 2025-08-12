# =============================================================================================
# Chunking the Text Tesseract OCR (via pytesseract)
# =============================================================================================

# Import necessary libraries
import os                      # For file path management
import pandas as pd            # For handling CSVs and tables
from nltk.tokenize import word_tokenize  # To break text into words (tokens)
from tqdm import tqdm          # To show a progress bar while processing

# STEP 1: Load the merged cleaned CSV from Tesseract
# Path to your cleaned & merged Tesseract text file
csv_path = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_text_Tesseract\cleaned_text_Tesseract\merged_cleaned_text_Tesseract.csv"

# Read the CSV file into a DataFrame (like a table in memory)
df = pd.read_csv(csv_path)


# STEP 2: Define chunking parameters
chunk_size = 500      # Number of tokens per chunk
overlap = 50          # Number of overlapping tokens between chunks

# Prepare an empty list to store each chunk
chunks = []
chunk_id = 0          # Start counter for chunk IDs

# STEP 3: Loop through each document and chunk it
# Iterate over each row in the DataFrame
for idx, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['filename']           # Name of the original text file
    text = str(row['text'])              # The actual text content (ensure it's a string)

    tokens = word_tokenize(text)         # Split the text into words (tokens)

    # Create overlapping chunks
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]          # Select 500 tokens with overlap
        chunk_text = ' '.join(chunk_tokens)              # Combine tokens into a string

        if chunk_text.strip():                           # Make sure chunk isn't empty
            chunks.append({
                'chunk_id': f"{filename}_{chunk_id}",    # Create a unique ID for this chunk
                'filename': filename,                    # Track original file name
                'chunk_text': chunk_text                 # Store the actual chunk text
            })
            chunk_id += 1                                # Increment chunk ID

# STEP 4: Save all chunks into a new CSV
# Where to save the chunked text
output_path = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_text_Tesseract\cleaned_text_Tesseract\chunked_text_Tesseract.csv"

# Convert list of chunks into a DataFrame
chunk_df = pd.DataFrame(chunks)

# Save to CSV
chunk_df.to_csv(output_path, index=False)



# =============================================================================================
# Chunking the Text PaddleOCR (via paddleocr)
# =============================================================================================
import os  # Used for handling file paths (not strictly needed here but a good habit)
import pandas as pd  # Used to work with CSV files and data tables
from nltk.tokenize import word_tokenize  # Used to split text into individual words (tokens)
from tqdm import tqdm  # Used to show a progress bar during long operations

# STEP 1: Load the merged cleaned CSV from Tesseract
# Path to the cleaned and merged PaddleOCR CSV file
csv_path = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_text_PaddleOCR2\cleaned_text_PaddleOCR2\merged_cleaned_text_PaddleOCR2.csv"
# Load the CSV into a DataFrame
df = pd.read_csv(csv_path)


# STEP 2: Define chunking parameters
# Set chunking parameters
chunk_size = 500      # Each chunk will contain 500 tokens (words)
overlap = 50          # 50 tokens from the end of one chunk will appear at the start of the next (to preserve context)

# Prepare output structure
chunks = []          # This will store the final list of chunks
chunk_id = 0         # This is a counter to give each chunk a unique ID


# STEP 3: Loop through each document and chunk it
# Go through each row (i.e., each document) in the CSV
for idx, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['filename']       # Get the name of the file the text came from
    text = str(row['text'])          # Get the text from that row (and make sure it's a string)
    
    tokens = word_tokenize(text)     # Split the text into individual words (tokens)

    # Split the tokens into chunks with overlap
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]         # Take 500 tokens, starting every (500 - 50 = 450)
        chunk_text = ' '.join(chunk_tokens)             # Join tokens back into a string
        if chunk_text.strip():                          # Only keep the chunk if it contains non-empty text
            chunks.append({
                'chunk_id': f"{filename}_{chunk_id}",   # Create a unique ID for the chunk
                'filename': filename,                   # Store the original filename
                'chunk_text': chunk_text                # Store the actual chunk of text
            })
            chunk_id += 1
        
# STEP 4: Save all chunks into a new CSV
# Define the path where the final chunked CSV will be saved
output_path = r"C:\Users\brian\OneDrive\Escritorio\Skills\Programming\Python\Project\extracted_text_PaddleOCR2\cleaned_text_PaddleOCR2\chunked_text_PaddleOCR.csv"
# Convert the chunks list into a DataFrame (table)
chunk_df = pd.DataFrame(chunks)
# Save the table into a CSV file without the index column
chunk_df.to_csv(output_path, index=False)

