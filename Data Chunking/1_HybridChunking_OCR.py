# Hybrid Chunking of OCR Text for RAG

## Chunking OCR Text hybrid

The goal of this script is to split long OCR-extracted texts into smaller, manageable chunks of 500 tokens with 50-token overlap. This makes the text suitable for embedding and retrieval in a Retrieval-Augmented Generation (RAG) pipeline.


### Part 1 – Chunking Text from Tesseract OCR

**Step 1: Import Required Libraries**
"""

# Import necessary libraries
import os                      # For file path management
import pandas as pd            # For handling CSVs and tables
from nltk.tokenize import word_tokenize  # To break text into words (tokens)
from tqdm import tqdm          # To show a progress bar while processing

"""**Step 2: Load Cleaned Tesseract Text**"""

# Path to your cleaned & merged Tesseract text file
csv_path = r"C:\Users\hp\Downloads\Project folder of Chatter\extracted_text_Tesseract\cleaned_text_Tesseract\merged_cleaned_text_Tesseract.csv"

# Read the CSV file into a DataFrame (like a table in memory)
df = pd.read_csv(csv_path)

import nltk
nltk.download('punkt')                         # only needs to happen once per environment
from nltk.tokenize import sent_tokenize, word_tokenize

# Parameters
chunk_size = 500
overlap = 50

# Hybrid chunking using nltk sentence tokenization first
all_chunks = []
chunk_id = 0

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['text']
    sentences = sent_tokenize(text)  # Step 1: sentence-level split

    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        if current_token_count + len(tokens) <= chunk_size:
            current_chunk.extend(tokens)
            current_token_count += len(tokens)
        else:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            all_chunks.append({'chunk_id': chunk_id, 'text': chunk_text})
            chunk_id += 1

            # Start new chunk with overlap
            overlap_tokens = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_tokens + tokens
            current_token_count = len(current_chunk)

    # Add the final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        all_chunks.append({'chunk_id': chunk_id, 'text': chunk_text})
        chunk_id += 1

# Create DataFrame of chunks
chunks_df = pd.DataFrame(all_chunks)

"""**Step 3: Define Chunking Parameters**"""

# Each chunk will have 500 tokens.
chunk_size = 500
# Number of overlapping tokens between chunks. Each chunk overlaps 50 tokens with the previous one to preserve context.
overlap = 50

# Prepare an empty list to store each chunk
chunks = []
# Start counter for chunk IDs
chunk_id = 0

"""**Step 4: Chunk Each Text Document**"""

# Iterate over each row in the DataFrame
for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Name of the original text file
    filename = row['filename']
    # The actual text content (ensure it's a string)
    text = str(row['text'])
    # Split the text into words (tokens)
    tokens = word_tokenize(text)

    # Create overlapping chunks
    for i in range(0, len(tokens), chunk_size - overlap):
        # Select 500 tokens with overlap
        chunk_tokens = tokens[i:i + chunk_size]
        # Combine tokens into a string
        chunk_text = ' '.join(chunk_tokens)

        # Make sure chunk isn't empty
        if chunk_text.strip():
            chunks.append({
                # Create a unique ID for this chunk
                'chunk_id': f"{filename}_{chunk_id}",
                # Track original file name
                'filename': filename,
                # Store the actual chunk text
                'chunk_text': chunk_text
            })
            # Increment chunk ID
            chunk_id += 1

"""**Step 5: Save Chunks to CSV**"""

#Where to save the chunked text
output_path = r"C:\Users\hp\Downloads\Project folder of Chatter\extracted_text_Tesseract\cleaned_text_Tesseract\chunked_text_Tesseract_hybrid.csv"

# Convert list of chunks into a DataFrame
chunk_df = pd.DataFrame(chunks)

# Save to CSV
chunk_df.to_csv(output_path, index=False)

chunk_df.head(n=4)

"""### Part 2 – Chunking Text from PaddleOCR

The structure is identical to the Tesseract chunking, with only the CSV input/output path changed.


"""

# Used for handling file paths (not strictly needed here but a good habit)
import os
# Used to work with CSV files and data tables
import pandas as pd
# Used to split text into individual words (tokens)
from nltk.tokenize import word_tokenize
# Used to show a progress bar during long operations
from tqdm import tqdm

"""**Step 2: Load Cleaned Tesseract Text**"""

# Path to the cleaned and merged PaddleOCR CSV file
csv_path = r"C:\Users\hp\Downloads\Project folder of Chatter\extracted_text_PaddleOCR2\cleaned_text_PaddleOCR2\merged_cleaned_text_PaddleOCR2.csv"
# Load the CSV into a DataFrame
df = pd.read_csv(csv_path)

# Parameters
chunk_size = 500
overlap = 50

# Hybrid chunking using nltk sentence tokenization first
all_chunks = []
chunk_id = 0

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['text']
    sentences = sent_tokenize(text)  # Step 1: sentence-level split

    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        if current_token_count + len(tokens) <= chunk_size:
            current_chunk.extend(tokens)
            current_token_count += len(tokens)
        else:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            all_chunks.append({'chunk_id': chunk_id, 'text': chunk_text})
            chunk_id += 1

            # Start new chunk with overlap
            overlap_tokens = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_tokens + tokens
            current_token_count = len(current_chunk)

    # Add the final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        all_chunks.append({'chunk_id': chunk_id, 'text': chunk_text})
        chunk_id += 1

# Create DataFrame of chunks
chunks_df = pd.DataFrame(all_chunks)

"""**Step 3: Define Chunking Parameters**"""

# Each chunk will contain 500 tokens (words)
chunk_size = 500
# 50 tokens from the end of one chunk will appear at the start of the next (to preserve context)
overlap = 50

# Prepare output structure
# This will store the final list of chunks
chunks = []
# This is a counter to give each chunk a unique ID
chunk_id = 0

"""**Step 4: Chunk Each Text Document**"""

# Iterate over each row in the DataFrame
for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Name of the original text file
    filename = row['filename']
    # The actual text content (ensure it's a string)
    text = str(row['text'])
    # Split the text into words (tokens)
    tokens = word_tokenize(text)

    # Create overlapping chunks
    for i in range(0, len(tokens), chunk_size - overlap):
        # Select 500 tokens with overlap
        chunk_tokens = tokens[i:i + chunk_size]
        # Combine tokens into a string
        chunk_text = ' '.join(chunk_tokens)

        # Make sure chunk isn't empty
        if chunk_text.strip():
            chunks.append({
                # Create a unique ID for this chunk
                'chunk_id': f"{filename}_{chunk_id}",
                # Track original file name
                'filename': filename,
                # Store the actual chunk text
                'chunk_text': chunk_text
            })
            # Increment chunk ID
            chunk_id += 1

"""**Step 5: Save Chunks to CSV**"""

#Where to save the chunked text
#output_path =r"C:\Users\hp\Downloads\Project folder of Chatter\extracted_text_Tesseract\cleaned_text_Tesseract\chunked_text_Tesseract_hybrid.csv"



# Define the path where the final chunked CSV will be saved
output_path = r"C:\Users\hp\Downloads\Project folder of Chatter\extracted_text_PaddleOCR2\cleaned_text_PaddleOCR2\chunked_text_PaddleOCR_hybrid.csv"

# Convert list of chunks into a DataFrame
chunk_df = pd.DataFrame(chunks)

# Save to CSV
chunk_df.to_csv(output_path, index=False)

chunk_df.head(n=4)

"""Both versions create a CSV file with the following columns:

+ chunk_id: Unique ID (e.g., filename_0, filename_1, ...)
+ filename: Name of the source document
+ chunk_text: The actual chunk of 500 tokens (with overlap)
"""
