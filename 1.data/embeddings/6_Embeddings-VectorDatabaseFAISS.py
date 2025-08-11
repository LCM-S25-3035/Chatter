'''
========================================================================================
Generate Embeddings and Store in Vector Database (FAISS)
You're converting your chunks of text into numerical vectors using a model (like sentence-transformers) so they can be stored and searched based on semantic meaning.

Workflow Overview:
1. Load the chunked CSV
    You already have:
        chunked_text_Tesseract.csv
        chunked_text_PaddleOCR.csv
2. Use a pre-trained embedding model
    We'll use sentence-transformers (e.g., 'all-MiniLM-L6-v2').
3. Create embeddings for each chunk
4. Store embeddings and metadata (chunk_id, filename, text) in a FAISS index.
5. Save FAISS index and metadata for later retrieval
========================================================================================
'''
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np


# === STEP 1: Load the chunked PaddleOCR CSV file ===
csv_path = r"C:\Users\manjo\Downloads\Project\extracted_text_PaddleOCR2\cleaned_text_PaddleOCR2_hybrid\chunked_text_PaddleOCR_hybrid.csv"
df = pd.read_csv(csv_path)

# === STEP 2: Load the pre-trained embedding model ===
# This model converts text into vector form
model = SentenceTransformer("all-MiniLM-L6-v2")

# === STEP 3: Generate embeddings for all chunks ===
texts = df["chunk_text"].tolist()  # Get list of text chunks
embeddings = model.encode(texts, show_progress_bar=True)  # Get sentence embeddings



embeddings = np.array(embeddings).astype("float32")  

# === STEP 4: Build the FAISS index ===
dimension = embeddings.shape[1]  # Get the size of each embedding vector
index = faiss.IndexFlatL2(dimension)  # Create a FAISS index using L2 (Euclidean) distance
index.add(embeddings)  # Add all embeddings to the index

# === STEP 5: Save the FAISS index and metadata ===
# Save the index file
faiss_path = r"C:\Users\manjo\Downloads\Project\faiss_index_PaddleOCR.index"
faiss.write_index(index, faiss_path)

# Save the metadata (chunk_id, filename, chunk_text) for later use
metadata_path = r"C:\Users\manjo\Downloads\Project\metadata_PaddleOCR.csv"
df.to_csv(metadata_path, index=False)
