import gradio as gr
import pandas as pd
import numpy as np
import faiss
import json
import os
import csv
import shutil
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
from paddleocr import PaddleOCR
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textwrap import wrap
from reportlab.lib.utils import ImageReader

# === File paths ===
base_dir = r"C:/Users/brian/OneDrive/Escritorio/Skills/Programming/Python/Project"
faiss_path = os.path.join(base_dir, "faiss_index_PaddleOCR.index")
csv_path = os.path.join(base_dir, "extracted_text_PaddleOCR2/cleaned_text_PaddleOCR2/chunked_text_PaddleOCR.csv")
text_folder = os.path.join(base_dir, "extracted_text_PaddleOCR2/cleaned_text_PaddleOCR2")
logo_path = os.path.join(base_dir, "Logo.png")
mistral_path = r"C:\llama_cpp\llama-b5478-bin-win-cpu-x64\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
CACHE_FILE = "cache.json"
FEEDBACK_FILE = "feedback_log.csv"

# === Load components ===
index = faiss.read_index(faiss_path)
df_chunks = pd.read_csv(csv_path)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(model_path=mistral_path, n_ctx=2048)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

# === Session Logs ===
cache = {}
chat_log = []

# === Cache & feedback helpers ===

def load_cache():
    return json.load(open(CACHE_FILE, "r", encoding="utf-8")) if os.path.exists(CACHE_FILE) else {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4, ensure_ascii=False)

def save_feedback(query, answer, source, feedback):
    data = {"query": query, "answer": answer, "source": source, "feedback": feedback}
    file_exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# === Reset everything ===

def reset_all():
    """Clear cache, feedback log, temp docs, and chat state."""
    open(CACHE_FILE, "w").write("{}")
    open(FEEDBACK_FILE, "w").write("query,answer,source,feedback\n")
    shutil.rmtree("temp_docs", ignore_errors=True)
    chat_log.clear()
    return [], None, None, None, None, None

# === OCR extraction ===

def extract_text_from_image(image_path):
    ocr_result = ocr_model.ocr(image_path, cls=True)
    extracted_text = "\n".join([line[1][0] for block in ocr_result for line in block])
    return extracted_text

# === FAISS retrieval (+ optional OCR chunk) ===
