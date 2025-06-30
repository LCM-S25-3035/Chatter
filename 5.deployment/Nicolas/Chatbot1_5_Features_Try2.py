# UPDATED VERSION: Chatbot1_5_Features_with_PDF.py (adds RLHF training)

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

# === Chatbot & PDF export setup ===
# (rest of your original code here: vector store init, chat functions, Gradio UI, export_csv, export_feedback, export_pdf, etc.)

# Path to feedback log
FEEDBACK_FILE = "feedback_log.csv"

# === NEW: RLHF training ===
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead


def train_rlhf(
    feedback_path: str = FEEDBACK_FILE,
    base_model: str = "mistral-7b-instruct-v0.1",
    output_dir: str = "mistral_rlhf_finetuned",
    epochs: int = 3,
    batch_size: int = 2,
    ppo_epochs: int = 4,
    learning_rate: float = 1.41e-5,
):
    """
    Train the chatbot via PPO-based RLHF using recorded human feedback.
    Reads feedback CSV with columns: query, answer, feedback (good/bad).
    Saves a finetuned model to `output_dir`.
    """
    # 1. Load feedback and map to numeric rewards
    df = pd.read_csv(feedback_path)
    df = df.dropna(subset=["query", "answer", "feedback"])
    df["reward"] = df["feedback"].str.strip().str.lower().map(lambda x: 1.0 if x == "good" else 0.0)

    # 2. Build a HF Dataset
    records = []
    for _, row in df.iterrows():
        records.append({
            "prompt": row["query"],
            "response": row["answer"],
            "reward": row["reward"],
        })
    ds = Dataset.from_pandas(pd.DataFrame(records))

    # 3. Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model, device_map="auto")

    # 4. PPO Configuration
    ppo_config = PPOConfig(
        model_name=base_model,
        learning_rate=learning_rate,
        batch_size=batch_size,
        forward_batch_size=1,
        ppo_epochs=ppo_epochs,
        log_with=None,
    )
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    # Helper to tokenize prompts
    def build_inputs(prompts):
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        return encoded.to(model.device)

    # 5. Training loop
    for epoch in range(epochs):
        ds_shuffled = ds.shuffle(seed=epoch)
        for start_idx in range(0, len(ds_shuffled), batch_size):
            batch = ds_shuffled.select(range(start_idx, min(start_idx + batch_size, len(ds_shuffled)))).to_dict()
            prompts = [p + tokenizer.eos_token for p in batch["prompt"]]
            inputs = build_inputs(prompts)

            # Generate responses
            gen_outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                do_sample=True,
                top_k=50,
                top_p=0.9,
            )

            # Rewards tensor
            rewards = torch.tensor(batch["reward"], dtype=torch.float, device=model.device)

            # PPO optimization step
            ppo_trainer.step(
                query_tensors=inputs["input_ids"],
                response_tensors=gen_outputs,
                rewards=rewards,
            )

    # 6. Save finetuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"RLHF finetuning completed! Model saved to \"{output_dir}\"")


# === Entry point ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Run RLHF training: python Chatbot1_5_Features_with_PDF.py train
        train_rlhf()
    else:
        # Launch the chat UI
        demo.launch()
