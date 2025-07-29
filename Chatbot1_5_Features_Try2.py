# UPDATED VERSION: Chatbot1_5_Features_with_RLHF.py (adds RLHF capabilities)

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
import pickle
from collections import defaultdict
import random

# === File paths ===
base_dir = r"C:/Users/brian/OneDrive/Escritorio/Skills/Programming/Python/Project"
faiss_path = os.path.join(base_dir, "faiss_index_PaddleOCR.index")
csv_path = os.path.join(base_dir, "extracted_text_PaddleOCR2/cleaned_text_PaddleOCR2/chunked_text_PaddleOCR.csv")
text_folder = os.path.join(base_dir, "extracted_text_PaddleOCR2/cleaned_text_PaddleOCR2")
logo_path = os.path.join(base_dir, "Logo.png")
mistral_path = r"C:\llama_cpp\llama-b5478-bin-win-cpu-x64\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
CACHE_FILE = "cache.json"
FEEDBACK_FILE = "feedback_log.csv"
RLHF_MODEL_FILE = "rlhf_model.pkl"
REWARD_HISTORY_FILE = "reward_history.json"

# === Load components ===
index = faiss.read_index(faiss_path)
df_chunks = pd.read_csv(csv_path)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(model_path=mistral_path, n_ctx=2048)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

# === Session Logs ===
cache = {}
chat_log = []

# === RLHF Components ===
class RLHFModel:
    def __init__(self):
        self.chunk_rewards = defaultdict(float)  # Rewards per chunk_id
        self.query_pattern_rewards = defaultdict(float)  # Rewards per query pattern
        self.source_preferences = defaultdict(float)  # Source file preferences
        self.response_features = defaultdict(lambda: defaultdict(float))  # Features for responses
        self.learning_rate = 0.1
        self.exploration_rate = 0.1  # For exploration vs exploitation
        self.temperature = 1.0  # For softmax selection
        
    def update_rewards(self, query, chunks_used, answer, source, feedback):
        """Update rewards based on human feedback"""
        reward = 1.0 if feedback == "good" else -1.0
        
        # Update chunk rewards
        for _, chunk in chunks_used.iterrows():
            chunk_id = chunk.get("chunk_id", "unknown")
            self.chunk_rewards[chunk_id] += reward * self.learning_rate
            
        # Update query pattern rewards (simplified pattern matching)
        query_words = set(query.lower().split())
        for word in query_words:
            self.query_pattern_rewards[word] += reward * self.learning_rate * 0.5
            
        # Update source preferences
        self.source_preferences[source] += reward * self.learning_rate
        
        # Update response features
        answer_length = len(answer.split())
        self.response_features["length"][answer_length // 50] += reward * self.learning_rate * 0.3
        
    def get_chunk_scores(self, chunks_df, query):
        """Calculate adjusted scores for chunks based on learned rewards"""
        scores = []
        query_words = set(query.lower().split())
        
        for _, chunk in chunks_df.iterrows():
            base_score = 1.0
            
            # Apply chunk-specific reward
            chunk_id = chunk.get("chunk_id", "unknown")
            if chunk_id in self.chunk_rewards:
                base_score += self.chunk_rewards[chunk_id]
                
            # Apply source preference
            filename = chunk.get("filename", "")
            if filename in self.source_preferences:
                base_score += self.source_preferences[filename] * 0.5
                
            # Apply query pattern matching bonus
            chunk_words = set(chunk.get("chunk_text", "").lower().split())
            overlap = len(query_words & chunk_words)
            for word in query_words:
                if word in self.query_pattern_rewards:
                    base_score += self.query_pattern_rewards[word] * overlap * 0.1
                    
            scores.append(base_score)
            
        return np.array(scores)
    
    def select_chunks_with_exploration(self, chunks_df, scores, k=2):
        """Select chunks using softmax with exploration"""
        if random.random() < self.exploration_rate:
            # Exploration: select randomly
            indices = np.random.choice(len(chunks_df), min(k, len(chunks_df)), replace=False)
        else:
            # Exploitation: use learned scores
            # Apply softmax to convert scores to probabilities
            exp_scores = np.exp(scores / self.temperature)
            probabilities = exp_scores / exp_scores.sum()
            
            # Sample based on probabilities
            indices = np.random.choice(
                len(chunks_df), 
                min(k, len(chunks_df)), 
                replace=False, 
                p=probabilities
            )
            
        return chunks_df.iloc[indices]
    
    def save(self, filepath):
        """Save the RLHF model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'chunk_rewards': dict(self.chunk_rewards),
                'query_pattern_rewards': dict(self.query_pattern_rewards),
                'source_preferences': dict(self.source_preferences),
                'response_features': dict(self.response_features),
                'learning_rate': self.learning_rate,
                'exploration_rate': self.exploration_rate,
                'temperature': self.temperature
            }, f)
    
    def load(self, filepath):
        """Load the RLHF model"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.chunk_rewards = defaultdict(float, data['chunk_rewards'])
                self.query_pattern_rewards = defaultdict(float, data['query_pattern_rewards'])
                self.source_preferences = defaultdict(float, data['source_preferences'])
                self.response_features = defaultdict(lambda: defaultdict(float), data['response_features'])
                self.learning_rate = data.get('learning_rate', 0.1)
                self.exploration_rate = data.get('exploration_rate', 0.1)
                self.temperature = data.get('temperature', 1.0)

# Initialize RLHF model
rlhf_model = RLHFModel()
rlhf_model.load(RLHF_MODEL_FILE)

# === Reward History Tracking ===
def load_reward_history():
    if os.path.exists(REWARD_HISTORY_FILE):
        with open(REWARD_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {"total_feedback": 0, "positive": 0, "negative": 0, "history": []}

def save_reward_history(history):
    with open(REWARD_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# === Cache & feedback helpers ===
def load_cache():
    return json.load(open(CACHE_FILE, "r", encoding="utf-8")) if os.path.exists(CACHE_FILE) else {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4, ensure_ascii=False)

def save_feedback(query, answer, source, feedback, chunks_used):
    data = {"query": query, "answer": answer, "source": source, "feedback": feedback}
    file_exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    
    # Update RLHF model
    rlhf_model.update_rewards(query, chunks_used, answer, source, feedback)
    rlhf_model.save(RLHF_MODEL_FILE)
    
    # Update reward history
    history = load_reward_history()
    history["total_feedback"] += 1
    if feedback == "good":
        history["positive"] += 1
    else:
        history["negative"] += 1
    history["history"].append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "feedback": feedback
    })
    save_reward_history(history)

# === Reset everything ===
def reset_all():
    """Clear cache, feedback log, temp docs, chat state, and RLHF model."""
    open(CACHE_FILE, "w").write("{}")
    open(FEEDBACK_FILE, "w").write("query,answer,source,feedback\n")
    shutil.rmtree("temp_docs", ignore_errors=True)
    chat_log.clear()
    
    # Reset RLHF model
    global rlhf_model
    rlhf_model = RLHFModel()
    rlhf_model.save(RLHF_MODEL_FILE)
    
    # Reset reward history
    save_reward_history({"total_feedback": 0, "positive": 0, "negative": 0, "history": []})
    
    return [], None, None, None, None, None, ""

# === OCR extraction ===
def extract_text_from_image(image_path):
    ocr_result = ocr_model.ocr(image_path, cls=True)
    extracted_text = "\n".join([line[1][0] for block in ocr_result for line in block])
    return extracted_text

# === RLHF-Enhanced FAISS retrieval ===
def search_top_k_with_rlhf(query, k=2, extra_text=None):
    """Enhanced search that uses RLHF scores"""
    embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(embedding).astype("float32"), k * 3)  # Get more candidates
    
    # Get candidate chunks
    candidate_chunks = df_chunks.iloc[indices[0]]
    
    # Calculate RLHF-adjusted scores
    rlhf_scores = rlhf_model.get_chunk_scores(candidate_chunks, query)
    
    # Combine FAISS similarity with RLHF scores
    faiss_scores = 1 / (1 + distances[0])  # Convert distances to similarities
    combined_scores = faiss_scores + rlhf_scores * 0.5  # Weighted combination
    
    # Select chunks using exploration/exploitation
    selected_chunks = rlhf_model.select_chunks_with_exploration(
        candidate_chunks.reset_index(drop=True), 
        combined_scores, 
        k=k
    )
    
    if extra_text:
        extra_df = pd.DataFrame([{"chunk_id": "ocr_chunk", "filename": "uploaded_image", "chunk_text": extra_text}])
        selected_chunks = pd.concat([selected_chunks, extra_df], ignore_index=True)
    
    return selected_chunks

# === Best semantic snippet ===
def find_best_semantic_snippet(chunks_df, question, model, max_length=250):
    question_vec = model.encode([question])[0]
    best_snippet, best_file, best_score = "", "", -1
    for _, row in chunks_df.iterrows():
        sentences = re.split(r'(?<=[.!?]) +', row["chunk_text"])
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            score = cosine_similarity([question_vec], [model.encode([sent])[0]])[0][0]
            if score > best_score:
                best_score = score
                best_snippet = sent
                best_file = row["filename"]
    if not best_snippet:
        return chunks_df.iloc[0]["filename"], chunks_df.iloc[0]["chunk_text"][:max_length].strip() + "..."
    return best_file, best_snippet if len(best_snippet) < max_length else best_snippet[:max_length].strip() + "..."

# === Prompt & answer generation ===
def build_prompt(chunks, question, chat_log=None):
    context = "\n".join(chunks["chunk_text"].tolist())
    history_text = ""
    if chat_log:
        for i, entry in enumerate(chat_log[-2:], start=1):
            history_text += f"Q{i}: {entry['question']}\nA{i}: {entry['answer']}\n"
    prompt = (
        "You are a helpful assistant. Answer the user's question using the provided context.\n\n"
        "### Prior Conversation ###\n" + history_text +
        "### Context from Documents ###\n" + context + "\n\n" +
        "### Question ###\n" + question + "\n\n" +
        "### Answer ###\n"
    )
    return prompt

def generate_answer(chunks, question, chat_log=None):
    response = llm(build_prompt(chunks, question, chat_log), max_tokens=200, stop=["</s>", "###"])
    return response["choices"][0]["text"].strip()

# === Main chat function with RLHF ===
def chatbot_ui(query, history, image_file):
    cache = load_cache()
    ocr_text = extract_text_from_image(image_file.name) if image_file else None

    if query in cache and rlhf_model.exploration_rate == 0:  # Only use cache if not exploring
        answer = cache[query]["answer"]
        source = cache[query]["source"]
        snippet = cache[query]["snippet"]
        filepath = os.path.join("temp_docs", source)
        chunks_used = pd.DataFrame()  # Empty for cached responses
    else:
        chunks_used = search_top_k_with_rlhf(query, k=2, extra_text=ocr_text)
        answer = generate_answer(chunks_used, query, chat_log)
        source, snippet = find_best_semantic_snippet(chunks_used, query, embedding_model)

        local_file_path = os.path.join(text_folder, source)
        os.makedirs("temp_docs", exist_ok=True)
        filepath = os.path.join("temp_docs", source)
        if os.path.exists(local_file_path):
            shutil.copy(local_file_path, filepath)
        else:
            filepath = None

        cache[query] = {"answer": answer, "source": source, "snippet": snippet}
        save_cache(cache)

    chat_log.append({
        "question": query, 
        "answer": answer, 
        "source": source, 
        "snippet": snippet,
        "chunks_used": chunks_used.to_dict('records')
    })

    display = f"💬 Answer: {answer}\n📄 Source: {source}\n🔍 Snippet: {snippet}"
    history.append((query, display))

    from gradio import update
    file_output = filepath if filepath and os.path.exists(filepath) else update(visible=False)
    return history, query, answer, source, chunks_used, file_output

# === Feedback with RLHF ===
def feedback_fn(query, answer, source, chunks_used, feedback, history):
    save_feedback(query, answer, source, feedback, chunks_used)
    reward_emoji = "🎯" if feedback == "good" else "📈"
    return history + [(f"✅ Feedback received: {feedback.upper()} {reward_emoji} (Model learning...)", "")]

# === Export functions ===
def export_chat_to_csv():
    from gradio import update
    if not chat_log:
        return update(value=None, visible=False)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"chat_export_{ts}.csv"
    # Remove chunks_used from export for cleaner CSV
    export_data = [{k: v for k, v in entry.items() if k != 'chunks_used'} for entry in chat_log]
    pd.DataFrame(export_data).to_csv(path, index=False)
    return update(value=path, visible=True)

def export_feedback_to_csv():
    from gradio import update
    if not os.path.exists(FEEDBACK_FILE):
        return update(value=None, visible=False)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = f"feedback_export_{ts}.csv"
    shutil.copy(FEEDBACK_FILE, new_path)
    return update(value=new_path, visible=True)

def export_chat_to_pdf():
    """Generate a nicely formatted PDF of the entire chat_log."""
    from gradio import update
    if not chat_log:
        return update(value=None, visible=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"chat_export_{ts}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 60, "Chatter – Chat Summary Report with RLHF")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 75, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Logo (optional)
    if os.path.exists(logo_path):
        try:
            c.drawImage(ImageReader(logo_path), width - 100, height - 100, width=40, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass

    # Body
    y = height - 130
    c.setFont("Helvetica", 11)
    for i, entry in enumerate(chat_log, start=1):
        block = [
            f"{i}. Question: {entry['question']}",
            f"   Answer: {entry['answer']}",
            f"   Source: {entry['source']}",
            f"   Snippet: {entry['snippet']}",
        ]
        for line in block:
            for wrap_line in wrap(line, width=100):
                if y < 80:
                    c.showPage()
                    c.setFont("Helvetica", 11)
                    y = height - 80
                c.drawString(50, y, wrap_line)
                y -= 16
            y -= 6
    c.save()
    return update(value=pdf_path, visible=True)

# === RLHF Model Stats ===
def get_rlhf_stats():
    """Get RLHF model statistics for display"""
    history = load_reward_history()
    total = history["total_feedback"]
    if total == 0:
        accuracy = 0
    else:
        accuracy = (history["positive"] / total) * 100
    
    stats = f"""
    ### 🤖 RLHF Model Stats
    - **Total Feedback:** {total}
    - **Positive:** {history['positive']} ({accuracy:.1f}%)
    - **Negative:** {history['negative']} ({100-accuracy:.1f}%)
    - **Learning Rate:** {rlhf_model.learning_rate}
    - **Exploration Rate:** {rlhf_model.exploration_rate}
    - **Temperature:** {rlhf_model.temperature}
    
    ### 📊 Top Performing Sources
    """
    
    # Add top sources
    sorted_sources = sorted(rlhf_model.source_preferences.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (source, score) in enumerate(sorted_sources, 1):
        stats += f"\n{i}. {source}: {score:.2f}"
    
    return stats

# === Feedback summary loader ===
def load_feedback_summary():
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame(columns=["query", "answer", "source", "feedback"])
    return pd.read_csv(FEEDBACK_FILE)

# === Update RLHF Parameters ===
def update_rlhf_params(learning_rate, exploration_rate, temperature):
    rlhf_model.learning_rate = learning_rate
    rlhf_model.exploration_rate = exploration_rate
    rlhf_model.temperature = temperature
    rlhf_model.save(RLHF_MODEL_FILE)
    return f"✅ RLHF parameters updated: LR={learning_rate}, ER={exploration_rate}, T={temperature}"

# === Gradio UI ===
with gr.Blocks(title="Chatter 1.5 – RLHF Enhanced") as demo:
    gr.Markdown("## 🤖 Chatter 1.5 – Complete Chatbot with RLHF (Reinforcement Learning from Human Feedback)")

    with gr.Tab("💬 Chat"):
        chatbot = gr.Chatbot()
        query = gr.Textbox(label="Ask something...")
        image_input = gr.File(label="Upload Image for OCR", file_types=["image"])
        file_viewer = gr.File(label="📄 View Full Document", visible=True)

        with gr.Row():
            btn_submit = gr.Button("Submit", variant="primary")
            btn_reset = gr.Button("♻️ Reset Everything")

        with gr.Row():
            btn_good = gr.Button("👍 Good Response", variant="secondary")
            btn_bad = gr.Button("👎 Bad Response", variant="secondary")

    with gr.Tab("📊 RLHF Dashboard"):
        gr.Markdown("### 🎯 Reinforcement Learning Control Panel")
        
        with gr.Row():
            lr_slider = gr.Slider(0.01, 1.0, value=0.1, label="Learning Rate", step=0.01)
            er_slider = gr.Slider(0.0, 0.5, value=0.1, label="Exploration Rate", step=0.01)
            temp_slider = gr.Slider(0.1, 2.0, value=1.0, label="Temperature", step=0.1)
        
        update_params_btn = gr.Button("🔧 Update RLHF Parameters")
        param_status = gr.Textbox(label="Status", interactive=False)
        
        stats_display = gr.Markdown(get_rlhf_stats())
        refresh_stats_btn = gr.Button("🔄 Refresh Stats")

    with gr.Tab("💾 Export & Analysis"):
        # Export buttons
        export_csv_btn = gr.Button("💾 Export Chat to CSV")
        export_file = gr.File(label="📥 Download Chat CSV", visible=False)

        export_pdf_btn = gr.Button("🧾 Export Chat to PDF")
        pdf_file = gr.File(label="📄 Download Chat PDF", visible=False)

        export_feedback_btn = gr.Button("📊 Export Feedback Log")
        export_feedback_file = gr.File(label="📥 Download Feedback CSV", visible=False)

        feedback_summary_btn = gr.Button("📈 Show Feedback Summary")
        feedback_table = gr.DataFrame(headers=["query", "answer", "source", "feedback"], interactive=False, visible=False)

    # State variables
    state_query = gr.State()
    state_answer = gr.State()
    state_source = gr.State()
    state_chunks_used = gr.State()

    # --- Callbacks ---
    btn_submit.click(
        chatbot_ui, 
        inputs=[query, chatbot, image_input], 
        outputs=[chatbot, state_query, state_answer, state_source, state_chunks_used, file_viewer]
    )

    btn_reset.click(
        reset_all, 
        inputs=[], 
        outputs=[chatbot, file_viewer, export_file, export_feedback_file, feedback_table, pdf_file, stats_display]
    )

    btn_good.click(
        feedback_fn, 
        inputs=[state_query, state_answer, state_source, state_chunks_used, gr.Textbox(value="good"), chatbot], 
        outputs=[chatbot]
    )
    
    btn_bad.click(
        feedback_fn, 
        inputs=[state_query, state_answer, state_source, state_chunks_used, gr.Textbox(value="bad"), chatbot], 
        outputs=[chatbot]
    )

    # RLHF parameter updates
    update_params_btn.click(
        update_rlhf_params,
        inputs=[lr_slider, er_slider, temp_slider],
        outputs=[param_status]
    )
    
    refresh_stats_btn.click(
        lambda: get_rlhf_stats(),
        inputs=[],
        outputs=[stats_display]
    )

    # Export callbacks
    export_csv_btn.click(export_chat_to_csv, inputs=[], outputs=[export_file])
    export_feedback_btn.click(export_feedback_to_csv, inputs=[], outputs=[export_feedback_file])
    export_pdf_btn.click(export_chat_to_pdf, inputs=[], outputs=[pdf_file])

    feedback_summary_btn.click(load_feedback_summary, inputs=[], outputs=[feedback_table])
    feedback_summary_btn.click(lambda: gr.update(visible=True), None, [feedback_table])

# === Launch ===
if __name__ == "__main__":
    demo.launch()