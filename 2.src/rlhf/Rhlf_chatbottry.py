import gradio as gr
import pandas as pd
import numpy as np
import faiss
import json
import os
import csv
import shutil
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
from paddleocr import PaddleOCR
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textwrap import wrap
from reportlab.lib.utils import ImageReader
from PIL import Image

# === File paths ===
base_dir = r"C:\Users\japje\Documents\B.G\Project"
faiss_path = os.path.join(base_dir, "faiss_index_PaddleOCR.index")
csv_path = os.path.join(base_dir, "extracted_text_PaddleOCR2/cleaned_text_PaddleOCR2/chunked_text_PaddleOCR.csv")
text_folder = os.path.join(base_dir, "extracted_text_PaddleOCR2/cleaned_text_PaddleOCR2")
logo_path = os.path.join(base_dir, "Logo.png")
mistral_path = r"C:\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
CACHE_FILE = "cache.json"
FEEDBACK_FILE = "feedback_log.csv"
REWARD_MODEL_PATH = "reward_model"

# === Load components ===
index = faiss.read_index(faiss_path)
df_chunks = pd.read_csv(csv_path)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(model_path=mistral_path, n_ctx=2048)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

# Load reward model if exists
if os.path.exists(REWARD_MODEL_PATH):
    reward_model = SentenceTransformer(REWARD_MODEL_PATH)
else:
    reward_model = None

# === Session Logs ===
cache = {}
chat_log = []

# === Helpers for cache and feedback ===
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
    open(CACHE_FILE, "w").write("{}")
    open(FEEDBACK_FILE, "w").write("query,answer,source,feedback\n")
    shutil.rmtree("temp_docs", ignore_errors=True)
    chat_log.clear()
    global reward_model
    reward_model = None
    return [], None, None, None, None, None

# === OCR extraction ===
def extract_text_from_image(image_file):
    try:
        if isinstance(image_file, str):
            img = Image.open(image_file)
        else:
            img = Image.open(image_file)
        img_np = np.array(img)
        ocr_result = ocr_model.ocr(img_np, cls=True)
        if not ocr_result:
            return ""
        extracted_text = "\n".join([line[1][0] for block in ocr_result for line in block])
        return extracted_text
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

# === FAISS retrieval ===
def search_top_k(query, k=2, extra_text=None):
    if extra_text:
        return pd.DataFrame([{
            "chunk_id": "ocr_chunk",
            "filename": "uploaded_image",
            "chunk_text": extra_text
        }])
    else:
        embedding = embedding_model.encode([query])
        distances, indices = index.search(np.array(embedding).astype("float32"), k)
        return df_chunks.iloc[indices[0]][["chunk_id", "filename", "chunk_text"]]

# === Semantic snippet ===
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

# === Prompt builder ===
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

# === Generate multiple answers (simulate candidates) ===
def generate_multiple_answers(chunks, question, chat_log=None, n_candidates=3):
    candidates = []
    for _ in range(n_candidates):
        response = llm(build_prompt(chunks, question, chat_log), max_tokens=200, stop=["</s>", "###"])
        text = response["choices"][0]["text"].strip()
        candidates.append(text)
    return candidates

# === Rerank answers with reward model ===
def rerank_answers_with_reward_model(query, answers):
    if reward_model is None:
        # No reward model: just pick first answer
        return answers[0]
    inputs = [query + " " + ans for ans in answers]
    scores = reward_model.predict(inputs)
    best_idx = np.argmax(scores)
    return answers[best_idx]

# === Main chat function ===
def chatbot_ui(query, history, image_file):
    global chat_log, cache

    cache = load_cache()
    ocr_text = extract_text_from_image(image_file) if image_file else None

    import hashlib
    ocr_hash = hashlib.md5((ocr_text or '').encode('utf-8')).hexdigest()[:8] if ocr_text else "no_ocr"
    cache_key = f"{query}_{ocr_hash}"

    if cache_key in cache:
        answer = cache[cache_key]["answer"]
        source = cache[cache_key]["source"]
        snippet = cache[cache_key]["snippet"]
        filepath = os.path.join("temp_docs", source)
    else:
        if image_file is not None:
            chunks = pd.DataFrame([{
                "chunk_id": "ocr_chunk",
                "filename": "uploaded_image",
                "chunk_text": ocr_text or "[No text extracted from uploaded image]"
            }])
        else:
            chunks = search_top_k(query, k=2)

        # Generate multiple answers
        candidates = generate_multiple_answers(chunks, query, chat_log, n_candidates=3)

        # Rerank with reward model
        answer = rerank_answers_with_reward_model(query, candidates)

        source, snippet = find_best_semantic_snippet(chunks, query, embedding_model)

        local_file_path = os.path.join(text_folder, source)
        os.makedirs("temp_docs", exist_ok=True)
        filepath = os.path.join("temp_docs", source)
        if os.path.exists(local_file_path):
            shutil.copy(local_file_path, filepath)
        else:
            filepath = None

        cache[cache_key] = {"answer": answer, "source": source, "snippet": snippet}
        save_cache(cache)

    chat_log.append({"question": query, "answer": answer, "source": source, "snippet": snippet})

    display = f"💬 Answer: {answer}\n📄 Source: {source}\n🔍 Snippet: {snippet}"
    history.append((query, display))

    from gradio import update
    file_output = filepath if filepath and os.path.exists(filepath) else update(visible=False)
    return history, query, answer, source, file_output

# === Feedback & CSV export ===
def feedback_fn(query, answer, source, feedback, history):
    save_feedback(query, answer, source, feedback)
    return history + [("✅ Feedback received: " + feedback.upper(), "")]

def export_chat_to_csv():
    from gradio import update
    if not chat_log:
        return update(value=None, visible=False)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"chat_export_{ts}.csv"
    pd.DataFrame(chat_log).to_csv(path, index=False)
    return update(value=path, visible=True)

def export_feedback_to_csv():
    from gradio import update
    if not os.path.exists(FEEDBACK_FILE):
        return update(value=None, visible=False)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = f"feedback_export_{ts}.csv"
    shutil.copy(FEEDBACK_FILE, new_path)
    return update(value=new_path, visible=True)

# === Export chat to PDF (same as your existing) ===
def export_chat_to_pdf():
    from gradio import update
    if not chat_log:
        return update(value=None, visible=False)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"chat_export_{ts}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 60, "Chatter – Chat Summary Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 75, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if os.path.exists(logo_path):
        try:
            c.drawImage(ImageReader(logo_path), width - 100, height - 100, width=40, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass
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

# === Train reward model from feedback CSV ===
def train_reward_model():
    global reward_model

    if not os.path.exists(FEEDBACK_FILE):
        return "No feedback data to train reward model."

    df = pd.read_csv(FEEDBACK_FILE)
    if df.empty or 'feedback' not in df.columns:
        return "Feedback data is empty or invalid."

    # Map feedback labels to numeric scores
    df['reward'] = df['feedback'].map({'good': 1.0, 'bad': 0.0})

    # Prepare training samples
    examples = []
    for _, row in df.iterrows():
        text = row['query'] + " " + row['answer']
        label = float(row['reward'])
        examples.append(InputExample(texts=[text], label=label))

    if len(examples) < 5:
        return "Not enough feedback data to train reward model. Need at least 5 samples."

    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=8)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=10)

    model.save(REWARD_MODEL_PATH)
    reward_model = model
    return "Reward model trained successfully on feedback data!"

# === Feedback summary loader ===
def load_feedback_summary():
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame(columns=["query", "answer", "source", "feedback"])
    return pd.read_csv(FEEDBACK_FILE)

# === Gradio UI ===
with gr.Blocks(title="Chatter 1.5 – RLHF Enabled with PDF") as demo:
    gr.Markdown("## 🤖 Chatter 1.5 – RLHF Feedback & PDF Enabled Chatbot")

    chatbot = gr.Chatbot()
    query = gr.Textbox(label="Ask something...")
    image_input = gr.File(label="Upload Image for OCR", file_types=["image"])
    file_viewer = gr.File(label="📄 View Full Document", visible=True)

    # Export / download widgets
    export_csv_btn = gr.Button("💾 Export Chat to CSV")
    export_file = gr.File(label="📥 Download Chat CSV", visible=False)

    export_pdf_btn = gr.Button("🧾 Export Chat to PDF")
    pdf_file = gr.File(label="📄 Download Chat PDF", visible=False)

    export_feedback_btn = gr.Button("📊 Export Feedback Log")
    export_feedback_file = gr.File(label="📥 Download Feedback CSV", visible=False)

    feedback_summary_btn = gr.Button("📈 Show Feedback Summary")
    feedback_table = gr.DataFrame(headers=["query", "answer", "source", "feedback"], interactive=False, visible=False)

    train_reward_btn = gr.Button("🔄 Train Reward Model on Feedback")
    train_reward_output = gr.Textbox(label="Reward Model Training Status", interactive=False)

    with gr.Row():
        btn_submit = gr.Button("Submit")
        btn_reset = gr.Button("♻️ Reset Chat & Cache")

    with gr.Row():
        btn_good = gr.Button("👍")
        btn_bad = gr.Button("👎")

    # State vars
    state_query = gr.State()
    state_answer = gr.State()
    state_source = gr.State()
    state_file = gr.State()

    # --- Callbacks ---
    btn_submit.click(chatbot_ui, inputs=[query, chatbot, image_input], outputs=[chatbot, state_query, state_answer, state_source, file_viewer])
    btn_reset.click(reset_all, inputs=[], outputs=[chatbot, file_viewer, export_file, export_feedback_file, feedback_table, pdf_file])

    btn_good.click(feedback_fn, inputs=[state_query, state_answer, state_source, gr.Textbox(value="good"), chatbot], outputs=[chatbot])
    btn_bad.click(feedback_fn, inputs=[state_query, state_answer, state_source, gr.Textbox(value="bad"), chatbot], outputs=[chatbot])

    export_csv_btn.click(export_chat_to_csv, inputs=[], outputs=[export_file])
    export_feedback_btn.click(export_feedback_to_csv, inputs=[], outputs=[export_feedback_file])
    export_pdf_btn.click(export_chat_to_pdf, inputs=[], outputs=[pdf_file])

    feedback_summary_btn.click(load_feedback_summary, inputs=[], outputs=[feedback_table])
    feedback_summary_btn.click(lambda: gr.update(visible=True), None, [feedback_table])

    train_reward_btn.click(train_reward_model, inputs=[], outputs=[train_reward_output])

demo.launch(
    server_name="127.0.0.1",
    server_port=8080,
    debug=True,
    show_error=True,
    show_api=True,
    prevent_thread_lock=True,  # Important for Windows + threads
    share=False  # Turn off tunneling to avoid delays
)