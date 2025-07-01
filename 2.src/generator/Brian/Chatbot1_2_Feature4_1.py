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
from gradio import update



# === File paths ===
base_dir = r"C:/Users/brian/OneDrive/Escritorio/Skills/Programming/Python/Project"
faiss_path = os.path.join(base_dir, "faiss_index_PaddleOCR.index")
csv_path = os.path.join(base_dir, "extracted_text_PaddleOCR2/cleaned_text_PaddleOCR2/chunked_text_PaddleOCR.csv")
text_folder = os.path.join(base_dir, "extracted_text_PaddleOCR2/cleaned_text_PaddleOCR2")
mistral_path = r"C:\llama_cpp\llama-b5478-bin-win-cpu-x64\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
CACHE_FILE = "cache.json"
FEEDBACK_FILE = "feedback_log.csv"

# === Load components ===
index = faiss.read_index(faiss_path)
df_chunks = pd.read_csv(csv_path)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(model_path=mistral_path, n_ctx=2048)

# === Session Logs ===
cache = {}
chat_log = []

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

# === Reset cache & feedback ===
def reset_all():
    open(CACHE_FILE, "w").write("{}")
    open(FEEDBACK_FILE, "w").write("query,answer,source,feedback\n")
    shutil.rmtree("temp_docs", ignore_errors=True)
    chat_log.clear()
    return [], None, None

# === FAISS Search ===
def search_top_k(query, k=2):
    embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(embedding).astype("float32"), k)
    return df_chunks.iloc[indices[0]][["chunk_id", "filename", "chunk_text"]]

# === Semantic Snippet Matching ===
def find_best_semantic_snippet(chunks_df, question, model, max_length=250):
    question_vec = model.encode([question])[0]
    best_snippet = ""
    best_score = -1
    best_file = ""

    for _, row in chunks_df.iterrows():
        sentences = re.split(r'(?<=[.!?]) +', row["chunk_text"])
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            sent_vec = model.encode([sent])[0]
            score = cosine_similarity([question_vec], [sent_vec])[0][0]
            if score > best_score:
                best_score = score
                best_snippet = sent
                best_file = row["filename"]

    if not best_snippet:
        return chunks_df.iloc[0]["filename"], chunks_df.iloc[0]["chunk_text"][:max_length].strip() + "..."
    return best_file, best_snippet if len(best_snippet) < max_length else best_snippet[:max_length].strip() + "..."

# === Prompt and LLM generation ===
def build_prompt(chunks, question):
    context = "\n".join(chunks["chunk_text"].tolist())
    return (
        "You are a helpful assistant. Use the information in the context below to answer the user's question.\n\n"
        f"### Context ###\n{context}\n\n"
        f"### Question ###\n{question}\n\n"
        f"### Answer ###\n"
    )

def generate_answer(chunks, question):
    prompt = build_prompt(chunks, question)
    response = llm(prompt, max_tokens=200, stop=["</s>", "###"])
    return response["choices"][0]["text"].strip()

def chatbot_ui(query, history):
    # === Main Chat Function ===
    cache = load_cache()
    
    if query in cache:
        answer = cache[query]["answer"]
        source = cache[query]["source"]
        snippet = cache[query]["snippet"]

        filepath = os.path.join("temp_docs", source)
    else:
        chunks = search_top_k(query, k=2)
        answer = generate_answer(chunks, query)
        source, snippet = find_best_semantic_snippet(chunks, query, embedding_model)

        local_file_path = os.path.join(text_folder, source)
        os.makedirs("temp_docs", exist_ok=True)
        filepath = os.path.join("temp_docs", source)
        #shutil.copy(local_file_path, filepath)
        if os.path.exists(local_file_path):
            shutil.copy(local_file_path, filepath)
        else:
            filepath = None  # prevent file error


        cache[query] = {"answer": answer, "source": source, "snippet": snippet}
        save_cache(cache)

    # Log for CSV export
    chat_log.append({
        "question": query,
        "answer": answer,
        "source": source,
        "snippet": snippet
    })

    display = (
        f"💬 **Answer:** {answer}\n"
        f"📄 **Source:** {source}\n"
        f"🔍 **Snippet:** {snippet}"
    )
    history.append((query, display))
    from gradio import update

    if filepath and os.path.exists(filepath):
        file_output = filepath
    else:
        file_output = update(visible=False)

    return history, query, answer, source, file_output


def feedback_fn(query, answer, source, feedback, history):
    save_feedback(query, answer, source, feedback)
    return history + [("✅ Feedback received: " + feedback.upper(), "")]

def export_chat_to_csv():
    if not chat_log:
        return update(value=None, visible=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = f"chat_export_{timestamp}.csv"
    df = pd.DataFrame(chat_log)
    df.to_csv(export_path, index=False)

    return update(value=export_path, visible=True)



# === Gradio UI ===
with gr.Blocks(title="Chatter 1.4 – Snippet + Source + Export") as demo:
    gr.Markdown("## 🤖 Chatter 1.4 – Citation Chatbot with Export and Reset")

    chatbot = gr.Chatbot()
    query = gr.Textbox(label="Ask something...")
    file_viewer = gr.File(label="📄 View Full Document", visible=True)
    export_button = gr.Button("💾 Export Chat to CSV")
    export_file = gr.File(label="📥 Download Chat CSV", visible=False)

    with gr.Row():
        btn_submit = gr.Button("Submit")
        btn_reset = gr.Button("♻️ Reset Chat & Cache")

    with gr.Row():
        btn_good = gr.Button("👍")
        btn_bad = gr.Button("👎")

    state_query = gr.State()
    state_answer = gr.State()
    state_source = gr.State()
    state_file = gr.State()

    btn_submit.click(
        chatbot_ui,
        inputs=[query, chatbot],
        outputs=[chatbot, state_query, state_answer, state_source, file_viewer]
    )

    btn_reset.click(
        reset_all,
        inputs=[],
        outputs=[chatbot, file_viewer, export_file]
    )

    btn_good.click(feedback_fn, inputs=[state_query, state_answer, state_source, gr.Textbox(value="good"), chatbot], outputs=[chatbot])
    btn_bad.click(feedback_fn, inputs=[state_query, state_answer, state_source, gr.Textbox(value="bad"), chatbot], outputs=[chatbot])

    export_button.click(
        fn=export_chat_to_csv,
        inputs=[],
        outputs=[export_file]
    )

demo.launch()