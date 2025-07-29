# UPDATED VERSION: Chatbot1_5_Features_with_PDF.py (adds PDF export feature)

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
base_dir = r"C:/Users/Brian/OneDrive/Escritorio/Skills/Programming/Python/Project"
faiss_path = os.path.join(base_dir, "faiss_index_PaddleOCR.index")
csv_path = os.path.join(base_dir, "extracted_text_PaddleOCR2/cleaned_text_PaddleOCR2/chunked_text_PaddleOCR_hybrid.csv")
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

# === RLHF LIGERO: plantillas e historial de pesos ===
templates = [
    "Template 1: Provide a concise answer without long paragraphs.",
    "Template 2: Provide a detailed answer with examples.",
    "Template 3: Provide a simple and direct answer."
]
WEIGHTS_FILE = "weights.json"

def load_weights():
    if os.path.exists(WEIGHTS_FILE):
        return json.load(open(WEIGHTS_FILE, "r", encoding="utf-8"))
    else:
        # pesos iniciales iguales
        w = [1.0 for _ in templates]
        with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(w, f, indent=4)
        return w

def update_weights(chosen_idx, reward=1.0):
    weights = load_weights()
    weights[chosen_idx] += reward
    # evita que baje de 0.1 para no anular nunca una plantilla
    weights[chosen_idx] = max(weights[chosen_idx], 0.1)
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=4)
    return weights

# === Reset everything ===

def reset_all():
    """Clear cache, feedback log, temp docs, and chat state."""
    open(CACHE_FILE, "w").write("{}")
    open(FEEDBACK_FILE, "w").write("query,answer,source,feedback\n")
    shutil.rmtree("temp_docs", ignore_errors=True)
    chat_log.clear()
    # Return placeholders for: chatbot, file_viewer, export_file, export_feedback_file, feedback_table, pdf_file
    return [], None, None, None, None, None

# === OCR extraction ===

def extract_text_from_image(image_path):
    ocr_result = ocr_model.ocr(image_path, cls=True)
    extracted_text = "\n".join([line[1][0] for block in ocr_result for line in block])
    return extracted_text

# === FAISS retrieval (+ optional OCR chunk) ===

def search_top_k(query, k=2, extra_text=None):
    embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(embedding).astype("float32"), k)
    faiss_results = df_chunks.iloc[indices[0]][["chunk_id", "filename", "chunk_text"]]

    if extra_text:
        extra_df = pd.DataFrame([{"chunk_id": "ocr_chunk", "filename": "uploaded_image", "chunk_text": extra_text}])
        faiss_results = pd.concat([faiss_results, extra_df], ignore_index=True)

    return faiss_results

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

def generate_variants(chunks, question, chat_log=None):
    weights = load_weights()
    total = sum(weights)
    # (opcional) normalizamos
    normalized = [w/total for w in weights]

    variants = []
    for idx, template in enumerate(templates):
        # anteponemos la instrucción al prompt base
        prompt = f"{template}\n\n" + build_prompt(chunks, question, chat_log)
        response = llm(prompt, max_tokens=200, stop=["</s>", "###"])
        answer = response["choices"][0]["text"].strip()
        variants.append((idx, answer))
    return variants

# === Main chat function ===

def chatbot_ui(query, history, image_file):
    cache = load_cache()
    ocr_text = extract_text_from_image(image_file.name) if image_file else None

    if query in cache:
        answer = cache[query]["answer"]
        source = cache[query]["source"]
        snippet = cache[query]["snippet"]
        filepath = os.path.join("temp_docs", source)
    else:
        # 1) obtenemos los chunks
        chunks = search_top_k(query, k=2, extra_text=ocr_text)

        # 2) (opcional) sacamos source/snippet y copiamos el fichero igual que en cache
        source, snippet = find_best_semantic_snippet(chunks, query, embedding_model)
        local_file_path = os.path.join(text_folder, source)
        os.makedirs("temp_docs", exist_ok=True)
        filepath = os.path.join("temp_docs", source)
        if os.path.exists(local_file_path):
            shutil.copy(local_file_path, filepath)
        else:
            filepath = None

        # 3) guardamos meta en caché (sin answer aún)
        cache[query] = {"answer": None, "source": source, "snippet": snippet}
        save_cache(cache)

        # 4) generamos las variantes
        variants = generate_variants(chunks, query, history)

        # 5) formateamos y mostramos
        option_texts = [f"Option {i+1}: {text}" for i, text in variants]
        history.append((query, "\n".join(option_texts)))

        # 6) devolvemos todas las salidas, ocultando el file_viewer
        return history, "", "", "", gr_update(visible=False), variants    

    chat_log.append({"question": query, "answer": answer, "source": source, "snippet": snippet})

    display = f"💬 Answer: {answer}\n📄 Source: {source}\n🔍 Snippet: {snippet}"
    history.append((query, display))

    file_output = filepath if filepath and os.path.exists(filepath) else gr_update(visible=False)
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

# === NEW: Export chat to PDF ===

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
    c.drawString(50, height - 60, "Chatter – Chat Summary Report")
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

# === Feedback summary loader ===

def load_feedback_summary():
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame(columns=["query", "answer", "source", "feedback"])
    return pd.read_csv(FEEDBACK_FILE)

def handle_selection(selection, variants, history):
    # selection ej. "Option 2"
    selected_idx = int(selection.split()[1]) - 1
    template_idx, chosen_text = variants[selected_idx]

    # 1) actualizo pesos
    update_weights(template_idx, reward=1.0)
    # 2) guardo feedback "good" para esta variante
    save_feedback(history[-1][0], chosen_text, f"template_{template_idx}", "good")
    # 3) reemplazo el historial con la respuesta final
    history[-1] = (history[-1][0], f"💬 Selected answer: {chosen_text}")
    return history



# === Gradio UI ===
with gr.Blocks(title="Chatter 1.5 – PDF Enhanced") as demo:
    gr.Markdown("## 🤖 Chatter 1.5 – Complete Chatbot with Feedback & PDF Export")

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

    with gr.Row():
        btn_submit = gr.Button("Submit")
        btn_reset = gr.Button("♻️ Reset Chat & Cache")

    with gr.Row():
        btn_good = gr.Button("👍")
        btn_bad = gr.Button("👎")
        # Estado para almacenar las variantes
        state_variants = gr.State()

        # Radio para elegir cuál variante es la mejor
        variants_radio = gr.Radio(
            choices=[f"Opción {i+1}" for i in range(len(templates))],
            label="Selecciona tu respuesta favorita",
            visible=False
        )
        btn_select = gr.Button("Seleccionar respuesta", visible=False)

    # State vars
    state_query = gr.State()
    state_answer = gr.State()
    state_source = gr.State()
    state_file = gr.State()
    from gradio import update as gr_update
    # --- Callbacks ---
    btn_submit.click(chatbot_ui, inputs=[query, chatbot, image_input], outputs=[chatbot, state_query, state_answer, state_source, file_viewer, state_variants])
    # 1) Cuando lleguen las variantes, muestro el Radio + botón
    state_variants.change(
        lambda variants: (gr_update(visible=True), gr_update(visible=True)),
        inputs=[state_variants],
        outputs=[variants_radio, btn_select]
    )

    # 2) Al hacer click en "Select Answer", proceso la selección
    btn_select.click(
        handle_selection,
        inputs=[variants_radio, state_variants, chatbot],
        outputs=[chatbot]
    )

    btn_reset.click(reset_all, inputs=[], outputs=[chatbot, file_viewer, export_file, export_feedback_file, feedback_table, pdf_file])

    btn_good.click(feedback_fn, inputs=[state_query, state_answer, state_source, gr.Textbox(value="good"), chatbot], outputs=[chatbot])
    btn_bad.click(feedback_fn, inputs=[state_query, state_answer, state_source, gr.Textbox(value="bad"), chatbot], outputs=[chatbot])

    export_csv_btn.click(export_chat_to_csv, inputs=[], outputs=[export_file])
    export_feedback_btn.click(export_feedback_to_csv, inputs=[], outputs=[export_feedback_file])
    export_pdf_btn.click(export_chat_to_pdf, inputs=[], outputs=[pdf_file])

    feedback_summary_btn.click(load_feedback_summary, inputs=[], outputs=[feedback_table])
    feedback_summary_btn.click(lambda: gr.update(visible=True), None, [feedback_table])

# === Launch ===
if __name__ == "__main__":
    demo.launch()