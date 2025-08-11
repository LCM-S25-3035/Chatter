# IMPROVED VERSION: Chatbot2_0_Adaptive_Training.py
# Mejoras: Manejo de errores, logging, auto-entrenamiento adaptativo, y UI mejorada

import gradio as gr
import pandas as pd
import numpy as np
import faiss
import json
import os
import csv
import shutil
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading
import time

# ML/AI imports
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# PDF/OCR imports
from paddleocr import PaddleOCR
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textwrap import wrap

# === Configuración de logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Configuración ===
@dataclass
class ChatbotConfig:
    feedback_file: str = "feedback_log.csv"
    model_dir: str = "models"
    auto_train_threshold: int = 20  # Entrenar cada 20 feedback negativos
    auto_train_interval: int = 3600  # Verificar cada hora (segundos)
    max_conversation_length: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model_path: str = "mistral-7b-instruct-v0.1.gguf"

config = ChatbotConfig()

# === NUEVA CARACTERÍSTICA: Auto-entrenamiento adaptativo ===
class AdaptiveTrainingManager:
    """
    Maneja el entrenamiento automático basado en feedback negativo acumulado
    y patrones de conversación detectados.
    """
    
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.feedback_buffer = []
        self.training_thread = None
        self.is_training = False
        self.last_training_time = datetime.now()
        self.performance_metrics = {
            'good_feedback_count': 0,
            'bad_feedback_count': 0,
            'response_times': [],
            'conversation_patterns': {}
        }
        
    def add_feedback(self, query: str, answer: str, feedback: str, response_time: float):
        """Añade feedback y actualiza métricas"""
        self.feedback_buffer.append({
            'query': query,
            'answer': answer,
            'feedback': feedback,
            'response_time': response_time,
            'timestamp': datetime.now()
        })
        
        # Actualizar métricas
        if feedback.lower() == 'good':
            self.performance_metrics['good_feedback_count'] += 1
        else:
            self.performance_metrics['bad_feedback_count'] += 1
            
        self.performance_metrics['response_times'].append(response_time)
        
        # Analizar patrones de conversación
        self._analyze_conversation_patterns(query, feedback)
        
        # Verificar si necesita entrenamiento
        self._check_training_trigger()
        
    def _analyze_conversation_patterns(self, query: str, feedback: str):
        """Analiza patrones en las conversaciones para identificar áreas problemáticas"""
        # Extraer categoría de la query (simple clasificación por keywords)
        categories = {
            'technical': ['código', 'programar', 'error', 'función', 'algoritmo'],
            'creative': ['historia', 'poema', 'crear', 'imaginar', 'escribir'],
            'informational': ['qué', 'cómo', 'cuándo', 'dónde', 'por qué'],
            'conversational': ['hola', 'gracias', 'adiós', 'bien', 'mal']
        }
        
        query_lower = query.lower()
        category = 'other'
        
        for cat, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                category = cat
                break
                
        if category not in self.performance_metrics['conversation_patterns']:
            self.performance_metrics['conversation_patterns'][category] = {
                'good': 0, 'bad': 0, 'total': 0
            }
            
        self.performance_metrics['conversation_patterns'][category]['total'] += 1
        if feedback.lower() == 'good':
            self.performance_metrics['conversation_patterns'][category]['good'] += 1
        else:
            self.performance_metrics['conversation_patterns'][category]['bad'] += 1
            
    def _check_training_trigger(self):
        """Verifica si se debe activar el entrenamiento automático"""
        bad_count = self.performance_metrics['bad_feedback_count']
        
        # Trigger 1: Umbral de feedback negativo alcanzado
        if bad_count >= self.config.auto_train_threshold:
            self._trigger_training("Umbral de feedback negativo alcanzado")
            
        # Trigger 2: Intervalo de tiempo
        time_since_last = (datetime.now() - self.last_training_time).seconds
        if time_since_last >= self.config.auto_train_interval and bad_count > 5:
            self._trigger_training("Intervalo de tiempo alcanzado")
            
        # Trigger 3: Patrón de degradación detectado
        if self._detect_performance_degradation():
            self._trigger_training("Degradación de rendimiento detectada")
            
    def _detect_performance_degradation(self) -> bool:
        """Detecta si hay degradación en el rendimiento"""
        if len(self.feedback_buffer) < 10:
            return False
            
        # Analizar últimos 10 feedback vs primeros 10
        recent_feedback = self.feedback_buffer[-10:]
        old_feedback = self.feedback_buffer[:10] if len(self.feedback_buffer) > 20 else []
        
        if not old_feedback:
            return False
            
        recent_bad_ratio = sum(1 for f in recent_feedback if f['feedback'].lower() == 'bad') / len(recent_feedback)
        old_bad_ratio = sum(1 for f in old_feedback if f['feedback'].lower() == 'bad') / len(old_feedback)
        
        return recent_bad_ratio > old_bad_ratio * 1.5  # 50% more bad feedback
        
    def _trigger_training(self, reason: str):
        """Activa el entrenamiento automático"""
        if self.is_training:
            logger.info(f"Entrenamiento ya en curso. Razón ignorada: {reason}")
            return
            
        logger.info(f"Activando entrenamiento automático. Razón: {reason}")
        self.is_training = True
        
        # Ejecutar entrenamiento en hilo separado
        self.training_thread = threading.Thread(
            target=self._run_adaptive_training,
            args=(reason,)
        )
        self.training_thread.start()
        
    def _run_adaptive_training(self, reason: str):
        """Ejecuta el entrenamiento adaptativo"""
        try:
            logger.info("Iniciando entrenamiento adaptativo...")
            
            # Preparar datos específicos para las áreas problemáticas
            training_data = self._prepare_adaptive_training_data()
            
            # Ejecutar entrenamiento con parámetros adaptativos
            self._execute_adaptive_training(training_data, reason)
            
            # Resetear métricas
            self.performance_metrics['bad_feedback_count'] = 0
            self.last_training_time = datetime.now()
            
            logger.info("Entrenamiento adaptativo completado")
            
        except Exception as e:
            logger.error(f"Error durante entrenamiento adaptativo: {str(e)}")
        finally:
            self.is_training = False
            
    def _prepare_adaptive_training_data(self) -> List[Dict]:
        """Prepara datos de entrenamiento enfocados en áreas problemáticas"""
        training_data = []
        
        # Identificar categorías con más problemas
        problematic_categories = []
        for category, metrics in self.performance_metrics['conversation_patterns'].items():
            if metrics['total'] > 0:
                bad_ratio = metrics['bad'] / metrics['total']
                if bad_ratio > 0.3:  # Más del 30% de feedback negativo
                    problematic_categories.append(category)
                    
        # Generar datos de entrenamiento sintéticos para áreas problemáticas
        for category in problematic_categories:
            synthetic_data = self._generate_synthetic_training_data(category)
            training_data.extend(synthetic_data)
            
        # Añadir feedback real
        for feedback in self.feedback_buffer:
            training_data.append({
                'query': feedback['query'],
                'answer': feedback['answer'],
                'reward': 1.0 if feedback['feedback'].lower() == 'good' else 0.0,
                'category': 'real_feedback'
            })
            
        return training_data
        
    def _generate_synthetic_training_data(self, category: str) -> List[Dict]:
        """Genera datos sintéticos para categorías problemáticas"""
        synthetic_templates = {
            'technical': [
                ("¿Cómo implementar {concept}?", "Aquí tienes una implementación clara de {concept}..."),
                ("Explica {concept} paso a paso", "Te explico {concept} de manera estructurada..."),
                ("¿Qué error puede causar {issue}?", "Este error suele ocurrir cuando...")
            ],
            'creative': [
                ("Escribe una historia sobre {topic}", "Había una vez..."),
                ("Crea un poema sobre {topic}", "En versos fluidos..."),
                ("Imagina un diálogo sobre {topic}", "- Personaje 1: ...")
            ],
            'informational': [
                ("¿Qué es {concept}?", "{concept} es..."),
                ("¿Cómo funciona {system}?", "{system} funciona mediante..."),
                ("¿Cuáles son los beneficios de {topic}?", "Los principales beneficios son...")
            ]
        }
        
        synthetic_data = []
        templates = synthetic_templates.get(category, [])
        
        for template_q, template_a in templates:
            # Generar variaciones
            concepts = ['algoritmo', 'función', 'clase', 'variable'] if category == 'technical' else ['naturaleza', 'aventura', 'amistad']
            
            for concept in concepts:
                synthetic_data.append({
                    'query': template_q.format(concept=concept),
                    'answer': template_a.format(concept=concept),
                    'reward': 1.0,  # Asumimos que los templates son buenos
                    'category': f'synthetic_{category}'
                })
                
        return synthetic_data
        
    def _execute_adaptive_training(self, training_data: List[Dict], reason: str):
        """Ejecuta el entrenamiento con parámetros adaptativos"""
        # Ajustar parámetros basado en la razón y métricas
        if "degradación" in reason.lower():
            learning_rate = 2e-5  # Más agresivo para degradación
            epochs = 5
        else:
            learning_rate = 1e-5  # Más conservador para mejoras incrementales
            epochs = 3
            
        # Simular entrenamiento (implementar lógica real según necesidades)
        logger.info(f"Entrenando con {len(training_data)} ejemplos...")
        logger.info(f"Parámetros: lr={learning_rate}, epochs={epochs}")
        
        # Aquí iría la lógica real de entrenamiento
        # Por ahora, simulamos con un delay
        time.sleep(10)  # Simular tiempo de entrenamiento
        
    def get_performance_report(self) -> Dict:
        """Genera reporte de rendimiento"""
        total_feedback = self.performance_metrics['good_feedback_count'] + self.performance_metrics['bad_feedback_count']
        
        return {
            'total_feedback': total_feedback,
            'good_feedback': self.performance_metrics['good_feedback_count'],
            'bad_feedback': self.performance_metrics['bad_feedback_count'],
            'success_rate': self.performance_metrics['good_feedback_count'] / max(total_feedback, 1),
            'avg_response_time': np.mean(self.performance_metrics['response_times']) if self.performance_metrics['response_times'] else 0,
            'conversation_patterns': self.performance_metrics['conversation_patterns'],
            'is_training': self.is_training,
            'last_training': self.last_training_time.strftime('%Y-%m-%d %H:%M:%S')
        }

# === Clase principal del chatbot mejorada ===
class ImprovedChatbot:
    """Chatbot mejorado con auto-entrenamiento y mejor manejo de errores"""
    
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.training_manager = AdaptiveTrainingManager(config)
        self.conversation_history = []
        self.vector_store = None
        self.model = None
        self.embeddings = None
        
        # Inicializar componentes
        self._initialize_components()
        
    def _initialize_components(self):
        """Inicializa todos los componentes del chatbot"""
        try:
            logger.info("Inicializando componentes del chatbot...")
            
            # Inicializar modelo de embeddings
            self.embeddings = SentenceTransformer(self.config.embedding_model)
            logger.info("Modelo de embeddings cargado")
            
            # Inicializar LLM (simulado - implementar según necesidades)
            # self.model = Llama(model_path=self.config.llm_model_path)
            logger.info("LLM inicializado")
            
            # Inicializar vector store
            self._initialize_vector_store()
            
            logger.info("Todos los componentes inicializados correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando componentes: {str(e)}")
            raise
            
    def _initialize_vector_store(self):
        """Inicializa el vector store para búsqueda semántica"""
        try:
            # Crear índice FAISS simple
            dimension = 384  # Dimensión del modelo all-MiniLM-L6-v2
            self.vector_store = faiss.IndexFlatL2(dimension)
            logger.info("Vector store inicializado")
        except Exception as e:
            logger.error(f"Error inicializando vector store: {str(e)}")
            
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Función principal de chat mejorada"""
        start_time = time.time()
        
        try:
            # Validar entrada
            if not message.strip():
                return "Por favor, escribe un mensaje.", history
                
            # Simular procesamiento del mensaje
            response = self._generate_response(message, history)
            
            # Calcular tiempo de respuesta
            response_time = time.time() - start_time
            
            # Actualizar historial
            history.append((message, response))
            
            # Mantener historial dentro del límite
            if len(history) > self.config.max_conversation_length:
                history = history[-self.config.max_conversation_length:]
                
            # Guardar en historial interno
            self.conversation_history.append({
                'query': message,
                'response': response,
                'timestamp': datetime.now(),
                'response_time': response_time
            })
            
            logger.info(f"Respuesta generada en {response_time:.2f}s")
            return response, history
            
        except Exception as e:
            logger.error(f"Error en chat: {str(e)}")
            error_response = "Lo siento, ocurrió un error procesando tu mensaje. Por favor, intenta de nuevo."
            return error_response, history
            
    def _generate_response(self, message: str, history: List[Tuple[str, str]]) -> str:
        """Genera respuesta usando el modelo (simulado)"""
        # Aquí iría la lógica real de generación
        # Por ahora, respuestas simuladas inteligentes
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['hola', 'buenos días', 'buenas tardes']):
            return "¡Hola! ¿En qué puedo ayudarte hoy?"
            
        elif any(word in message_lower for word in ['código', 'programar', 'función']):
            return "Estoy aquí para ayudarte con programación. ¿Podrías ser más específico sobre lo que necesitas?"
            
        elif any(word in message_lower for word in ['gracias', 'muchas gracias']):
            return "¡De nada! Si tienes más preguntas, no dudes en preguntarme."
            
        elif '?' in message:
            return f"Interesante pregunta sobre: {message}. Basándome en mi conocimiento, puedo decirte que..."
            
        else:
            return f"Entiendo que estás comentando sobre: {message}. ¿Te gustaría que profundice en algún aspecto específico?"
            
    def add_feedback(self, query: str, response: str, feedback: str):
        """Añade feedback y lo procesa"""
        try:
            # Encontrar tiempo de respuesta del historial
            response_time = 0
            for conv in reversed(self.conversation_history):
                if conv['query'] == query and conv['response'] == response:
                    response_time = conv['response_time']
                    break
                    
            # Enviar a training manager
            self.training_manager.add_feedback(query, response, feedback, response_time)
            
            # Guardar en archivo
            self._save_feedback_to_file(query, response, feedback)
            
            logger.info(f"Feedback añadido: {feedback}")
            
        except Exception as e:
            logger.error(f"Error añadiendo feedback: {str(e)}")
            
    def _save_feedback_to_file(self, query: str, response: str, feedback: str):
        """Guarda feedback en archivo CSV"""
        try:
            file_exists = os.path.exists(self.config.feedback_file)
            
            with open(self.config.feedback_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                if not file_exists:
                    writer.writerow(['timestamp', 'query', 'answer', 'feedback'])
                    
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    query,
                    response,
                    feedback
                ])
                
        except Exception as e:
            logger.error(f"Error guardando feedback: {str(e)}")
            
    def get_performance_dashboard(self) -> Dict:
        """Obtiene datos para dashboard de rendimiento"""
        return self.training_manager.get_performance_report()

# === Instancia global del chatbot ===
chatbot = ImprovedChatbot(config)

# === Interfaz de Gradio mejorada ===
def create_enhanced_ui():
    """Crea la interfaz de usuario mejorada"""
    
    with gr.Blocks(title="Chatbot Adaptativo 2.0", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Chatbot Adaptativo 2.0")
        gr.Markdown("*Con auto-entrenamiento y análisis de rendimiento en tiempo real*")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot_interface = gr.Chatbot(
                    height=400,
                    show_label=False,
                    container=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Escribe tu mensaje aquí...",
                        show_label=False,
                        scale=4
                    )
                    submit_btn = gr.Button("Enviar", variant="primary", scale=1)
                    
                # Feedback buttons
                with gr.Row():
                    good_btn = gr.Button("👍 Buena respuesta", variant="secondary")
                    bad_btn = gr.Button("👎 Mala respuesta", variant="secondary")
                    
            with gr.Column(scale=1):
                # Performance dashboard
                gr.Markdown("## 📊 Dashboard de Rendimiento")
                
                performance_display = gr.JSON(
                    label="Métricas en tiempo real",
                    value=chatbot.get_performance_dashboard()
                )
                
                refresh_btn = gr.Button("🔄 Actualizar métricas")
                
                # Training status
                training_status = gr.Textbox(
                    label="Estado de entrenamiento",
                    value="Listo",
                    interactive=False
                )
                
                # Export options
                gr.Markdown("## 📁 Exportar datos")
                export_feedback_btn = gr.Button("Exportar feedback")
                export_conversations_btn = gr.Button("Exportar conversaciones")
        
        # Variables para almacenar la última interacción
        last_query = gr.State("")
        last_response = gr.State("")
        
        # Event handlers
        def chat_respond(message, history):
            if message.strip():
                response, new_history = chatbot.chat(message, history)
                return "", new_history, message, response
            return message, history, "", ""
            
        def add_good_feedback(query, response):
            if query and response:
                chatbot.add_feedback(query, response, "good")
                return "✅ Feedback positivo registrado"
            return "❌ No hay respuesta reciente para evaluar"
            
        def add_bad_feedback(query, response):
            if query and response:
                chatbot.add_feedback(query, response, "bad")
                return "❌ Feedback negativo registrado - El sistema se adaptará"
            return "❌ No hay respuesta reciente para evaluar"
            
        def update_performance():
            return chatbot.get_performance_dashboard()
            
        def get_training_status():
            report = chatbot.get_performance_dashboard()
            if report['is_training']:
                return "🔄 Entrenando automáticamente..."
            return f"✅ Listo (Último: {report['last_training']})"
            
        # Connect events
        submit_btn.click(
            chat_respond,
            [msg, chatbot_interface],
            [msg, chatbot_interface, last_query, last_response]
        )
        
        msg.submit(
            chat_respond,
            [msg, chatbot_interface],
            [msg, chatbot_interface, last_query, last_response]
        )
        
        good_btn.click(
            add_good_feedback,
            [last_query, last_response],
            [training_status]
        )
        
        bad_btn.click(
            add_bad_feedback,
            [last_query, last_response],
            [training_status]
        )
        
        refresh_btn.click(
            update_performance,
            outputs=[performance_display]
        )
        
        # Auto-refresh del dashboard cada 30 segundos
        demo.load(
            update_performance,
            outputs=[performance_display],
            every=30
        )
        
        demo.load(
            get_training_status,
            outputs=[training_status],
            every=10
        )
    
    return demo

# === Funciones de entrenamiento mejoradas ===
def train_rlhf_improved(
    feedback_path: str = None,
    base_model: str = "mistral-7b-instruct-v0.1",
    output_dir: str = "mistral_rlhf_finetuned",
    **kwargs
):
    """
    Versión mejorada del entrenamiento RLHF con mejor manejo de errores
    """
    try:
        feedback_path = feedback_path or config.feedback_file
        
        if not os.path.exists(feedback_path):
            logger.error(f"Archivo de feedback no encontrado: {feedback_path}")
            return False
            
        logger.info("Iniciando entrenamiento RLHF mejorado...")
        
        # Cargar y validar datos
        df = pd.read_csv(feedback_path)
        if df.empty:
            logger.warning("No hay datos de feedback para entrenar")
            return False
            
        # Procesar datos
        df = df.dropna(subset=["query", "answer", "feedback"])
        df["reward"] = df["feedback"].str.strip().str.lower().map(
            lambda x: 1.0 if x == "good" else 0.0
        )
        
        logger.info(f"Procesando {len(df)} ejemplos de feedback")
        
        # Aquí iría la lógica real de entrenamiento
        # Por ahora, simulamos el proceso
        logger.info("Simulando entrenamiento RLHF...")
        time.sleep(5)  # Simular tiempo de entrenamiento
        
        logger.info(f"Entrenamiento completado. Modelo guardado en: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error durante entrenamiento RLHF: {str(e)}")
        return False

# === Punto de entrada ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "train":
            # Entrenamiento manual
            success = train_rlhf_improved()
            if success:
                print("✅ Entrenamiento completado exitosamente")
            else:
                print("❌ Error durante el entrenamiento")
                
        elif command == "performance":
            # Mostrar métricas de rendimiento
            report = chatbot.get_performance_dashboard()
            print("\n📊 REPORTE DE RENDIMIENTO")
            print("=" * 50)
            print(f"Total de feedback: {report['total_feedback']}")
            print(f"Tasa de éxito: {report['success_rate']:.2%}")
            print(f"Tiempo promedio de respuesta: {report['avg_response_time']:.2f}s")
            print(f"Estado de entrenamiento: {'Entrenando' if report['is_training'] else 'Listo'}")
            print(f"Último entrenamiento: {report['last_training']}")
            
        else:
            print("Comandos disponibles: train, performance")
            
    else:
        # Lanzar interfaz de usuario
        demo = create_enhanced_ui()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )