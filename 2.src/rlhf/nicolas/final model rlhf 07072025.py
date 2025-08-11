"""
Pipeline RLHF Optimizado para Fine-tuning de Modelos de Lenguaje
Incluye: SFT, Reward Model Training, y PPO con optimizaciones
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, Dataset
from trl import (
    SFTTrainer,
    PPOTrainer, 
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import wandb
from tqdm import tqdm
import gc

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RLHFConfig:
    """Configuración centralizada para todo el pipeline RLHF"""
    # Modelo base
    base_model_name: str = "gpt2"
    
    # Paths
    dataset_path: str = "wiki_hybrid_chunks_300.csv"
    sft_output_dir: str = "./sft_model_optimized"
    reward_output_dir: str = "./reward_model_optimized"
    ppo_output_dir: str = "./ppo_model_optimized"
    
    # Training parameters
    sft_epochs: int = 3
    reward_epochs: int = 3
    ppo_epochs: int = 3
    
    # Batch sizes
    sft_batch_size: int = 8
    reward_batch_size: int = 16
    ppo_batch_size: int = 4
    
    # Learning rates
    sft_lr: float = 5e-5
    reward_lr: float = 2e-5
    ppo_lr: float = 1.41e-5
    
    # Optimization
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    fp16: bool = True
    
    # LoRA config
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Sequence lengths
    max_length: int = 512
    max_prompt_length: int = 256
    
    # PPO specific
    ppo_steps: int = 20000
    chunk_size: int = 128
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

class OptimizedDataset:
    """Dataset optimizado con caching y preprocessing eficiente"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.tokenizer = None
        self.dataset = None
        
    def load_and_preprocess(self, tokenizer: AutoTokenizer) -> Dataset:
        """Carga y preprocesa el dataset con optimizaciones"""
        self.tokenizer = tokenizer
        
        # Cargar dataset
        logger.info(f"Cargando dataset desde {self.config.dataset_path}")
        self.dataset = load_dataset(
            "csv", 
            data_files=self.config.dataset_path,
            cache_dir="./.cache"
        )['train']
        
        # Filtrar ejemplos vacíos o muy cortos
        self.dataset = self.dataset.filter(
            lambda x: len(x['chunk_text']) > 50 if 'chunk_text' in x else False
        )
        
        # Aplicar tokenización con batching para eficiencia
        logger.info("Tokenizando dataset...")
        tokenized_dataset = self.dataset.map(
            self._tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=self.config.num_workers,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    def _tokenize_function(self, examples: Dict) -> Dict:
        """Función de tokenización optimizada con batching"""
        if 'chunk_text' in examples:
            texts = examples['chunk_text']
        else:
            # Combinar prompt y response si están separados
            texts = [
                f"{p} {r}" 
                for p, r in zip(examples.get('prompt', ['']*len(examples)), 
                               examples.get('response', ['']*len(examples)))
            ]
        
        model_inputs = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Para entrenamiento de modelo causal, labels = input_ids
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs

class SupervisedFineTuning:
    """Módulo optimizado para Supervised Fine-Tuning"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            fp16=config.fp16
        )
        
    def train(self, dataset: Dataset) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Entrena el modelo con SFT optimizado"""
        logger.info("Iniciando Supervised Fine-Tuning...")
        
        # Cargar modelo y tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Aplicar LoRA si está habilitado
        if self.config.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["c_attn", "c_proj"]  # Para GPT2
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        # Habilitar gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=self.config.sft_output_dir,
            num_train_epochs=self.config.sft_epochs,
            per_device_train_batch_size=self.config.sft_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.sft_lr,
            fp16=self.config.fp16,
            save_strategy="epoch",
            save_total_limit=2,
            logging_steps=50,
            warmup_steps=100,
            report_to="wandb" if wandb.run else "none",
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.num_workers,
            remove_unused_columns=False
        )
        
        # Data collator para modelos de lenguaje
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # Optimización para tensor cores
        )
        
        # Crear trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            max_seq_length=self.config.max_length,
            packing=True  # Empaqueta múltiples ejemplos en una secuencia
        )
        
        # Entrenar
        trainer.train()
        
        # Guardar modelo
        trainer.save_model(self.config.sft_output_dir)
        tokenizer.save_pretrained(self.config.sft_output_dir)
        
        # Limpiar memoria
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return model, tokenizer

class RewardModelTraining:
    """Módulo para entrenar un modelo de recompensa"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        
    def prepare_reward_dataset(self, dataset: Dataset) -> Dataset:
        """Prepara dataset para entrenamiento del modelo de recompensa"""
        # En un escenario real, necesitarías datos con preferencias humanas
        # Aquí simulamos con scores aleatorios para demostración
        def add_reward_labels(examples):
            # Simular scores de recompensa (en producción vendrían de anotadores)
            batch_size = len(examples[list(examples.keys())[0]])
            examples['labels'] = np.random.randn(batch_size).tolist()
            return examples
        
        reward_dataset = dataset.map(
            add_reward_labels,
            batched=True,
            desc="Adding reward labels"
        )
        
        return reward_dataset
    
    def train(self, dataset: Dataset) -> AutoModelForSequenceClassification:
        """Entrena el modelo de recompensa"""
        logger.info("Entrenando modelo de recompensa...")
        
        # Usar un modelo pre-entrenado como base
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base",  # Mejor que GPT2 para clasificación
            num_labels=1,  # Regresión para scores de recompensa
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32
        )
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        
        # Preparar dataset
        reward_dataset = self.prepare_reward_dataset(dataset)
        
        # Configurar entrenamiento
        training_args = TrainingArguments(
            output_dir=self.config.reward_output_dir,
            num_train_epochs=self.config.reward_epochs,
            per_device_train_batch_size=self.config.reward_batch_size,
            learning_rate=self.config.reward_lr,
            fp16=self.config.fp16,
            save_strategy="epoch",
            logging_steps=50,
            warmup_ratio=0.1,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            dataloader_num_workers=self.config.num_workers
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=reward_dataset,
            tokenizer=tokenizer
        )
        
        # Entrenar
        trainer.train()
        
        # Guardar
        trainer.save_model(self.config.reward_output_dir)
        
        return model

class OptimizedPPOTraining:
    """PPO optimizado con batching eficiente y memoria reducida"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.device = torch.device(config.device)
        
    def train(
        self, 
        sft_model: AutoModelForCausalLM,
        reward_model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        prompts: List[str]
    ):
        """Entrena con PPO optimizado"""
        logger.info("Iniciando entrenamiento PPO optimizado...")
        
        # Crear modelo con value head para PPO
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.sft_output_dir
        ).to(self.device)
        
        # Crear modelo de referencia
        ref_model = create_reference_model(model)
        
        # Configuración PPO optimizada
        ppo_config = PPOConfig(
            model_name=self.config.base_model_name,
            learning_rate=self.config.ppo_lr,
            batch_size=self.config.ppo_batch_size,
            mini_batch_size=self.config.ppo_batch_size // 2,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            ppo_epochs=4,
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=0.1,
            use_score_scaling=True,
            use_score_norm=True,
            score_clip=10.0
        )
        
        # Crear PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            optimizer=torch.optim.AdamW(model.parameters(), lr=self.config.ppo_lr),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        
        # Función de recompensa optimizada
        def compute_rewards_batch(texts: List[str]) -> List[float]:
            """Calcula recompensas en batch para eficiencia"""
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = reward_model(**inputs)
                rewards = outputs.logits.squeeze(-1).cpu().numpy()
            
            return rewards.tolist()
        
        # Generador de batches de prompts
        def prompt_generator():
            """Genera prompts de manera eficiente"""
            prompt_dataset = [
                "Explain machine learning in simple terms.",
                "What are the benefits of renewable energy?",
                "How does the human brain process information?",
                "Describe the process of photosynthesis.",
                "What is quantum computing?",
                # Agregar más prompts diversos
            ]
            
            for _ in range(self.config.ppo_steps):
                yield np.random.choice(prompt_dataset)
        
        # Training loop optimizado
        generation_kwargs = {
            "max_new_tokens": 128,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.8,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        
        progress_bar = tqdm(total=self.config.ppo_steps, desc="PPO Training")
        
        for step, prompt in enumerate(prompt_generator()):
            # Tokenizar prompt
            query_tensor = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generar respuesta
            with torch.no_grad():
                response_tensor = ppo_trainer.generate(
                    query_tensor["input_ids"],
                    **generation_kwargs
                )
            
            # Decodificar respuesta
            response_text = tokenizer.decode(
                response_tensor[0][len(query_tensor["input_ids"][0]):],
                skip_special_tokens=True
            )
            
            # Calcular recompensa
            full_text = prompt + " " + response_text
            reward = compute_rewards_batch([full_text])[0]
            
            # PPO step
            stats = ppo_trainer.step(
                [query_tensor["input_ids"]],
                [response_tensor],
                [torch.tensor([reward])]
            )
            
            # Logging
            if step % 100 == 0:
                logger.info(f"Step {step}: reward={reward:.4f}")
                if wandb.run:
                    wandb.log({"reward": reward, "step": step})
            
            progress_bar.update(1)
            
            # Checkpoint periódico
            if step % 1000 == 0 and step > 0:
                ppo_trainer.save_pretrained(f"{self.config.ppo_output_dir}/checkpoint-{step}")
        
        progress_bar.close()
        
        # Guardar modelo final
        ppo_trainer.save_pretrained(self.config.ppo_output_dir)
        tokenizer.save_pretrained(self.config.ppo_output_dir)
        
        logger.info("Entrenamiento PPO completado!")

def main():
    """Pipeline principal RLHF"""
    # Configuración
    config = RLHFConfig(
        base_model_name="gpt2",
        dataset_path="wiki_hybrid_chunks_300.csv",
        use_lora=True,
        fp16=True,
        gradient_checkpointing=True
    )
    
    # Inicializar wandb (opcional)
    # wandb.init(project="rlhf-optimization", config=config.__dict__)
    
    try:
        # 1. Preparar dataset
        logger.info("=== Fase 1: Preparación de Dataset ===")
        dataset_manager = OptimizedDataset(config)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        dataset = dataset_manager.load_and_preprocess(tokenizer)
        logger.info(f"Dataset preparado: {len(dataset)} ejemplos")
        
        # 2. Supervised Fine-Tuning
        logger.info("\n=== Fase 2: Supervised Fine-Tuning ===")
        sft_trainer = SupervisedFineTuning(config)
        sft_model, tokenizer = sft_trainer.train(dataset)
        
        # 3. Entrenar modelo de recompensa
        logger.info("\n=== Fase 3: Entrenamiento de Modelo de Recompensa ===")
        reward_trainer = RewardModelTraining(config)
        reward_model = reward_trainer.train(dataset)
        
        # 4. PPO Training
        logger.info("\n=== Fase 4: PPO Training ===")
        ppo_trainer = OptimizedPPOTraining(config)
        
        # Generar prompts para PPO
        prompts = [
            "What is artificial intelligence?",
            "Explain climate change.",
            "How do vaccines work?",
            # Agregar más prompts
        ]
        
        ppo_trainer.train(sft_model, reward_model, tokenizer, prompts)
        
        logger.info("\n=== Pipeline RLHF Completado Exitosamente! ===")
        
    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}")
        raise
    
    finally:
        # Limpieza final
        gc.collect()
        torch.cuda.empty_cache()
        # if wandb.run:
        #     wandb.finish()

if __name__ == "__main__":
    main()