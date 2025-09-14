#!/usr/bin/env python3
"""
GPU-Optimized Fast GPT-2 Training Script for Medical Q&A
"""

import json
import torch
import os
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

class MedicalDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256, max_examples: int = 1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Loading dataset from {jsonl_path} (max {max_examples} examples)")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                if line.strip() and count < max_examples:
                    self.examples.append(json.loads(line))
                    count += 1
        
        print(f"Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format the example as Question: ... Answer: ...
        question = example.get('instruction', example.get('input', ''))
        answer = example.get('output', '')
        
        # Create shorter training text
        text = f"Q: {question}\nA: {answer}<|endoftext|>"
        
        # Tokenize with shorter length
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def check_gpu_setup():
    """Check and print GPU information"""
    print("=== GPU Setup Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        return True
    else:
        print("No CUDA GPUs available - using CPU")
        return False

def main():
    # Check GPU setup
    use_gpu = check_gpu_setup()
    
    # GPU-optimized configuration
    if use_gpu:
        config = {
            'model_name': 'gpt2',
            'train_file': 'data_prepared/train.jsonl',
            'val_file': 'data_prepared/val.jsonl',
            'output_dir': './outputs/gpt2-medical-gpu',
            'max_length': 512,  # Can use longer sequences on GPU
            'batch_size': 8,    # Larger batch size for GPU
            'num_epochs': 2,    # More epochs since it's faster
            'learning_rate': 5e-5,  # Standard learning rate
            'warmup_steps': 100,
            'logging_steps': 10,
            'save_steps': 200,
            'eval_steps': 200,
            'max_train_examples': 2000,  # More examples for better training
            'max_val_examples': 200,
        }
    else:
        # CPU fallback configuration (ultra-fast)
        config = {
            'model_name': 'gpt2',
            'train_file': 'data_prepared/train.jsonl',
            'val_file': 'data_prepared/val.jsonl',
            'output_dir': './outputs/gpt2-medical-cpu',
            'max_length': 128,  # Very short for CPU
            'batch_size': 1,    # Minimal batch size
            'num_epochs': 1,    # Single epoch
            'learning_rate': 2e-4,  # Higher learning rate for fast convergence
            'warmup_steps': 20,
            'logging_steps': 5,
            'save_steps': 50,
            'eval_steps': 50,
            'max_train_examples': 500,  # Even fewer examples
            'max_val_examples': 50,
        }
    
    print(f"\n🚀 Starting {'GPU' if use_gpu else 'CPU'} GPT-2 Medical Training")
    print(f"Configuration: {config}")
    
    # Set device
    if use_gpu:
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
    model = GPT2LMHeadModel.from_pretrained(config['model_name'])
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
    
    # Move model to device
    model = model.to(device)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MedicalDataset(
        config['train_file'], 
        tokenizer, 
        config['max_length'],
        config['max_train_examples']
    )
    val_dataset = MedicalDataset(
        config['val_file'], 
        tokenizer, 
        config['max_length'],
        config['max_val_examples']
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        overwrite_output_dir=True,
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        warmup_steps=config['warmup_steps'],
        learning_rate=config['learning_rate'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        eval_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to=None,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_accumulation_steps=2 if use_gpu else 4,
        fp16=use_gpu,  # Use mixed precision on GPU
        dataloader_num_workers=2 if use_gpu else 0,
        # GPU optimizations
        ddp_find_unused_parameters=False if use_gpu else None,
        group_by_length=True,  # Group similar length sequences
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    
    print(f"🏋️ Starting {'GPU' if use_gpu else 'CPU'} training...")
    total_steps = len(train_dataset) // (config['batch_size'] * training_args.gradient_accumulation_steps) * config['num_epochs']
    print(f"Total steps: {total_steps}")
    
    if use_gpu:
        print("GPU training should complete in 2-5 minutes")
    else:
        print("CPU training should complete in 5-10 minutes")
    
    # Train the model
    trainer.train()
    
    print("💾 Saving final model...")
    
    # Save the final model
    final_output_dir = "./model_final"
    os.makedirs(final_output_dir, exist_ok=True)
    
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"✅ Training complete! Model saved to {final_output_dir}")
    
    # Test the model
    print("🧪 Testing the trained model...")
    test_model(final_output_dir)

def test_model(model_path):
    """Test the trained model with sample questions"""
    try:
        print(f"Loading model from {model_path}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test with multiple sample questions
        test_questions = [
            "What are the symptoms of diabetes?",
            "How to treat a cold?",
            "What causes headaches?",
            "What is high blood pressure?"
        ]
        
        for test_question in test_questions:
            prompt = f"Q: {test_question}\nA:"
            
            print(f"\n🔍 Test question: {test_question}")
            
            # Generate response
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            print(f"✨ Generated response: {generated_text}")
        
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    main()
