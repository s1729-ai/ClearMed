#!/usr/bin/env python3
"""
Fast GPT-2 Training Script for Medical Q&A
Optimized for quick training with smaller model and fewer examples
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

def main():
    # Fast training configuration
    config = {
        'model_name': 'gpt2',
        'train_file': 'data_prepared/train.jsonl',
        'val_file': 'data_prepared/val.jsonl',
        'output_dir': './outputs/gpt2-medical-fast',
        'max_length': 256,  # Reduced from 512
        'batch_size': 1,    # Reduced from 4
        'num_epochs': 1,    # Reduced from 3
        'learning_rate': 1e-4,  # Higher learning rate
        'warmup_steps': 50,  # Reduced warmup
        'logging_steps': 5,
        'save_steps': 100,   # Save more frequently
        'eval_steps': 100,
        'max_train_examples': 1000,  # Only use 1000 examples
        'max_val_examples': 100,     # Only use 100 validation examples
    }
    
    print("🚀 Starting FAST GPT-2 Medical Training")
    print(f"Configuration: {config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
    model = GPT2LMHeadModel.from_pretrained(config['model_name'])
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
    
    # Load smaller datasets for faster training
    print("Loading smaller datasets...")
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
    
    # Fast training arguments
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
        gradient_accumulation_steps=4,  # Simulate larger batch size
        fp16=False,  # Don't use fp16 on CPU
        dataloader_num_workers=0,  # No multiprocessing
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )
    
    print("🏋️ Starting FAST training...")
    print(f"Total steps: {len(train_dataset) // config['batch_size'] * config['num_epochs']}")
    
    # Train the model
    trainer.train()
    
    print("💾 Saving final model...")
    
    # Save the final model
    final_output_dir = "./model_final"
    os.makedirs(final_output_dir, exist_ok=True)
    
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"✅ Fast training complete! Model saved to {final_output_dir}")
    
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
            "What causes headaches?"
        ]
        
        for test_question in test_questions:
            prompt = f"Q: {test_question}\nA:"
            
            print(f"\n🔍 Test question: {test_question}")
            
            # Generate response
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 80,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            print(f"✨ Generated response: {generated_text}")
        
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    main()
