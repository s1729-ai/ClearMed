#!/usr/bin/env python3
"""
Local GPT-2 Training Script for Medical Q&A
This script trains a GPT-2 model on your prepared medical dataset
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
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Loading dataset from {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        
        print(f"Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format the example as Question: ... Answer: ...
        question = example.get('instruction', example.get('input', ''))
        answer = example.get('output', '')
        
        # Create the training text
        text = f"Question: {question}\nAnswer: {answer}<|endoftext|>"
        
        # Tokenize
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
    # Configuration
    config = {
        'model_name': 'gpt2',  # Using base GPT-2
        'train_file': 'data_prepared/train.jsonl',
        'val_file': 'data_prepared/val.jsonl',
        'output_dir': './outputs/gpt2-medical',
        'max_length': 512,
        'batch_size': 4,
        'num_epochs': 3,
        'learning_rate': 5e-5,
        'warmup_steps': 100,
        'logging_steps': 10,
        'save_steps': 500,
        'eval_steps': 500,
    }
    
    print("🚀 Starting GPT-2 Medical Training")
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
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MedicalDataset(config['train_file'], tokenizer, config['max_length'])
    val_dataset = MedicalDataset(config['val_file'], tokenizer, config['max_length'])
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is autoregressive, not masked LM
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
        eval_strategy='steps',  # Changed from evaluation_strategy
        save_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to=None,  # Disable wandb logging
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    print("🏋️ Starting training...")
    
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
    """Test the trained model with a sample question"""
    try:
        # Load the trained model
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test with a sample question
        test_question = "What are the symptoms of diabetes?"
        prompt = f"Question: {test_question}\nAnswer:"
        
        print(f"Test question: {test_question}")
        
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
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
        
        print(f"Generated response: {generated_text}")
        
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    main()
