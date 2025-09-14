"""
Ultra simple GPT-2 medical training - no complex features
"""
import json
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch
from torch.utils.data import Dataset
import os

class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze()
        }

def load_data(file_path, max_examples=200):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            data = json.loads(line)
            text = f"Q: {data['input']}\nA: {data['output']}"
            texts.append(text)
    return texts

def main():
    print("🚀 Starting Simple GPT-2 Medical Training")
    
    # Load data
    train_texts = load_data("data_prepared/train.jsonl", 200)
    val_texts = load_data("data_prepared/val.jsonl", 20)
    
    print(f"Loaded {len(train_texts)} training examples")
    print(f"Loaded {len(val_texts)} validation examples")
    
    # Setup tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    
    # Create datasets
    train_dataset = SimpleDataset(train_texts, tokenizer)
    val_dataset = SimpleDataset(val_texts, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Minimal training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/gpt2-medical-simple",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=10,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        prediction_loss_only=True,
        dataloader_pin_memory=False,
        max_grad_norm=None,  # Disable gradient clipping
        remove_unused_columns=False,
        report_to=None  # Disable wandb
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save model
    output_dir = "./outputs/gpt2-medical-simple/final"
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Training complete! Model saved to {output_dir}")
    
    # Quick test
    print("\n🧪 Testing model...")
    model.eval()
    test_prompt = "Q: What is diabetes?\nA:"
    
    with torch.no_grad():
        inputs = tokenizer(test_prompt, return_tensors="pt")
        outputs = model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + 50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
