import json
import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# Step 1: Install Dependencies
print("Installing dependencies...")
os.system("pip install -q transformers accelerate peft datasets sentencepiece tiktoken protobuf<5")

# Step 2: Configuration
@dataclass
class Config:
    # Using Llama 3.2 3B for better performance
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    data_dir: str = "data_prepared"
    output_dir: str = "outputs/lora-medqa-4datasets"
    bf16: bool = True  # Enable for Colab GPUs
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    max_steps: int = -1
    logging_steps: int = 5
    save_steps: int = 200
    eval_steps: int = 200
    warmup_ratio: float = 0.1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    max_seq_len: int = 512

# Step 3: Data Loading Functions
def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def load_csv_dataset(csv_path: Path):
    """Load the CSV dataset"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df.to_dict('records')

def load_json_dataset(json_path: Path):
    """Load the JSON dataset"""
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def convert_csv_to_training_format(entry):
    """Convert a single CSV entry to training format"""
    training_examples = []
    
    question = entry['question'].strip()
    answer = entry['answer'].strip()
    
    if not question or not answer:
        return []
    
    # Create training example
    training_example = {
        'system': "You are a helpful medical assistant. Answer carefully and concisely. If unsure, say you don't know.",
        'instruction': question,
        'input': question,
        'output': answer
    }
    
    training_examples.append(training_example)
    return training_examples

def convert_json_to_training_format(entry):
    """Convert a single JSON entry to training format"""
    training_examples = []
    
    question = entry['question'].strip()
    context = entry.get('context', '').strip()
    
    # Get the best answer (first one, or use labeled summaries if available)
    if 'labelled_summaries' in entry and entry['labelled_summaries']:
        # Use the information summary if available
        summaries = entry['labelled_summaries']
        if 'INFORMATION_SUMMARY' in summaries:
            answer = summaries['INFORMATION_SUMMARY']
        elif 'SUGGESTION_SUMMARY' in summaries:
            answer = summaries['SUGGESTION_SUMMARY']
        elif 'EXPERIENCE_SUMMARY' in summaries:
            answer = summaries['EXPERIENCE_SUMMARY']
        elif 'CAUSE_SUMMARY' in summaries:
            answer = summaries['CAUSE_SUMMARY']
        else:
            # Fall back to first answer
            answer = entry['answers'][0] if entry['answers'] else ""
    else:
        # Use the first answer
        answer = entry['answers'][0] if entry['answers'] else ""
    
    # Clean up the answer
    answer = answer.strip()
    if not answer:
        return []
    
    # Create training example
    training_example = {
        'system': "You are a helpful medical assistant. Answer carefully and concisely. If unsure, say you don't know.",
        'instruction': question,
        'input': f"{question}{f'\\nContext: {context}' if context else ''}",
        'output': answer
    }
    
    training_examples.append(training_example)
    
    # If there are multiple answers, create additional examples
    for i, additional_answer in enumerate(entry['answers'][1:], 1):
        if additional_answer.strip():
            additional_example = {
                'system': "You are a helpful medical assistant. Answer carefully and concisely. If unsure, say you don't know.",
                'instruction': question,
                'input': f"{question}{f'\\nContext: {context}' if context else ''}",
                'output': additional_answer.strip()
            }
            training_examples.append(additional_example)
    
    return training_examples

# Step 4: Data Preparation
def prepare_data():
    """Prepare the combined medical dataset from all 4 sources"""
    print("Preparing combined medical dataset from all 4 datasets...")
    
    all_training_examples = []
    dataset_stats = {}
    
    # Load and process CSV dataset
    csv_path = Path('medquad.csv')
    if csv_path.exists():
        print("Processing CSV dataset (medquad.csv)...")
        csv_data = load_csv_dataset(csv_path)
        print(f"Loaded {len(csv_data)} entries from medquad.csv")
        
        csv_examples = 0
        for entry in csv_data:
            examples = convert_csv_to_training_format(entry)
            all_training_examples.extend(examples)
            csv_examples += len(examples)
        
        dataset_stats['CSV'] = csv_examples
        print(f"Generated {csv_examples} training examples from CSV")
    else:
        print("CSV dataset not found, skipping...")
    
    # Load and process test.json dataset
    test_json_path = Path('test.json')
    if test_json_path.exists():
        print("Processing JSON dataset (test.json)...")
        test_data = load_json_dataset(test_json_path)
        print(f"Loaded {len(test_data)} entries from test.json")
        
        test_examples = 0
        for entry in test_data:
            examples = convert_json_to_training_format(entry)
            all_training_examples.extend(examples)
            test_examples += len(examples)
        
        dataset_stats['test.json'] = test_examples
        print(f"Generated {test_examples} training examples from test.json")
    else:
        print("test.json dataset not found, skipping...")
    
    # Load and process train.json dataset
    train_json_path = Path('train.json')
    if train_json_path.exists():
        print("Processing JSON dataset (train.json)...")
        train_data = load_json_dataset(train_json_path)
        print(f"Loaded {len(train_data)} entries from train.json")
        
        train_examples = 0
        for entry in train_data:
            examples = convert_json_to_training_format(entry)
            all_training_examples.extend(examples)
            train_examples += len(examples)
        
        dataset_stats['train.json'] = train_examples
        print(f"Generated {train_examples} training examples from train.json")
    else:
        print("train.json dataset not found, skipping...")
    
    # Load and process valid.json dataset
    valid_json_path = Path('valid.json')
    if valid_json_path.exists():
        print("Processing JSON dataset (valid.json)...")
        valid_data = load_json_dataset(valid_json_path)
        print(f"Loaded {len(valid_data)} entries from valid.json")
        
        valid_examples = 0
        for entry in valid_data:
            examples = convert_json_to_training_format(entry)
            all_training_examples.extend(examples)
            valid_examples += len(examples)
        
        dataset_stats['valid.json'] = valid_examples
        print(f"Generated {valid_examples} training examples from valid.json")
    else:
        print("valid.json dataset not found, skipping...")
    
    if not all_training_examples:
        raise FileNotFoundError("No datasets found. Please ensure at least one dataset is available.")
    
    # Shuffle the data
    random.seed(42)
    random.shuffle(all_training_examples)
    
    # Split into train/validation
    n = len(all_training_examples)
    split = max(1, int(0.95 * n))
    train_examples = all_training_examples[:split]
    val_examples = all_training_examples[split:]
    
    # Create output directory
    out_dir = Path('data_prepared')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Write JSONL files
    def write_jsonl(path: Path, examples):
        with path.open('w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    write_jsonl(out_dir / 'train.jsonl', train_examples)
    write_jsonl(out_dir / 'val.jsonl', val_examples)
    
    print(f"Wrote {len(train_examples)} train and {len(val_examples)} val examples to {out_dir}")
    
    # Print comprehensive statistics
    print(f"\n=== COMPREHENSIVE DATASET STATISTICS ===")
    print(f"Total training examples: {len(all_training_examples)}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    print(f"\n=== BREAKDOWN BY DATASET ===")
    for dataset, count in dataset_stats.items():
        print(f"{dataset}: {count} examples")
    
    # Show some sample questions
    print(f"\n=== SAMPLE QUESTIONS ===")
    for i in range(min(5, len(all_training_examples))):
        print(f"{i+1}. {all_training_examples[i]['instruction']}")

# Step 5: Dataset Class
def format_prompt(ex: Dict) -> str:
    system = ex.get("system", "You are a helpful medical assistant.")
    instruction = ex.get("instruction", "")
    user_input = ex.get("input", "")
    output = ex.get("output", "")
    return (
        f"Medical Assistant: {system}\n\n"
        f"Question: {user_input or instruction}\n"
        f"Answer: {output}\n"
    )

class JsonlDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer: AutoTokenizer, max_len: int):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text = format_prompt(self.items[idx])
        toks = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )
        toks = {k: v.squeeze(0) for k, v in toks.items()}
        toks["labels"] = toks["input_ids"].clone()
        return toks

# Step 6: Main Training Function
def main():
    print("=== Medical LLM Training with 4 Combined Datasets ===")
    
    cfg = Config()
    
    # Step 6.1: Prepare Data
    print("\nStep 1: Preparing combined dataset from all 4 sources...")
    prepare_data()
    
    # Step 6.2: Setup Paths
    data_dir = Path(cfg.data_dir)
    train_path = data_dir / 'train.jsonl'
    val_path = data_dir / 'val.jsonl'
    
    if not train_path.exists():
        raise FileNotFoundError("Training data not found. Please ensure datasets are available.")

    # Step 6.3: Load Tokenizer and Model
    print("\nStep 2: Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Step 6.4: Setup LoRA
    print("\nStep 3: Setting up LoRA...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Llama 3.2 target modules
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)

    # Step 6.5: Load Datasets
    print("\nStep 4: Loading datasets...")
    train_items = load_jsonl(train_path)
    val_items = load_jsonl(val_path) if val_path.exists() else []
    train_ds = JsonlDataset(train_items, tokenizer, cfg.max_seq_len)
    eval_ds = JsonlDataset(val_items, tokenizer, cfg.max_seq_len) if val_items else None

    print(f"Train dataset size: {len(train_ds)}")
    if eval_ds:
        print(f"Validation dataset size: {len(eval_ds)}")

    # Step 6.6: Setup Training Arguments
    print("\nStep 5: Setting up training...")
    args = TrainingArguments(
        output_dir=str(Path(cfg.output_dir)),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.eval_steps,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        fp16=False,
        dataloader_pin_memory=True,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    # Step 6.7: Setup Trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # Step 6.8: Train
    print("\nStep 6: Starting training...")
    print("This will take some time. You can monitor progress below.")
    trainer.train()
    
    # Step 6.9: Save Model
    print("\nStep 7: Saving model...")
    model.save_pretrained(str(Path(cfg.output_dir) / 'adapter'))
    tokenizer.save_pretrained(str(Path(cfg.output_dir) / 'adapter'))
    
    print(f"\n=== Training Complete! ===")
    print(f"Model saved to: {cfg.output_dir}/adapter")
    print(f"You can now download the trained model from the '{cfg.output_dir}/adapter' directory")
    print("\nNote: This model was trained on 4 combined datasets for maximum medical knowledge coverage.")

if __name__ == '__main__':
    main()
