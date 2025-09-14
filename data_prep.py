import csv
import json
import random
from pathlib import Path
from typing import Iterator, Dict, Any, List

def read_csv_rows(csv_path: Path) -> Iterator[Dict[str, Any]]:
    """Read CSV file and yield rows as dictionaries"""
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def convert_csv_to_training_format(row: Dict[str, Any]) -> Dict[str, str]:
    """Convert a CSV row to training format"""
    question = row.get('question', '').strip()
    answer = row.get('answer', '').strip()
    
    if not question or not answer:
        return None
    
    return {
        'system': "You are a helpful medical assistant. Answer carefully and concisely. If unsure, say you don't know.",
        'instruction': question,
        'input': question,
        'output': answer
    }

def convert_json_to_training_format(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a JSON entry to training format"""
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

def write_jsonl(path: Path, data: list):
    """Write data to JSONL file"""
    with path.open('w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    print("=== Combined Medical Dataset Preparation (4 Datasets) ===")
    
    # Paths for all datasets
    csv_path = Path('medquad.csv')
    test_json_path = Path('test.json')
    train_json_path = Path('train.json')
    valid_json_path = Path('valid.json')
    output_dir = Path('data_prepared')
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    all_training_examples = []
    dataset_stats = {}
    
    # Process CSV dataset
    if csv_path.exists():
        print("Processing CSV dataset (medquad.csv)...")
        rows = list(read_csv_rows(csv_path))
        print(f"Found {len(rows)} rows in CSV")
        
        csv_examples = 0
        for row in rows:
            formatted = convert_csv_to_training_format(row)
            if formatted:
                all_training_examples.append(formatted)
                csv_examples += 1
        
        dataset_stats['CSV'] = csv_examples
        print(f"Generated {csv_examples} training examples from CSV")
    else:
        print("CSV dataset not found, skipping...")
    
    # Process test.json dataset
    if test_json_path.exists():
        print("Processing JSON dataset (test.json)...")
        with test_json_path.open('r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"Found {len(test_data)} entries in test.json")
        
        test_examples = 0
        for entry in test_data:
            examples = convert_json_to_training_format(entry)
            all_training_examples.extend(examples)
            test_examples += len(examples)
        
        dataset_stats['test.json'] = test_examples
        print(f"Generated {test_examples} training examples from test.json")
    else:
        print("test.json dataset not found, skipping...")
    
    # Process train.json dataset
    if train_json_path.exists():
        print("Processing JSON dataset (train.json)...")
        with train_json_path.open('r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        print(f"Found {len(train_data)} entries in train.json")
        
        train_examples = 0
        for entry in train_data:
            examples = convert_json_to_training_format(entry)
            all_training_examples.extend(examples)
            train_examples += len(examples)
        
        dataset_stats['train.json'] = train_examples
        print(f"Generated {train_examples} training examples from train.json")
    else:
        print("train.json dataset not found, skipping...")
    
    # Process valid.json dataset
    if valid_json_path.exists():
        print("Processing JSON dataset (valid.json)...")
        with valid_json_path.open('r', encoding='utf-8') as f:
            valid_data = json.load(f)
        
        print(f"Found {len(valid_data)} entries in valid.json")
        
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
        print("Error: No datasets found. Please ensure at least one dataset is available.")
        return
    
    # Shuffle data
    random.seed(42)
    random.shuffle(all_training_examples)
    
    # Split into train/validation
    split_idx = int(0.95 * len(all_training_examples))
    train_data = all_training_examples[:split_idx]
    val_data = all_training_examples[split_idx:]
    
    # Write files
    train_path = output_dir / 'train.jsonl'
    val_path = output_dir / 'val.jsonl'
    
    write_jsonl(train_path, train_data)
    write_jsonl(val_path, val_data)
    
    print(f"Wrote {len(train_data)} training examples to {train_path}")
    print(f"Wrote {len(val_data)} validation examples to {val_path}")
    
    # Show comprehensive statistics
    print(f"\n=== COMPREHENSIVE DATASET STATISTICS ===")
    print(f"Total training examples: {len(all_training_examples)}")
    print(f"Training set: {len(train_data)}")
    print(f"Validation set: {len(val_data)}")
    
    print(f"\n=== BREAKDOWN BY DATASET ===")
    for dataset, count in dataset_stats.items():
        print(f"{dataset}: {count} examples")
    
    # Show sample questions
    print(f"\n=== SAMPLE QUESTIONS ===")
    for i in range(min(5, len(all_training_examples))):
        print(f"{i+1}. {all_training_examples[i]['instruction']}")

if __name__ == '__main__':
    main()
