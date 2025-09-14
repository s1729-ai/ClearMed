import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model, adapter_dir):
    """Load the base model and LoRA adapter"""
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter from: {adapter_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Try to load LoRA adapter
    try:
        model = PeftModel.from_pretrained(model, adapter_dir)
        print("LoRA adapter loaded successfully!")
    except Exception as e:
        print(f"LoRA adapter not found, using base model: {e}")
    
    model.eval()
    return tokenizer, model

def build_prompt(question, system="You are a helpful medical assistant. Answer carefully and concisely."):
    """Build the prompt for the model"""
    return f"Medical Assistant: {system}\n\nQuestion: {question}\nAnswer:"

def generate_response(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7):
    """Generate response from the model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the answer part (after "Answer:")
    if "Answer:" in full_response:
        answer = full_response.split("Answer:")[-1].strip()
    else:
        # Fallback: remove the prompt part
        answer = full_response.replace(prompt, "").strip()
    
    return answer

def main():
    # Configuration
    base_model = "microsoft/DialoGPT-medium"
    adapter_dir = "outputs/lora-medqa-4datasets/adapter"
    
    # Load model
    tokenizer, model = load_model(base_model, adapter_dir)
    
    print("\n=== Medical Q&A Inference (4 Combined Datasets) ===")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            # Get user input
            question = input("Ask a medical question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            # Build prompt and generate response
            prompt = build_prompt(question)
            answer = generate_response(model, tokenizer, prompt)
            
            print(f"\n🤖 Answer: {answer}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
