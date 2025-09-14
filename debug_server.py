#!/usr/bin/env python3
"""
Debug script to test the server's model loader
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

from model_loader import MedicalLLM

def debug_model_loader():
    """Debug the model loader to see what's happening"""
    print("🔍 Debugging Model Loader...")
    
    # Create LLM instance
    llm = MedicalLLM(use_fallback=True)
    
    print(f"LLM instance created: {llm}")
    print(f"Medical knowledge base keys: {list(llm.medical_knowledge.keys())}")
    
    # Test a question
    question = "What are the symptoms of diabetes?"
    print(f"\n📝 Testing question: {question}")
    
    # Test medical knowledge base directly
    medical_response = llm._get_medical_response(question)
    print(f"Medical KB response: {medical_response[:200] if medical_response else 'None'}...")
    
    # Test the full generate_response method
    try:
        print("\n🔄 Testing generate_response method...")
        response = llm.generate_response(question, max_length=100)
        print(f"Generated response: {response[:200]}...")
    except Exception as e:
        print(f"❌ Error in generate_response: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_loader()
