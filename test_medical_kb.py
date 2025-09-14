#!/usr/bin/env python3
"""
Test script for the medical knowledge base
"""

from model_loader import MedicalLLM

def test_medical_knowledge_base():
    """Test the medical knowledge base functionality"""
    print("🧪 Testing Medical Knowledge Base...")
    
    # Create LLM instance
    llm = MedicalLLM(use_fallback=True)
    
    # Test medical knowledge base responses
    test_questions = [
        "What are the symptoms of diabetes?",
        "How to treat a headache?",
        "What is hypertension?",
        "How to prevent common cold?",
        "What are the symptoms of cancer?"  # Not in knowledge base
    ]
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        
        # Test medical knowledge base directly
        medical_response = llm._get_medical_response(question)
        if medical_response:
            print(f"✅ Medical KB Response: {medical_response[:200]}...")
        else:
            print("❌ No medical KB response")
        
        # Test quality check
        test_response = "This is a test response with medical information about symptoms and treatment."
        is_good = llm._is_response_quality_good(test_response)
        print(f"Quality check result: {is_good}")

if __name__ == "__main__":
    test_medical_knowledge_base()
