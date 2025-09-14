#!/usr/bin/env python3
"""
Test script for the trained medical LLM model
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from model_loader import MedicalLLM

def test_model_loading():
    """Test if the model can be loaded"""
    print("🧪 Testing Model Loading...")
    
    try:
        llm = MedicalLLM()
        llm.load_model()
        print("✅ Model loaded successfully!")
        return llm
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def test_model_generation(llm):
    """Test if the model can generate responses"""
    if not llm:
        print("❌ Cannot test generation - model not loaded")
        return
    
    print("\n🧪 Testing Model Generation...")
    
    test_questions = [
        "What are the symptoms of diabetes?",
        "How to treat a common cold?",
        "What is hypertension?",
        "What are the benefits of exercise?"
    ]
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        try:
            response = llm.generate_response(question, max_length=100)
            print(f"🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_model_health(llm):
    """Test model health check"""
    if not llm:
        print("❌ Cannot test health - model not loaded")
        return
    
    print("\n🧪 Testing Model Health...")
    
    try:
        is_healthy, message = llm.health_check()
        if is_healthy:
            print(f"✅ Health check passed: {message}")
        else:
            print(f"❌ Health check failed: {message}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def main():
    """Main test function"""
    print("🚀 Medical LLM Model Test Suite")
    print("=" * 50)
    
    # Test 1: Model Loading
    llm = test_model_loading()
    
    if llm:
        # Test 2: Health Check
        test_model_health(llm)
        
        # Test 3: Generation
        test_model_generation(llm)
        
        print("\n🎉 All tests completed!")
    else:
        print("\n❌ Tests failed - model could not be loaded")
        sys.exit(1)

if __name__ == "__main__":
    main()
