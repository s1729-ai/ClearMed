#!/usr/bin/env python3
"""
Startup script for Medical AI System
This script will:
1. Check dependencies
2. Test the trained model
3. Start the FastAPI server
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers',
        'fastapi',
        'uvicorn',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    return True

def check_model_files():
    """Check if required model files exist"""
    print("\n🔍 Checking model files...")
    
    required_files = [
        'model.safetensors',
        'config.json',
        'generation_config.json'
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            missing_files.append(file)
            print(f"❌ {file}")
    
    if missing_files:
        print(f"\n❌ Missing model files: {', '.join(missing_files)}")
        return False
    
    return True

def test_model():
    """Test the trained model"""
    print("\n🧪 Testing trained model...")
    
    try:
        from model_loader import MedicalLLM
        
        llm = MedicalLLM()
        llm.load_model()
        
        # Quick health check
        is_healthy, message = llm.health_check()
        if is_healthy:
            print("✅ Model test passed!")
            return True
        else:
            print(f"❌ Model health check failed: {message}")
            return False
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting Medical AI Server...")
    
    try:
        # Start the server
        subprocess.run([
            sys.executable, "server.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🏥 Medical AI System Startup")
    print("=" * 40)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Check model files
    if not check_model_files():
        print("❌ Model files check failed. Exiting.")
        sys.exit(1)
    
    # Step 3: Test model
    if not test_model():
        print("❌ Model test failed. Exiting.")
        sys.exit(1)
    
    print("\n🎉 All checks passed! Starting server...")
    
    # Step 4: Start server
    start_server()

if __name__ == "__main__":
    main()
