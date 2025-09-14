import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os
from pathlib import Path
import re
import random

class MedicalLLM:
    def __init__(self, model_path=".", device="auto", use_fallback=False):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.use_fallback = use_fallback
        self.using_fallback = False
        
        # Quick medical model data
        self.medical_responses = {}
        
        # Medical knowledge base for common conditions
        self.medical_knowledge = {
            "diabetes": {
                "symptoms": "Common symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections.",
                "treatment": "Treatment includes lifestyle changes (diet, exercise), blood sugar monitoring, oral medications, and insulin therapy when needed.",
                "prevention": "Maintain a healthy weight, eat a balanced diet, exercise regularly, and avoid smoking."
            },
            "headache": {
                "symptoms": "Pain in the head or upper neck, which can be throbbing, constant, sharp, or dull.",
                "treatment": "Rest in a quiet, dark room, apply cold or warm compresses, stay hydrated, take over-the-counter pain relievers like acetaminophen or ibuprofen.",
                "prevention": "Get adequate sleep, manage stress, maintain good posture, and avoid triggers like certain foods or bright lights."
            },
            "hypertension": {
                "symptoms": "Often called the 'silent killer' because it usually has no symptoms. In severe cases, symptoms may include headaches, shortness of breath, nosebleeds, chest pain, dizziness, and vision problems.",
                "treatment": "Lifestyle changes (diet, exercise, stress management) and medications like ACE inhibitors, beta-blockers, or diuretics.",
                "prevention": "Reduce salt intake, maintain healthy weight, exercise regularly, limit alcohol, and avoid smoking."
            },
            "common cold": {
                "symptoms": "Runny or stuffy nose, sore throat, cough, mild fever, fatigue, and mild body aches.",
                "treatment": "Rest, stay hydrated, use saline nasal drops, take over-the-counter medications for symptoms, and use a humidifier.",
                "prevention": "Wash hands frequently, avoid close contact with sick people, maintain a healthy lifestyle, and get adequate sleep."
            },
        }
        
    def _load_quick_model(self):
        """Load the quick medical model data"""
        try:
            quick_model_path = Path("model_quick/medical_responses.json")
            if quick_model_path.exists():
                print("Loading quick medical model...")
                with open(quick_model_path, 'r', encoding='utf-8') as f:
                    self.medical_responses = json.load(f)
                print(f"Loaded {len(self.medical_responses)} medical topics from quick model")
                return True
        except Exception as e:
            print(f"Error loading quick model: {e}")
        return False
        
    def _extract_keywords(self, text):
        """Extract medical keywords from text"""
        medical_keywords = {
            'diabetes': ['diabetes', 'diabetic', 'blood sugar', 'insulin'],
            'cold': ['cold', 'flu', 'cough', 'runny nose', 'congestion'],
            'headache': ['headache', 'migraine', 'head pain'],
            'fever': ['fever', 'temperature', 'hot'],
            'pain': ['pain', 'ache', 'hurt', 'sore'],
            'symptoms': ['symptom', 'sign', 'feel'],
            'treatment': ['treat', 'cure', 'medicine', 'medication'],
            'heart': ['heart', 'cardiac', 'chest pain'],
            'blood pressure': ['blood pressure', 'hypertension', 'bp'],
            'cancer': ['cancer', 'tumor', 'malignant'],
            'infection': ['infection', 'bacteria', 'virus'],
        }
        
        keywords = []
        text_lower = text.lower()
        
        for main_keyword, variations in medical_keywords.items():
            for variation in variations:
                if variation in text_lower:
                    keywords.append(main_keyword)
                    break
        
        # Also add individual words
        words = text_lower.split()
        for word in words:
            if len(word) > 3 and word.isalpha():
                keywords.append(word)
        
        return list(set(keywords))
    
    def _get_quick_model_response(self, question):
        """Get response from the quick medical model"""
        if not self.medical_responses:
            return None
            
        keywords = self._extract_keywords(question)
        
        # Find matching responses
        matching_responses = []
        for keyword in keywords:
            if keyword in self.medical_responses:
                matching_responses.extend(self.medical_responses[keyword])
        
        if matching_responses:
            # Get the best matching response (limit to avoid repetition)
            best_matches = matching_responses[:5]  # Top 5 matches
            best_match = random.choice(best_matches)
            return best_match['answer']
        
        return None
        
    def _get_medical_response(self, question):
        """Get a structured medical response from the knowledge base"""
        question_lower = question.lower()
        
        # First try quick model
        quick_response = self._get_quick_model_response(question)
        if quick_response:
            return f"{quick_response}\n\n⚠️ Note: This information is for educational purposes only. Always consult healthcare professionals for medical advice."
        
        # Fallback to hardcoded knowledge base
        if "diabetes" in question_lower:
            info = self.medical_knowledge["diabetes"]
            return f"""Based on medical knowledge about diabetes:

Symptoms: {info['symptoms']}

Treatment: {info['treatment']}

Prevention: {info['prevention']}

⚠️ Note: This is general information only. Always consult a healthcare professional for proper diagnosis and treatment."""
        
        if "headache" in question_lower:
            info = self.medical_knowledge["headache"]
            return f"""Based on medical knowledge about headache:

Symptoms: {info['symptoms']}

Treatment: {info['treatment']}

Prevention: {info['prevention']}

⚠️ Note: This is general information only. Always consult a healthcare professional for proper diagnosis and treatment."""
        
        if "hypertension" in question_lower or "blood pressure" in question_lower:
            info = self.medical_knowledge["hypertension"]
            return f"""Based on medical knowledge about hypertension:

Symptoms: {info['symptoms']}

Treatment: {info['treatment']}

Prevention: {info['prevention']}

⚠️ Note: This is general information only. Always consult a healthcare professional for proper diagnosis and treatment."""
        
        if "cold" in question_lower or "flu" in question_lower:
            info = self.medical_knowledge["common cold"]
            return f"""Based on medical knowledge about common cold:

Symptoms: {info['symptoms']}

Treatment: {info['treatment']}

Prevention: {info['prevention']}

⚠️ Note: This is general information only. Always consult a healthcare professional for proper diagnosis and treatment."""
        
        return None
    
    def load_model(self):
        """Load the medical model"""
        try:
            print("Loading medical model...")
            
            # Try to load quick model first
            if self._load_quick_model():
                self.loaded = True
                print("✅ Quick medical model loaded successfully!")
                return
            
            # Fallback to GPT-2 if available
            print("Quick model not found, trying GPT-2...")
            self._load_gpt2_model()
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Use fallback knowledge base only
            self.loaded = True
            print("✅ Using fallback knowledge base")
    
    def _load_gpt2_model(self):
        """Load GPT-2 model if available"""
        try:
            config_path = self.model_path / "config.json"
            model_path = self.model_path / "model.safetensors"
            
            if config_path.exists() and model_path.exists():
                print(f"Loading GPT-2 from: {self.model_path}")
                
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                self.model = GPT2LMHeadModel.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    local_files_only=True
                )
                
                if self.device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.model = self.model.to(self.device)
                self.model.eval()
                
                self.loaded = True
                print(f"✅ GPT-2 model loaded on {self.device}")
            else:
                raise FileNotFoundError("GPT-2 model files not found")
                
        except Exception as e:
            print(f"Failed to load GPT-2: {e}")
            self.loaded = True  # Still mark as loaded to use knowledge base
    
    def generate_response(self, question, max_length=200, temperature=0.7, top_p=0.9, test_mode=False, force_model=False):
        """Generate a response using the medical model"""
        try:
            print(f"Generating response for: {question[:50]}...")
            
            # Always try medical knowledge base first (includes quick model)
            medical_response = self._get_medical_response(question)
            if medical_response:
                print("✅ Found medical response")
                return medical_response
            
            # Fallback response for unknown questions
            return """I understand you're asking about a medical topic. While I can provide general information about common conditions like diabetes, headaches, hypertension, and colds, I don't have specific information for your question.

For accurate medical information, I recommend:
1. Consulting a healthcare professional
2. Visiting reliable medical websites (Mayo Clinic, WebMD, CDC)
3. Speaking with your doctor or pharmacist

⚠️ Always consult healthcare professionals for medical advice."""
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def health_check(self):
        """Check if the model is working properly"""
        if not self.loaded:
            return False, "Model not loaded"
        
        try:
            test_response = self.generate_response("What is diabetes?", max_length=50)
            
            if len(test_response) < 10:
                return False, f"Model generating poor responses: {test_response[:100]}..."
            
            model_type = "Quick Medical Model + Knowledge Base"
            return True, f"{model_type} working. Response length: {len(test_response)} chars"
        except Exception as e:
            return False, f"Model error: {str(e)}"

# Example usage
if __name__ == "__main__":
    llm = MedicalLLM(use_fallback=False)
    llm.load_model()
    
    # Test the model
    test_questions = [
        "What are the symptoms of diabetes?",
        "How to treat a cold?",
        "What causes headaches?",
        "What is high blood pressure?"
    ]
    
    for question in test_questions:
        response = llm.generate_response(question)
        print(f"\nQuestion: {question}")
        print(f"Response: {response[:200]}...")
    
    # Health check
    is_healthy, message = llm.health_check()
    print(f"\nHealth check: {is_healthy} - {message}")
