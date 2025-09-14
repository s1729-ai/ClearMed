from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from model_loader import MedicalLLM

app = FastAPI(title="Medical Q&A API - Trained Model")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the medical LLM
medical_llm = None

class ChatRequest(BaseModel):
    question: str
    contexts: Optional[List[str]] = []
    max_new_tokens: int = 200
    temperature: float = 0.7

class SummaryRequest(BaseModel):
    content: str
    summary_type: str  # "doctor" or "patient"
    max_new_tokens: int = 300
    temperature: float = 0.5

class ChatResponse(BaseModel):
    answer: str
    model_used: str
    success: bool
    message: str

class SummaryResponse(BaseModel):
    summary: str
    summary_type: str
    model_used: str
    success: bool
    message: str

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts"""
    global medical_llm
    try:
        # Force reload the model loader
        import importlib
        import model_loader
        importlib.reload(model_loader)
        
        # Use the trained model by default (no fallback)
        medical_llm = model_loader.MedicalLLM(use_fallback=False)
        medical_llm.load_model()
        print("✅ Medical LLM loaded successfully on startup!")
        
        # Test the medical knowledge base
        test_response = medical_llm._get_medical_response("What is orgasm?")
        if test_response:
            print("✅ Medical knowledge base is working!")
        else:
            print("❌ Medical knowledge base not working!")
            
    except Exception as e:
        print(f"❌ Failed to load model on startup: {e}")
        # Try with fallback as last resort
        try:
            print("🔄 Trying with fallback model...")
            medical_llm = model_loader.MedicalLLM(use_fallback=True)
            medical_llm.load_model()
            print("✅ Fallback model loaded successfully!")
        except Exception as e2:
            print(f"❌ Failed to load fallback model: {e2}")
            medical_llm = None

@app.get("/health")
async def health_check():
    """Check if the model is working properly"""
    global medical_llm
    
    if medical_llm is None:
        return HealthResponse(
            status="error",
            message="Model not loaded",
            model_loaded=False
        )
    
    try:
        is_healthy, message = medical_llm.health_check()
        return HealthResponse(
            status="healthy" if is_healthy else "error",
            message=message,
            model_loaded=True
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            message=f"Health check failed: {str(e)}",
            model_loaded=True      
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate a medical response using the trained model"""
    global medical_llm

    print(f"Received question: {request.question}")
    
    if medical_llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use better parameters for medical Q&A with the trained model
        response = medical_llm.generate_response(
            question=request.question,
            max_length=min(request.max_new_tokens, 400),  # Increased max length
            temperature=min(request.temperature, 0.8),  # Cap temperature for more coherent responses
            top_p=0.9,  # Higher top_p for better diversity
            force_model=True  # Force use of trained model instead of knowledge base
        )
        
        return ChatResponse(
            answer=response,
            model_used="Trained Medical GPT-2" if not medical_llm.using_fallback else "Fallback GPT-2",
            success=True,
            message="Response generated successfully"
        )
    except Exception as e:
        return ChatResponse(
            answer="",
            model_used="Trained Medical GPT-2" if not medical_llm.using_fallback else "Fallback GPT-2", 
            success=False,
            message=f"Error generating response: {str(e)}"
        )

@app.post("/chat-fallback", response_model=ChatResponse)
async def chat_fallback(request: ChatRequest):
    """Generate a medical response using knowledge base fallback"""
    global medical_llm

    print(f"Received question for fallback: {request.question}")
    
    if medical_llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use knowledge base fallback
        response = medical_llm.generate_response(
            question=request.question,
            max_length=min(request.max_new_tokens, 400),
            temperature=min(request.temperature, 0.8),
            top_p=0.9,
            force_model=False  # Use knowledge base
        )
        
        return ChatResponse(
            answer=response,
            model_used="Knowledge Base Fallback",
            success=True,
            message="Response generated using knowledge base"
        )
    except Exception as e:
        return ChatResponse(
            answer="",
            model_used="Knowledge Base Fallback", 
            success=False,
            message=f"Error generating response: {str(e)}"
        )

@app.post("/generate-summary", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """Generate doctor or patient summary using the LLM"""
    global medical_llm
    
    if medical_llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Build the appropriate prompt based on summary type
        if request.summary_type.lower() == "doctor":
            prompt = f"""As a medical professional, provide a clinical summary using precise medical terminology and technical language. Keep all medical terms, diagnoses, and treatment details. Do NOT simplify medical terminology.
Content to summarize:
{request.content}
Clinical Summary:"""
            
        elif request.summary_type.lower() == "patient":
            prompt = f"""Rewrite this medical information in simple, easy-to-understand language for patients. Avoid medical jargon. Replace complex terms with layman's equivalents. Add safety disclaimers if instructions involve medicine or dosages.
Medical content:
{request.content}
Patient-friendly summary:"""
            
        else:
            return SummaryResponse(
                summary="",
                summary_type=request.summary_type,
                model_used="Trained Medical GPT-2",
                success=False,
                message="Invalid summary type. Use 'doctor' or 'patient'."
            )
        
        # Generate summary using the model
        summary = medical_llm.generate_response(
            question=prompt,
            max_length=request.max_new_tokens,
            temperature=request.temperature
        )
        
        return SummaryResponse(
            summary=summary,
            summary_type=request.summary_type,
            model_used="Trained Medical GPT-2",
            success=True,
            message="Summary generated successfully"
        )
    except Exception as e:
        return SummaryResponse(
            summary="",
            summary_type=request.summary_type,
            model_used="Trained Medical GPT-2",
            success=False,
            message=f"Error generating summary: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    global medical_llm
    
    if medical_llm is None:
        return {"error": "Model not loaded"}
    
    try:
        return {
            "model_type": "GPT-2 (Trained on Medical Q&A)",
            "parameters": f"{medical_llm.model.num_parameters():,}",
            "device": medical_llm.device,
            "loaded": medical_llm.loaded,
            "training_datasets": "4 combined medical datasets (28,562 examples)",
            "base_model": "GPT-2 with custom medical fine-tuning"
        }
    except Exception as e:
        return {"error": f"Failed to get model info: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
