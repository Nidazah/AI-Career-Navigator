import os
import io
import re
import json
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Career Navigator",
    description="Professional CV Analysis & Career Roadmap Generator",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client for Groq
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    logger.warning("GROQ_API_KEY not found in environment variables. Please set it in .env file.")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"

# ==================== Models ====================

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class AnalysisResponse(BaseModel):
    current_skills: List[str]
    target_roles: List[str]
    missing_skills: List[str]
    roadmap: List[Dict]
    skill_scores: Dict[str, int]

# ==================== Helper Functions ====================

def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from uploaded PDF or TXT file."""
    filename = file.filename or ""
    content = file.file.read()
    
    # Validate file size (5MB max)
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB.")
    
    # Handle PDF files
    if filename.lower().endswith(".pdf"):
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                full_text = "\n".join(text_parts)
                if not full_text.strip():
                    raise HTTPException(
                        status_code=400, 
                        detail="Could not extract text from PDF. The file may be scanned or image-based."
                    )
                return full_text
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")
    
    # Handle TXT files
    elif filename.lower().endswith(".txt"):
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read text file: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or TXT.")

def call_groq_api(messages: List[Dict], temperature: float = 0.3, json_mode: bool = False) -> str:
    """Make API call to Groq with proper error handling."""
    try:
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2000,
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

def parse_json_response(text: str) -> Dict:
    """Extract JSON from AI response with multiple fallback strategies."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON object in the text
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # If all fails, return None
    return None # type: ignore

def generate_default_analysis() -> Dict:
    """Generate default analysis response when AI fails."""
    return {
        "current_skills": ["Communication", "Problem Solving", "Teamwork"],
        "target_roles": ["Software Developer", "Project Manager"],
        "missing_skills": ["Python", "Cloud Computing", "Machine Learning"],
        "roadmap": [
            {
                "step": 1,
                "title": "Skill Assessment & Planning",
                "description": "Evaluate your current skill level and create a personalized learning plan.",
                "duration": "1-2 weeks"
            },
            {
                "step": 2,
                "title": "Core Skill Development",
                "description": "Focus on acquiring the most critical missing skills through courses and projects.",
                "duration": "2-3 months"
            },
            {
                "step": 3,
                "title": "Portfolio Building",
                "description": "Create practical projects to demonstrate your new skills to employers.",
                "duration": "1-2 months"
            }
        ],
        "skill_scores": {
            "current_proficiency": 45,
            "target_proficiency": 80
        }
    }

# ==================== API Endpoints ====================

@app.get("/")
async def serve_frontend():
    """Serve the main HTML page."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse(
        status_code=404,
        content={"error": "index.html not found. Please ensure the file exists."}
    )

@app.post("/analyze")
async def analyze_cv(file: UploadFile = File(...)):
    """Analyze CV and generate career roadmap."""
    try:
        # Step 1: Extract text from file
        logger.info(f"Processing file: {file.filename}")
        cv_text = await run_in_threadpool(extract_text_from_file, file)
        
        # Truncate if too long
        if len(cv_text) > 10000:
            cv_text = cv_text[:10000] + "... [truncated]"
        
        logger.info(f"Extracted {len(cv_text)} characters of text")
        
        # Step 2: Build the prompt
        system_prompt = """You are an expert AI Career Advisor and Technical Recruiter with 20+ years of experience.
        Analyze the CV provided and generate a comprehensive career development plan.
        Return ONLY a valid JSON object with no additional text or explanations."""
        
        user_prompt = f"""
        Analyze this CV and provide a detailed career roadmap. Use this exact JSON structure:
        
        {{
            "current_skills": ["skill1", "skill2", ...],
            "target_roles": ["role1", "role2", ...],
            "missing_skills": ["skill1", "skill2", ...],
            "roadmap": [
                {{
                    "step": 1,
                    "title": "Step title",
                    "description": "Detailed actionable description",
                    "duration": "Estimated time to complete"
                }}
            ],
            "skill_scores": {{
                "current_proficiency": 0-100,
                "target_proficiency": 0-100
            }}
        }}
        
        CV Content:
        {cv_text}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Step 3: Call AI
        logger.info("Calling Groq API for analysis...")
        ai_response = await run_in_threadpool(
            call_groq_api, 
            messages, 
            temperature=0.2, 
            json_mode=True
        )
        
        logger.info(f"Received AI response: {len(ai_response)} characters")
        
        # Step 4: Parse response
        parsed_data = parse_json_response(ai_response)
        
        if parsed_data is None:
            logger.error("Failed to parse AI response as JSON")
            logger.debug(f"Raw response: {ai_response[:500]}...")
            # Return default analysis
            return generate_default_analysis()
        
        # Step 5: Validate and fill missing fields
        required_fields = ["current_skills", "target_roles", "missing_skills", "roadmap", "skill_scores"]
        for field in required_fields:
            if field not in parsed_data:
                if field == "skill_scores":
                    parsed_data[field] = {"current_proficiency": 50, "target_proficiency": 75}
                else:
                    parsed_data[field] = []
        
        # Ensure roadmap has at least 3 steps
        if not parsed_data["roadmap"] or len(parsed_data["roadmap"]) < 3:
            parsed_data["roadmap"] = generate_default_analysis()["roadmap"]
        
        # Validate skill scores
        if "current_proficiency" not in parsed_data["skill_scores"]:
            parsed_data["skill_scores"]["current_proficiency"] = 50
        if "target_proficiency" not in parsed_data["skill_scores"]:
            parsed_data["skill_scores"]["target_proficiency"] = 80
        
        logger.info("Analysis completed successfully")
        return parsed_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze: {str(e)}", exc_info=True)
        # Return default analysis instead of failing
        return generate_default_analysis()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat with AI career assistant."""
    try:
        system_prompt = {
            "role": "system",
            "content": """You are an expert AI Career Navigator and Mentor. 
            Provide clear, structured, practical, and encouraging career guidance.
            Be specific, actionable, and professional.
            Use Markdown for formatting (bold, lists, headings).
            Keep responses focused and helpful."""
        }
        
        messages = [system_prompt] + request.history + [
            {"role": "user", "content": request.message}
        ]
        
        response = await run_in_threadpool(
            call_groq_api,
            messages,
            temperature=0.5,
            json_mode=False
        )
        
        return {"response": response}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return {"response": f"I apologize, but I encountered an error: {str(e)}. Please try again."}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL,
        "api_key_configured": bool(API_KEY)
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please try again later.",
            "error_type": type(exc).__name__
        }
    )

# ==================== Run Server ====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )