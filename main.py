import os
import io
import re
import json
import logging
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Configure logging for better debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="AI Career Navigator",
    description="Professional CV Analysis & Career Roadmap Generator",
    version="2.0.0"
)

# CORS middleware to allow cross-origin requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (configure properly for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize OpenAI client for Groq API (alternative to OpenAI)
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    logger.warning("GROQ_API_KEY not found in environment variables. Please set it in .env file.")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"  # Groq API endpoint
)

MODEL = "openai/gpt-oss-20b"  # Model to use for AI inference

# ==================== Models ====================

class ChatRequest(BaseModel):
    """Pydantic model for chat request validation"""
    message: str  # User's message
    history: List[Dict[str, str]] = []  # Chat history for context

# ==================== Helper Functions ====================

def extract_text_from_file(file: UploadFile) -> str:
    """
    Extract text from uploaded PDF or TXT file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Extracted text as string
        
    Raises:
        HTTPException: If file format is unsupported or extraction fails
    """
    filename = file.filename or ""
    content = file.file.read()

    # Validate file size (5MB max to prevent memory issues)
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB.")

    # Handle PDF files using pdfplumber library
    if filename.lower().endswith(".pdf"):
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text_parts = []
                # Extract text from each page
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
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

    # Handle TXT files (plain text)
    elif filename.lower().endswith(".txt"):
        try:
            return content.decode("utf-8", errors="ignore")  # Decode with error handling
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read text file: {str(e)}")

    # Unsupported file format
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or TXT.")


def call_groq_api(messages: List[Dict], temperature: float = 0.3, json_mode: bool = False) -> str:
    """
    Make API call to Groq with proper error handling.
    
    Args:
        messages: List of message objects for the conversation
        temperature: Controls randomness (0.0-1.0, lower = more deterministic)
        json_mode: Whether to request JSON response format
        
    Returns:
        AI response as string
        
    Raises:
        HTTPException: If API call fails
    """
    try:
        # Build API request parameters
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2000,  # Limit response length
        }

        # Add JSON mode if requested
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Make the API call
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


def parse_json_response(text: str) -> Dict:
    """
    Extract JSON from AI response with multiple fallback strategies.
    
    Args:
        text: Raw AI response text
        
    Returns:
        Parsed JSON as dictionary or None if parsing fails
    """
    # Strategy 1: Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Strategy 3: Find any JSON object in the text using regex
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # If all strategies fail, return None
    return None  # type: ignore


def generate_default_analysis() -> Dict:
    """
    Generate a fallback/default analysis response when AI fails.
    
    Returns:
        Default analysis dictionary with placeholder data
    """
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
    """Serve the main HTML frontend page."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse(
        status_code=404,
        content={"error": "index.html not found. Please ensure the file exists."}
    )


@app.post("/analyze")
async def analyze_cv(file: UploadFile = File(...)):
    """
    Analyze CV and generate a comprehensive career roadmap.
    
    Args:
        file: Uploaded CV file (PDF or TXT)
        
    Returns:
        JSON with skills, target roles, missing skills, roadmap, and scores
    """
    try:
        # Step 1: Extract text from uploaded file
        logger.info(f"Processing file: {file.filename}")
        cv_text = await run_in_threadpool(extract_text_from_file, file)

        # Truncate if too long to avoid token limits
        if len(cv_text) > 10000:
            cv_text = cv_text[:10000] + "... [truncated]"

        logger.info(f"Extracted {len(cv_text)} characters of text")

        # Step 2: Build the prompt for AI
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

        # Step 3: Call AI for analysis
        logger.info("Calling Groq API for analysis...")
        ai_response = await run_in_threadpool(
            call_groq_api,
            messages,
            temperature=0.2,  # Low temperature for consistent JSON output
            json_mode=True
        )

        logger.info(f"Received AI response: {len(ai_response)} characters")

        # Step 4: Parse the JSON response
        parsed_data = parse_json_response(ai_response)

        if parsed_data is None:
            logger.error("Failed to parse AI response as JSON")
            logger.debug(f"Raw response: {ai_response[:500]}...")
            return generate_default_analysis()

        # Step 5: Validate and ensure all required fields exist
        required_fields = ["current_skills", "target_roles", "missing_skills", "roadmap", "skill_scores"]
        for field in required_fields:
            if field not in parsed_data:
                # Provide default values for missing fields
                if field == "skill_scores":
                    parsed_data[field] = {"current_proficiency": 50, "target_proficiency": 75}
                else:
                    parsed_data[field] = []

        # Ensure roadmap has at least 3 steps (minimum viable plan)
        if not parsed_data["roadmap"] or len(parsed_data["roadmap"]) < 3:
            parsed_data["roadmap"] = generate_default_analysis()["roadmap"]

        # Validate skill scores are within range
        if "current_proficiency" not in parsed_data["skill_scores"]:
            parsed_data["skill_scores"]["current_proficiency"] = 50
        if "target_proficiency" not in parsed_data["skill_scores"]:
            parsed_data["skill_scores"]["target_proficiency"] = 80

        logger.info("Analysis completed successfully")
        return parsed_data

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error in analyze: {str(e)}", exc_info=True)
        return generate_default_analysis()  # Fallback to default


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat with AI career assistant for interactive guidance.
    
    Args:
        request: ChatRequest with message and history
        
    Returns:
        AI response as JSON
    """
    try:
        # System prompt to set AI's role and behavior
        system_prompt = {
            "role": "system",
            "content": """You are an expert AI Career Navigator and Mentor.
            Provide clear, structured, practical, and encouraging career guidance.
            Be specific, actionable, and professional.
            Use Markdown for formatting (bold, lists, headings).
            Keep responses focused and helpful."""
        }

        # Build conversation with history
        messages = [system_prompt] + request.history + [
            {"role": "user", "content": request.message}
        ]

        # Call AI for chat response
        response = await run_in_threadpool(
            call_groq_api,
            messages,
            temperature=0.5,  # Balanced temperature for conversational responses
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
    """Health check endpoint for monitoring and uptime checks."""
    return {
        "status": "healthy",
        "model": MODEL,
        "api_key_configured": bool(API_KEY)  # Check if API key is present
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for all unhandled exceptions.
    Provides consistent error responses and logging.
    """
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
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    # Start uvicorn server with reload enabled for development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Listen on all interfaces
        port=port,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )