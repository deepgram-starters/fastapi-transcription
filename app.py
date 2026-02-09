"""
FastAPI Transcription Starter - Backend Server

This is a simple FastAPI server that provides a transcription API endpoint
powered by Deepgram's Speech-to-Text service. It's designed to be easily
modified and extended for your own projects.

Key Features:
- Single API endpoint: POST /api/transcription
- Accepts both file uploads and URLs
- JWT session auth with page nonce (production only)
- Async/await for better performance
- Automatic OpenAPI docs at /docs
- Serves built frontend from frontend/dist/
"""

import os
import secrets
import time
from typing import Optional

import jwt
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from deepgram import DeepgramClient
from dotenv import load_dotenv
import toml

# Load .env without overriding existing env vars
load_dotenv(override=False)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "port": int(os.environ.get("PORT", 8081)),
    "host": os.environ.get("HOST", "0.0.0.0"),
}

DEFAULT_MODEL = "nova-3"

# ============================================================================
# SESSION AUTH - JWT tokens with page nonce for production security
# ============================================================================

SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)
REQUIRE_NONCE = bool(os.environ.get("SESSION_SECRET"))

# In-memory nonce store: nonce -> expiry timestamp
session_nonces = {}
NONCE_TTL = 5 * 60  # 5 minutes
JWT_EXPIRY = 3600  # 1 hour


def generate_nonce():
    """Generates a single-use nonce and stores it with an expiry."""
    nonce = secrets.token_hex(16)
    session_nonces[nonce] = time.time() + NONCE_TTL
    return nonce


def consume_nonce(nonce):
    """Validates and consumes a nonce (single-use). Returns True if valid."""
    expiry = session_nonces.pop(nonce, None)
    if expiry is None:
        return False
    return time.time() < expiry


def cleanup_nonces():
    """Remove expired nonces."""
    now = time.time()
    expired = [k for k, v in session_nonces.items() if now >= v]
    for k in expired:
        del session_nonces[k]


# Read frontend/dist/index.html template for nonce injection
_index_html_template = None
try:
    with open(os.path.join(os.path.dirname(__file__), "frontend", "dist", "index.html")) as f:
        _index_html_template = f.read()
except FileNotFoundError:
    pass  # No built frontend (dev mode)


def require_session(authorization: str = Header(None)):
    """FastAPI dependency for JWT session validation."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "MISSING_TOKEN",
                    "message": "Authorization header with Bearer token is required",
                }
            }
        )
    token = authorization[7:]
    try:
        jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "INVALID_TOKEN",
                    "message": "Session expired, please refresh the page",
                }
            }
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "INVALID_TOKEN",
                    "message": "Invalid session token",
                }
            }
        )


# ============================================================================
# API KEY LOADING
# ============================================================================

def load_api_key():
    """
    Loads the Deepgram API key from environment variables
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")

    if not api_key:
        print("\n❌ ERROR: Deepgram API key not found!\n")
        print("Please set your API key using one of these methods:\n")
        print("1. Create a .env file (recommended):")
        print("   DEEPGRAM_API_KEY=your_api_key_here\n")
        print("2. Environment variable:")
        print("   export DEEPGRAM_API_KEY=your_api_key_here\n")
        print("Get your API key at: https://console.deepgram.com\n")
        raise ValueError("DEEPGRAM_API_KEY environment variable is required")

    return api_key

api_key = load_api_key()

# ============================================================================
# SETUP - Initialize FastAPI, Deepgram, and middleware
# ============================================================================

# Initialize Deepgram client with API key
deepgram = DeepgramClient(api_key=api_key)

# Initialize FastAPI app
app = FastAPI(
    title="Deepgram Transcription API",
    description="Speech-to-text transcription powered by Deepgram",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom exception handler to ensure error responses match the contract format
    """
    # If detail is already in the expected format, return it as-is
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )

    # Handle form parsing errors on transcribe endpoint
    if request.url.path == "/api/transcription" and exc.status_code == 400:
        error_msg = str(exc.detail)
        if "parsing" in error_msg.lower() or "multipart" in error_msg.lower():
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "ValidationError",
                        "code": "INVALID_INPUT",
                        "message": "Either 'file' or 'url' must be provided"
                    }
                }
            )

    # Otherwise, wrap it in the expected format
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "Error",
                "code": "ERROR",
                "message": str(exc.detail)
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle FastAPI validation errors - including empty form data
    """
    # For any validation error on the transcribe endpoint, assume it's missing input
    # This handles empty forms gracefully
    if request.url.path == "/api/transcription":
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "ValidationError",
                    "code": "INVALID_INPUT",
                    "message": "Either 'file' or 'url' must be provided"
                }
            }
        )

    # For other endpoints, return generic validation error
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "type": "ValidationError",
                "code": "INVALID_INPUT",
                "message": "Invalid request"
            }
        }
    )

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_transcription_input(file: Optional[UploadFile], url: Optional[str]):
    """
    Validates that either a file or URL was provided in the request

    Args:
        file: FastAPI UploadFile object
        url: URL string

    Returns:
        dict: Input type and data, or None if invalid
    """
    # URL-based transcription
    if url:
        return {"type": "url", "data": url}

    # File-based transcription
    if file:
        return {"type": "file", "data": file}

    # Neither provided
    return None

async def transcribe_audio(input_data, model=DEFAULT_MODEL):
    """
    Sends a transcription request to Deepgram

    Args:
        input_data: dict with 'type' and 'data' keys
        model: Model name to use (e.g., "nova-3")

    Returns:
        dict: Deepgram API response
    """
    # URL transcription
    if input_data["type"] == "url":
        response = deepgram.listen.v1.media.transcribe_url(
            url=input_data["data"],
            model=model,
            smart_format=True,
        )
        return response

    # File transcription
    file_obj = input_data["data"]
    file_content = await file_obj.read()

    response = deepgram.listen.v1.media.transcribe_file(
        request=file_content,
        model=model,
        smart_format=True,
    )
    return response

def format_transcription_response(transcription_response, model_name):
    """
    Formats Deepgram's response into the starter contract format

    Args:
        transcription_response: Raw Deepgram API response
        model_name: Name of model used for transcription

    Returns:
        dict: Formatted response matching the STT contract
    """
    # Access the results from the Deepgram response
    result = transcription_response.results.channels[0].alternatives[0]
    metadata = transcription_response.metadata

    if not result:
        raise ValueError("No transcription results returned from Deepgram")

    # Build response object matching the contract
    response = {
        "transcript": result.transcript or "",
    }

    # Add optional fields if available
    if hasattr(result, 'words') and result.words:
        response["words"] = [
            {
                "text": word.word,
                "start": word.start,
                "end": word.end,
                "speaker": word.speaker if hasattr(word, 'speaker') else None
            }
            for word in result.words
        ]

    if metadata and hasattr(metadata, 'duration'):
        response["duration"] = metadata.duration

    # Add metadata
    response["metadata"] = {
        "model_uuid": metadata.model_uuid if hasattr(metadata, 'model_uuid') else None,
        "request_id": metadata.request_id if hasattr(metadata, 'request_id') else None,
        "model_name": model_name,
    }

    return response

# ============================================================================
# SESSION ROUTES - Auth endpoints (unprotected)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve index.html with injected session nonce (production only)."""
    if not _index_html_template:
        raise HTTPException(status_code=404, detail="Frontend not built. Run make build first.")
    cleanup_nonces()
    nonce = generate_nonce()
    html = _index_html_template.replace(
        "</head>",
        f'<meta name="session-nonce" content="{nonce}">\n</head>'
    )
    return HTMLResponse(content=html)


@app.get("/api/session")
async def get_session(x_session_nonce: str = Header(None)):
    """Issues a JWT. In production, requires valid nonce via X-Session-Nonce header."""
    if REQUIRE_NONCE:
        if not x_session_nonce or not consume_nonce(x_session_nonce):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "type": "AuthenticationError",
                        "code": "INVALID_NONCE",
                        "message": "Valid session nonce required. Please refresh the page.",
                    }
                }
            )
    token = jwt.encode(
        {"iat": int(time.time()), "exp": int(time.time()) + JWT_EXPIRY},
        SESSION_SECRET,
        algorithm="HS256",
    )
    return JSONResponse(content={"token": token})


# ============================================================================
# API ROUTES
# ============================================================================

@app.post("/api/transcription")
async def transcribe(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    model: str = Form(DEFAULT_MODEL),
    _auth=Depends(require_session)
):
    """
    POST /api/transcription

    Main transcription endpoint. Accepts either:
    - A file upload (multipart/form-data with 'file' field)
    - A URL to audio file (form data with 'url' field)

    Optional parameters:
    - model: Deepgram model to use (default: "nova-3")

    Returns:
        JSON response matching the STT contract
    """
    try:
        # Validate input - must have either file or URL
        input_data = validate_transcription_input(file, url)

        if not input_data:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "ValidationError",
                        "code": "INVALID_INPUT",
                        "message": "Either 'file' or 'url' must be provided"
                    }
                }
            )

        # Transcribe the audio
        transcription_response = await transcribe_audio(input_data, model)

        # Format and return the response
        formatted_response = format_transcription_response(
            transcription_response,
            model
        )

        return JSONResponse(content=formatted_response, status_code=200)

    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise

    except ValueError as e:
        # Validation errors (400)
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "ValidationError",
                    "code": "INVALID_INPUT",
                    "message": str(e)
                }
            }
        )

    except Exception as e:
        # Transcription errors (500)
        print(f"Transcription error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "TranscriptionError",
                    "code": "TRANSCRIPTION_FAILED",
                    "message": "Transcription failed. Please try again."
                }
            }
        )

@app.get("/api/metadata")
async def get_metadata():
    """
    GET /api/metadata

    Returns metadata about this starter application from deepgram.toml
    Required for standardization compliance
    """
    try:
        with open('deepgram.toml', 'r') as f:
            config = toml.load(f)

        if 'meta' not in config:
            raise HTTPException(
                status_code=500,
                detail={
                    'error': 'INTERNAL_SERVER_ERROR',
                    'message': 'Missing [meta] section in deepgram.toml'
                }
            )

        return JSONResponse(content=config['meta'], status_code=200)

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'INTERNAL_SERVER_ERROR',
                'message': 'deepgram.toml file not found'
            }
        )

    except Exception as e:
        print(f"Error reading metadata: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'INTERNAL_SERVER_ERROR',
                'message': f'Failed to read metadata from deepgram.toml: {str(e)}'
            }
        )

# ============================================================================
# SERVER START
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    nonce_status = " (nonce required)" if REQUIRE_NONCE else ""
    print("\n" + "=" * 70)
    print(f"🚀 FastAPI Transcription Server running at http://localhost:{CONFIG['port']}")
    print(f"📚 API docs: http://localhost:{CONFIG['port']}/docs")
    print("\nAPI Routes:")
    print(f"  GET  /api/session{nonce_status}")
    print(f"  POST /api/transcription (auth required)")
    print(f"  GET  /api/metadata")
    print("=" * 70 + "\n")

    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
