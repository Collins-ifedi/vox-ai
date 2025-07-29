# server.py
import os
import sys
import json
import logging
import asyncio
import random
import time
from http import HTTPStatus
from pathlib import Path
from typing import Dict

# --- NEW: Required for making HTTP requests to Brevo ---
import httpx

# --- Starlette Imports ---
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute, Mount
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket
from starlette.middleware.cors import CORSMiddleware

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Your original application logic is imported here
from trading_assistant_nlp_handler import TradingAssistantNLPHandler

# --- Config ---
MEDIA_DIR = PROJECT_ROOT / "generated_media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UnifiedServer")

# --- Global Session Management ---
ACTIVE_SESSIONS: Dict[int, "ChatSession"] = {}

# --- NEW: In-memory store for verification codes ---
# In a production environment, you would use a more persistent store like Redis or a database.
VERIFICATION_CODES: Dict[str, Dict] = {}


# -------------------------------------------------------
# AI Query Processing (Unchanged)
# -------------------------------------------------------
async def process_api_query(query: str, user_id: str) -> str:
    logger.info(f"[AI] User {user_id} -> {query}")
    try:
        handler_instance = TradingAssistantNLPHandler(user_id=user_id)
        await handler_instance.initialize()
        response_text, _ = await handler_instance.handle_query(query)
        return response_text
    except Exception:
        logger.exception(f"Error in NLP handler for user {user_id}")
        return "An internal AI error occurred."


# -------------------------------------------------------
# WebSocket ChatSession Class (Unchanged)
# -------------------------------------------------------
class ChatSession:
    def __init__(self, websocket: WebSocket, user_id: int):
        self.websocket = websocket
        self.user_id = user_id
        self.nlp_handler = TradingAssistantNLPHandler(user_id=user_id)

    async def initialize(self):
        await self.nlp_handler.initialize()
        welcome = await self.nlp_handler.get_dynamic_welcome()
        await self.send(welcome, "assistant")

    async def send(self, text: str, sender: str):
        await self.websocket.send_text(json.dumps({"type": "new_message", "sender": sender, "text": text}))

    async def handle_user_query(self, text: str):
        response, should_exit = await self.nlp_handler.handle_query(text)
        await self.send(response, "assistant")
        if should_exit:
            await self.websocket.close()


# -------------------------------------------------------
# HTTP API Endpoint (Unchanged)
# -------------------------------------------------------
async def api_generate_endpoint(request):
    """Handles POST requests to generate an AI response."""
    try:
        body = await request.json()
        query = body.get("query")
        user_id = body.get("userId")

        if not query or user_id is None:
            return JSONResponse(
                {"error": "query and userId required"},
                status_code=HTTPStatus.BAD_REQUEST,
            )

        response_text = await process_api_query(query, int(user_id))

        return JSONResponse({"response": response_text})

    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=HTTPStatus.BAD_REQUEST)
    except Exception as e:
        logger.exception(f"HTTP API error: {e}")
        return JSONResponse({"error": "Internal Server Error"}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


# -------------------------------------------------------
# --- NEW: Email Verification Endpoints ---
# -------------------------------------------------------
async def send_verification_email_endpoint(request):
    """Handles sending a verification email using the Brevo API."""
    try:
        body = await request.json()
        email = body.get("email")
        if not email:
            return JSONResponse({"error": "Email is required"}, status_code=HTTPStatus.BAD_REQUEST)

        api_key = os.getenv("EMAIL_API_KEY")
        if not api_key:
            logger.error("EMAIL_API_KEY environment variable not set.")
            return JSONResponse({"error": "Email service is not configured"}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

        code = str(random.randint(100000, 999999))
        VERIFICATION_CODES[email] = {"code": code, "timestamp": time.time()}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.brevo.com/v3/smtp/email",
                headers={"api-key": api_key, "Content-Type": "application/json"},
                json={
                    "sender": {"name": "Voxaroid", "email": "noreply@voxaroid.com"},
                    "to": [{"email": email}],
                    "subject": f"Your Voxaroid Verification Code: {code}",
                    "htmlContent": f"Your verification code is: <strong>{code}</strong>. It will expire in 5 minutes.",
                },
            )

        if response.status_code >= 400:
            logger.error(f"Brevo API error: {response.text}")
            return JSONResponse({"error": "Failed to send verification email"}, status_code=response.status_code)
        
        logger.info(f"Verification email sent to {email}")
        return JSONResponse({"message": "Verification email sent successfully"})

    except Exception as e:
        logger.exception(f"Email sending error: {e}")
        return JSONResponse({"error": "Internal Server Error"}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


async def verify_code_endpoint(request):
    """Handles verifying the 6-digit code provided by the user."""
    try:
        body = await request.json()
        email = body.get("email")
        code = body.get("code")

        if not email or not code:
            return JSONResponse({"error": "Email and code are required"}, status_code=HTTPStatus.BAD_REQUEST)

        stored_info = VERIFICATION_CODES.get(email)
        if not stored_info:
            return JSONResponse({"error": "No verification code found for this email"}, status_code=HTTPStatus.BAD_REQUEST)
        
        # Check if code is older than 5 minutes (300 seconds)
        if time.time() - stored_info["timestamp"] > 300:
            del VERIFICATION_CODES[email]
            return JSONResponse({"error": "Verification code has expired"}, status_code=HTTPStatus.BAD_REQUEST)

        if stored_info["code"] == code:
            del VERIFICATION_CODES[email]  # Clean up used code
            return JSONResponse({"message": "Verification successful"})
        else:
            return JSONResponse({"error": "Invalid verification code"}, status_code=HTTPStatus.BAD_REQUEST)

    except Exception as e:
        logger.exception(f"Code verification error: {e}")
        return JSONResponse({"error": "Internal Server Error"}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

# -------------------------------------------------------
# WebSocket Endpoint (Unchanged)
# -------------------------------------------------------
async def websocket_handler_endpoint(websocket: WebSocket):
    """Handles the WebSocket connection and message flow."""
    await websocket.accept()
    logger.info(f"WebSocket client connected: {websocket.client}")
    session = None
    user_id = None

    try:
        init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        user_id = init_data.get("userId")
        if not user_id or init_data.get("type") != "init":
            await websocket.close(code=1008, reason="Invalid init message")
            return

        session = ChatSession(websocket, user_id)
        ACTIVE_SESSIONS[user_id] = session
        await session.initialize()

        async for message in websocket.iter_text():
            data = json.loads(message)
            if data.get("type") == "send_message":
                await session.handle_user_query(data.get("text", ""))

    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        if user_id and user_id in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[user_id]
            logger.info(f"Session removed for user {user_id}")
        logger.info(f"WebSocket closed for client {websocket.client}")


# -------------------------------------------------------
# Application Setup
# -------------------------------------------------------
# Define the routes for the application
routes = [
    Route("/api/generate", endpoint=api_generate_endpoint, methods=["POST", "OPTIONS"]),
    # --- NEW: Add the email verification routes ---
    Route("/api/send-verification", endpoint=send_verification_email_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/verify-code", endpoint=verify_code_endpoint, methods=["POST", "OPTIONS"]),
    WebSocketRoute("/ws", endpoint=websocket_handler_endpoint),
    # This Mount serves all your frontend files (index.html, main.js, etc.)
    # It MUST be the last route.
    Mount("/", app=StaticFiles(directory=str(PROJECT_ROOT), html=True), name="static"),
]

# Create the main Starlette application instance
app = Starlette(debug=False, routes=routes)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# To run this server:
# 1. Install httpx: pip install httpx
# 2. Set environment variable and run:
#    EMAIL_API_KEY="YOUR_BREVO_KEY_HERE" uvicorn Server:app --host 0.0.0.0 --port 8000