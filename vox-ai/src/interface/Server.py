# server.py
import os
import sys
import json
import logging
import asyncio
import random
import time
import shutil
from http import HTTPStatus
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# --- Third-party Imports ---
import httpx
import stripe
from passlib.context import CryptContext
from jose import JWTError, jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# --- Starlette Imports ---
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute, Mount
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.datastructures import UploadFile
from starlette.dependencies import Depends

# --- Database Imports ---
import databases
import sqlalchemy

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading_assistant_nlp_handler import TradingAssistantNLPHandler

# --- Config & Environment Variables ---
MEDIA_DIR = PROJECT_ROOT / "generated_media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# API Keys and Secrets
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a_secure_default_secret_key_for_development")
FRONTEND_DOMAIN = os.getenv("FRONTEND_DOMAIN", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL")

# Social & Phone Auth Config
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
APPLE_CLIENT_ID = os.getenv("APPLE_CLIENT_ID")

# JWT Config
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
logger = logging.getLogger("VoxaroidServer")

# --- Security & Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- API Client Setup ---
if STRIPE_API_KEY: stripe.api_key = STRIPE_API_KEY

# --- Database Setup ---
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set. Application will not run.")
    sys.exit(1)

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("userId", sqlalchemy.String, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String),
    sqlalchemy.Column("email", sqlalchemy.String, unique=True, index=True),
    sqlalchemy.Column("password_hash", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("phone", sqlalchemy.String, nullable=True, index=True),
    sqlalchemy.Column("is_email_verified", sqlalchemy.Boolean, default=False),
    sqlalchemy.Column("is_phone_verified", sqlalchemy.Boolean, default=True),
    sqlalchemy.Column("subscription_plan", sqlalchemy.String, default="free"),
    sqlalchemy.Column("stripe_customer_id", sqlalchemy.String, nullable=True, index=True),
)

verification_codes = sqlalchemy.Table(
    "verification_codes",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("identifier", sqlalchemy.String, index=True), # Email or phone number
    sqlalchemy.Column("code", sqlalchemy.String),
    sqlalchemy.Column("timestamp", sqlalchemy.Float),
)

# --- START OF MODIFIED SECTION ---
conversation_history = sqlalchemy.Table(
    "conversation_history",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    # Added index=True for faster lookups by user_id
    sqlalchemy.Column("user_id", sqlalchemy.String, sqlalchemy.ForeignKey("users.userId"), index=True),
    sqlalchemy.Column("role", sqlalchemy.String), # 'user' or 'assistant'
    sqlalchemy.Column("content", sqlalchemy.Text),
    sqlalchemy.Column("timestamp", sqlalchemy.DateTime, default=datetime.utcnow),
)
# --- END OF MODIFIED SECTION ---


# --- JWT Token Utilities (Unchanged) ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request) -> Dict:
    """Dependency to get the current authenticated user from a JWT."""
    auth_header = request.headers.get("Authorization")
    credentials_exception = HTTPException(
        status_code=HTTPStatus.UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not auth_header or not auth_header.startswith("Bearer "):
        raise credentials_exception
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    query = users.select().where(users.c.email == email)
    user = await database.fetch_one(query)
    
    if user is None:
        raise credentials_exception # User from token not found in DB
    
    return user

# --- Chat & AI Processing (Unchanged) ---
class ChatSession:
    def __init__(self, websocket: WebSocket, user_id: int):
        self.websocket, self.user_id = websocket, user_id
        self.nlp_handler = TradingAssistantNLPHandler(user_id=user_id)
        logger.info(f"ChatSession created for user_id: {user_id}")
    async def initialize(self):
        await self.nlp_handler.initialize()
        welcome = await self.nlp_handler.get_dynamic_welcome()
        await self.send(welcome, "assistant")
    async def send(self, text: str, sender: str):
        await self.websocket.send_text(json.dumps({"type": "new_message", "sender": sender, "text": text}))
    async def handle_user_query(self, text: str):
        response, should_exit = await self.nlp_handler.handle_query(text)
        await self.send(response, "assistant")
        if should_exit: await self.websocket.close()

async def websocket_handler_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket client connected: {websocket.client}")
    session, user_id = None, None
    try:
        init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        user_id = init_data.get("userId")
        if not user_id or init_data.get("type") != "init":
            await websocket.close(code=1008, reason="Invalid init message")
            return
        
        # In a real app, you might want to verify the user_id against a token here
        session = ChatSession(websocket, user_id)
        await session.initialize()
        async for message in websocket.iter_text():
            data = json.loads(message)
            if data.get("type") == "send_message": await session.handle_user_query(data.get("text", ""))
    except asyncio.TimeoutError: logger.warning("WebSocket init timed out.")
    except Exception as e: logger.exception(f"WebSocket error: {e}")
    finally:
        logger.info(f"WebSocket closed for client {websocket.client}")

# --- Helper Functions ---
def verify_password(plain_password, hashed_password): return pwd_context.verify(plain_password, hashed_password)
def get_password_hash(password): return pwd_context.hash(password)
def generate_code() -> str: return str(random.randint(100000, 999999))

async def send_email_with_brevo(email: str, subject: str, html_content: str):
    if not BREVO_API_KEY:
        logger.error("BREVO_API_KEY not configured."); raise HTTPException(500, "Email service not available.")
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.brevo.com/v3/smtp/email", headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"}, json={"sender": {"name": "Voxaroid", "email": "noreply@voxaroid.com"}, "to": [{"email": email}], "subject": subject, "htmlContent": html_content})
    response.raise_for_status()


async def send_sms_with_brevo(phone_number: str, body: str):
    if not BREVO_API_KEY:
        logger.error("BREVO_API_KEY not configured for SMS.")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="SMS service is not configured.")
    sender_name = "Voxaroid" 
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.brevo.com/v3/transactionalSMS/sms",
                headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
                json={"sender": sender_name, "recipient": phone_number, "content": body, "type": "transactional"}
            )
            response.raise_for_status()
            logger.info(f"Brevo SMS sent successfully to {phone_number}")
        except httpx.HTTPStatusError as e:
            logger.exception(f"Brevo SMS failed for {phone_number}. Status: {e.response.status_code}, Response: {e.response.text}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to send SMS verification code.")
        except httpx.RequestError as e:
            logger.exception(f"Network error while sending SMS to {phone_number}: {e}")
            raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="SMS service is currently unavailable.")

# --- Authentication Endpoints ---
async def signup_endpoint(request: Request):
    body = await request.json()
    name, email, password, phone = body.get("name"), body.get("email"), body.get("password"), body.get("phone")
    if not all([name, email, password]): raise HTTPException(400, "Name, email, and password are required")
    
    query = users.select().where(users.c.email == email)
    if await database.fetch_one(query):
        raise HTTPException(409, "User with this email already exists")

    hashed_password = get_password_hash(password)
    user_id = f"user_{int(time.time())}"
    
    insert_query = users.insert().values(
        userId=user_id, name=name, email=email, password_hash=hashed_password, phone=phone,
        is_email_verified=False, is_phone_verified=False if phone else True
    )
    
    code = generate_code()
    verification_insert = verification_codes.insert().values(identifier=email, code=code, timestamp=time.time())

    try:
        async with database.transaction():
            await database.execute(insert_query)
            await database.execute(verification_insert)
            await send_email_with_brevo(email=email, subject=f"Your Voxaroid Verification Code: {code}", html_content=f"Welcome to Voxaroid! Your verification code is: <strong>{code}</strong>. It will expire in 5 minutes.")
        
        logger.info(f"Signup initiated for {email}")
        return JSONResponse({"message": "Account created. Please verify your email."}, status_code=201)
    except Exception as e:
        logger.exception(f"Failed to send signup email to {email}: {e}")
        # The transaction will rollback automatically on exception
        raise HTTPException(500, "Could not send verification email.")

async def login_endpoint(request: Request):
    body = await request.json()
    email, password = body.get("email"), body.get("password")
    
    query = users.select().where(users.c.email == email)
    user = await database.fetch_one(query)
    
    if not user or not verify_password(password, user['password_hash']):
        raise HTTPException(401, "Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    if not user['is_email_verified']: raise HTTPException(403, "Email not verified")
    
    access_token = create_access_token(data={"sub": user['email'], "userId": user['userId']}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    logger.info(f"User {email} logged in successfully.")
    return JSONResponse({"access_token": access_token, "token_type": "bearer"})

async def verify_email_endpoint(request: Request):
    body = await request.json()
    email, code = body.get("email"), body.get("code")
    
    user_query = users.select().where(users.c.email == email)
    user = await database.fetch_one(user_query)
    
    code_query = verification_codes.select().where(verification_codes.c.identifier == email)
    stored_info = await database.fetch_one(code_query)

    if not stored_info or not user: raise HTTPException(404, "Verification request not found.")
    if time.time() - stored_info["timestamp"] > 300: raise HTTPException(400, "Verification code has expired.")
    if stored_info["code"] != code: raise HTTPException(400, "Invalid verification code.")
    
    async with database.transaction():
        await database.execute(verification_codes.delete().where(verification_codes.c.identifier == email))
        await database.execute(users.update().where(users.c.email == email).values(is_email_verified=True))

    logger.info(f"Email verified for {email}")
    return JSONResponse({"message": "Email verification successful", "userId": user["userId"], "email": user["email"], "name": user["name"], "needsPhoneVerification": bool(user.get("phone")), "phone": user.get("phone")})

async def send_phone_verification_endpoint(request: Request):
    body = await request.json()
    phone = body.get("phone")
    if not phone:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Phone number is required.")
    
    code = generate_code()
    
    # Clean up old codes for this number
    await database.execute(verification_codes.delete().where(verification_codes.c.identifier == phone))
    
    insert_query = verification_codes.insert().values(identifier=phone, code=code, timestamp=time.time())

    try:
        await database.execute(insert_query)
        await send_sms_with_brevo(
            phone_number=phone, 
            body=f"Your Voxaroid verification code is: {code}"
        )
        return JSONResponse({"message": "Verification code sent to your phone."})
    except HTTPException as e:
        await database.execute(verification_codes.delete().where(verification_codes.c.identifier == phone))
        raise e

async def verify_phone_endpoint(request: Request):
    body = await request.json()
    phone, code = body.get("phone"), body.get("code")
    if not all([phone, code]):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Phone and code are required.")

    code_query = verification_codes.select().where(verification_codes.c.identifier == phone)
    stored_info = await database.fetch_one(code_query)

    if not stored_info: 
        raise HTTPException(HTTPStatus.NOT_FOUND, "Verification request not found. Please request a new code.")
    
    user_query = users.select().where(users.c.phone == phone)
    user = await database.fetch_one(user_query)
            
    if not user:
        raise HTTPException(HTTPStatus.NOT_FOUND, "User with this phone number not found.")

    if time.time() - stored_info["timestamp"] > 300: # 5 minutes
        await database.execute(verification_codes.delete().where(verification_codes.c.identifier == phone))
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Verification code has expired. Please request a new one.")
    if stored_info["code"] != code: 
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Invalid verification code.")

    async with database.transaction():
        await database.execute(verification_codes.delete().where(verification_codes.c.identifier == phone))
        await database.execute(users.update().where(users.c.phone == phone).values(is_phone_verified=True))
    
    logger.info(f"Phone number {phone} verified for user {user['email']}")
    
    access_token = create_access_token(
        data={"sub": user["email"], "userId": user["userId"]}, 
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return JSONResponse({
        "message": "Phone verification successful.", 
        "access_token": access_token, 
        "token_type": "bearer",
        "email": user["email"],
        "name": user["name"],
        "userId": user["userId"],
    })

async def resend_email_code_endpoint(request: Request):
    body = await request.json()
    email = body.get("email")
    if not email:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Email is required.")
    
    user_query = users.select().where(users.c.email == email)
    if not await database.fetch_one(user_query):
        raise HTTPException(HTTPStatus.NOT_FOUND, "User with this email not found.")

    code = generate_code()
    await database.execute(verification_codes.delete().where(verification_codes.c.identifier == email))
    await database.execute(verification_codes.insert().values(identifier=email, code=code, timestamp=time.time()))
    
    try:
        await send_email_with_brevo(
            email=email,
            subject=f"Your New Voxaroid Verification Code: {code}",
            html_content=f"Your new verification code is: <strong>{code}</strong>. It will expire in 5 minutes."
        )
        logger.info(f"Resent verification code to {email}")
        return JSONResponse({"message": "A new verification code has been sent to your email."})
    except Exception as e:
        logger.exception(f"Failed to resend verification email to {email}: {e}")
        raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not send verification email.")

async def resend_phone_code_endpoint(request: Request):
    body = await request.json()
    phone = body.get("phone")
    if not phone:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Phone number is required.")

    user_query = users.select().where(users.c.phone == phone)
    if not await database.fetch_one(user_query):
        raise HTTPException(HTTPStatus.NOT_FOUND, "User with this phone number not found.")
        
    code = generate_code()
    await database.execute(verification_codes.delete().where(verification_codes.c.identifier == phone))
    await database.execute(verification_codes.insert().values(identifier=phone, code=code, timestamp=time.time()))

    try:
        await send_sms_with_brevo(
            phone_number=phone,
            body=f"Your new Voxaroid verification code is: {code}"
        )
        logger.info(f"Resent verification code to {phone}")
        return JSONResponse({"message": "A new verification code has been sent to your phone."})
    except HTTPException as e:
        await database.execute(verification_codes.delete().where(verification_codes.c.identifier == phone))
        raise e

async def google_auth_endpoint(request: Request):
    if not GOOGLE_CLIENT_ID: raise HTTPException(500, "Google Sign-In is not configured.")
    body = await request.json()
    token = body.get("credential")
    try:
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = idinfo['email']
        name = idinfo.get('name', 'Google User')
        
        user_query = users.select().where(users.c.email == email)
        user = await database.fetch_one(user_query)

        if not user:
            user_id = f"user_{int(time.time())}"
            insert_query = users.insert().values(
                userId=user_id, name=name, email=email, password_hash=None,
                phone=None, is_email_verified=True, is_phone_verified=True
            )
            await database.execute(insert_query)
            user = await database.fetch_one(user_query)
            logger.info(f"New user created via Google Sign-In: {email}")

        access_token = create_access_token(data={"sub": user["email"], "userId": user["userId"]}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        logger.info(f"User {email} logged in via Google.")
        return JSONResponse({"access_token": access_token, "token_type": "bearer", "email": user['email'], "name": user['name'], "userId": user['userId']})
        
    except ValueError:
        raise HTTPException(401, "Invalid Google token.")


async def apple_auth_endpoint(request: Request):
    if not APPLE_CLIENT_ID:
        raise HTTPException(500, "Apple Sign-In is not configured on the server.")

    body = await request.json()
    identity_token = body.get('authorization', {}).get('id_token')
    if not identity_token:
        raise HTTPException(400, "Apple identity token not provided.")

    try:
        unverified_header = jwt.get_unverified_header(identity_token)
        key_id = unverified_header['kid']

        async with httpx.AsyncClient() as client:
            apple_keys_url = "https://appleid.apple.com/auth/keys"
            response = await client.get(apple_keys_url)
            response.raise_for_status()
            apple_keys_data = response.json()
            
        jwk_key = next((key for key in apple_keys_data['keys'] if key['kid'] == key_id), None)
        if not jwk_key:
            raise HTTPException(401, "Public key from Apple not found for the given token.")

        claims = jwt.decode(
            identity_token,
            jwk_key,
            algorithms=['RS256'],
            issuer='https://appleid.apple.com',
            audience=APPLE_CLIENT_ID,
        )
        email = claims['email']
        
        user_query = users.select().where(users.c.email == email)
        user = await database.fetch_one(user_query)

        if not user:
            user_id = f"user_{int(time.time())}"
            insert_query = users.insert().values(
                userId=user_id, name=f"User {user_id}", email=email, password_hash=None,
                phone=None, is_email_verified=True, is_phone_verified=True
            )
            await database.execute(insert_query)
            user = await database.fetch_one(user_query)
            logger.info(f"New user created via Apple Sign-In: {email}")

        access_token = create_access_token(
            data={"sub": user["email"], "userId": user["userId"]},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        logger.info(f"User {email} logged in via Apple.")
        return JSONResponse({"access_token": access_token, "token_type": "bearer", "email": user['email'], "name": user['name'], "userId": user['userId']})

    except httpx.RequestError as e:
        logger.exception("Could not connect to Apple's servers to verify token.")
        raise HTTPException(503, "Service unavailable. Could not verify Apple token.")
    except JWTError as e:
        logger.warning(f"Invalid Apple token: {e}")
        raise HTTPException(401, f"Invalid or expired Apple token.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during Apple authentication: {e}")
        raise HTTPException(500, "An internal server error occurred.")


# --- Subscription & Payment Endpoints ---
async def create_checkout_session_endpoint(request: Request, current_user: dict = Depends(get_current_user)):
    if not STRIPE_API_KEY: 
        raise HTTPException(500, "Payment system not configured.")
    try:
        data = await request.json()
        price_id = data.get('priceId')
        
        user_id = current_user['userId']
        customer_id = current_user.get("stripe_customer_id")

        if not customer_id:
             customer = stripe.Customer.create(email=current_user['email'], name=current_user['name'])
             customer_id = customer.id
             update_query = users.update().where(users.c.userId == user_id).values(stripe_customer_id=customer_id)
             await database.execute(update_query)

        checkout_session = stripe.checkout.Session.create(
            customer=customer_id,
            line_items=[{'price': price_id, 'quantity': 1}],
            mode='subscription',
            success_url=f"{FRONTEND_DOMAIN}/payment-success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_DOMAIN}/payment-cancel",
            metadata={'user_id': user_id}
        )
        return JSONResponse({'sessionId': checkout_session.id})
    except Exception as e:
        logger.exception(f"Stripe session creation failed: {e}")
        raise HTTPException(500, str(e))

async def stripe_webhook_endpoint(request: Request):
    if not STRIPE_WEBHOOK_SECRET: logger.error("Stripe webhook secret not configured."); return Response(500)
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        logger.warning(f"Webhook signature verification failed: {e}"); return Response(400)

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_id = session.get('metadata', {}).get('user_id')
        stripe_customer_id = session.get('customer')
        
        user_query = users.select().where(users.c.userId == user_id)
        user = await database.fetch_one(user_query)
        if user:
            line_item = session['display_items'][0]
            plan_name = line_item['plan']['nickname']
            update_query = users.update().where(users.c.userId == user_id).values(
                subscription_plan=plan_name.lower().replace(" ", "_"),
                stripe_customer_id=stripe_customer_id
            )
            await database.execute(update_query)
            logger.info(f"User {user_id} subscribed to {plan_name}. Stripe customer: {stripe_customer_id}")

    if event['type'] in ['customer.subscription.deleted', 'customer.subscription.updated']:
        subscription = event['data']['object']
        stripe_customer_id = subscription.get('customer')
        
        user_query = users.select().where(users.c.stripe_customer_id == stripe_customer_id)
        user = await database.fetch_one(user_query)
        
        if user:
            if subscription.get('cancel_at_period_end'):
                logger.info(f"Subscription for user {user['userId']} will be cancelled at period end.")
            else:
                update_query = users.update().where(users.c.userId == user['userId']).values(subscription_plan='free')
                await database.execute(update_query)
                logger.info(f"Subscription for user {user['userId']} cancelled. Downgraded to free.")

    return Response(status_code=200)

# --- Legacy & File Analysis API Endpoints ---
async def legacy_api_generate_endpoint(request):
    try:
        body = await request.json()
        query, user_id = body.get("query"), body.get("userId")
        if not query or user_id is None: return JSONResponse({"error": "Query and userId are required."}, 400)
        
        user_check_query = users.select().where(users.c.userId == user_id)
        if not await database.fetch_one(user_check_query):
            return JSONResponse({"error": "User not found."}, 404)

        handler_instance = TradingAssistantNLPHandler(user_id=user_id)
        await handler_instance.initialize()
        response_text, _ = await handler_instance.handle_query(query)
        return JSONResponse({"response": response_text})
    except json.JSONDecodeError: return JSONResponse({"error": "Invalid JSON body."}, 400)
    except Exception as e: logger.exception(f"HTTP API error: {e}"); return JSONResponse({"error": "Internal Server Error."}, 500)

async def analyze_file_endpoint(request: Request, current_user: dict = Depends(get_current_user)):
    form = await request.form()
    file_upload: Optional[UploadFile] = form.get("file")
    prompt: Optional[str] = form.get("prompt")
    
    user_id = current_user['userId']

    if not all([file_upload, prompt]):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="A 'file' and 'prompt' are required in the form data.")

    upload_dir = MEDIA_DIR / f"user_{user_id}" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    temp_filename = f"{int(time.time())}_{file_upload.filename}"
    saved_file_path = upload_dir / temp_filename
    
    try:
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file_upload.file, buffer)
        
        logger.info(f"File from user {user_id} saved temporarily to {saved_file_path}")

        query_for_handler = f'analyze file "{saved_file_path}" {prompt}'
        
        handler_instance = TradingAssistantNLPHandler(user_id=user_id)
        await handler_instance.initialize()
        response_text, _ = await handler_instance.handle_query(query_for_handler)
        
        return JSONResponse({"analysis": response_text})

    except Exception as e:
        logger.exception(f"Error during file analysis for user {user_id}: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="An internal error occurred during file analysis.")
    
    finally:
        if saved_file_path.exists():
            os.remove(saved_file_path)
            logger.info(f"Cleaned up uploaded file: {saved_file_path}")

# -------------------------------------------------------
# Application Setup
# -------------------------------------------------------
async def on_startup():
    await database.connect()
    # Optional: Create tables if they don't exist.
    # In a full production setup, you would use a migration tool like Alembic.
    engine = sqlalchemy.create_engine(DATABASE_URL)
    metadata.create_all(engine)
    logger.info("Database connected and tables verified.")


async def on_shutdown():
    await database.disconnect()
    logger.info("Database disconnected.")

routes = [
    WebSocketRoute("/ws", endpoint=websocket_handler_endpoint),
    # Auth
    Route("/api/auth/signup", endpoint=signup_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/auth/login", endpoint=login_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/auth/verify-email", endpoint=verify_email_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/auth/send-phone-verification", endpoint=send_phone_verification_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/auth/verify-phone", endpoint=verify_phone_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/auth/google", endpoint=google_auth_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/auth/apple", endpoint=apple_auth_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/auth/resend-email-code", endpoint=resend_email_code_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/auth/resend-phone-code", endpoint=resend_phone_code_endpoint, methods=["POST", "OPTIONS"]),
    # Subscriptions (Now Protected)
    Route("/api/create-checkout-session", endpoint=create_checkout_session_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/stripe-webhook", endpoint=stripe_webhook_endpoint, methods=["POST"]),
    # Core API (Now Protected)
    Route("/api/generate", endpoint=legacy_api_generate_endpoint, methods=["POST", "OPTIONS"]),
    Route("/api/analyze-file", endpoint=analyze_file_endpoint, methods=["POST", "OPTIONS"]),
    # Static files (must be last)
    Mount("/", app=StaticFiles(directory=str(PROJECT_ROOT), html=True), name="static"),
]
middleware = [Middleware(CORSMiddleware, allow_origins=[FRONTEND_DOMAIN], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])]
app = Starlette(
    debug=True, 
    routes=routes, 
    middleware=middleware,
    on_startup=[on_startup],
    on_shutdown=[on_shutdown]
)