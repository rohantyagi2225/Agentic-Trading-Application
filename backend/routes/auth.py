import secrets
from datetime import datetime

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.dependencies import get_current_user, get_db
from backend.config.settings import get_settings
from backend.db.models.user import User
from backend.security.auth import create_token, hash_password, verify_password
from backend.services.email_service import EmailService


router = APIRouter(tags=["Auth"])
email_service = EmailService()


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=8)
    display_name: str = Field(..., min_length=2)


class LoginRequest(BaseModel):
    email: str
    password: str


class VerifyEmailRequest(BaseModel):
    token: str


class ResendVerificationRequest(BaseModel):
    email: str


def _serialize_user(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "display_name": user.display_name,
        "created_at": user.created_at.isoformat(),
        "is_demo_user": user.is_demo_user,
        "is_active": user.is_active,
        "email_verified": user.email_verified,
        "demo_balance": user.demo_balance,
    }


def _send_verification(user: User) -> None:
    settings = get_settings()
    verification_url = f"{settings.APP_BASE_URL.rstrip('/')}/verify-email?token={user.email_verification_token}"
    email_service.send_verification_email(user.email, verification_url)


@router.post("/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email is already registered")

    base_name = payload.display_name.strip() or email.split("@")[0]
    username_base = "".join(ch.lower() for ch in base_name if ch.isalnum()) or "trader"
    username = username_base
    suffix = 1
    while db.query(User).filter(User.username == username).first():
        suffix += 1
        username = f"{username_base}{suffix}"

    user = User(
        email=email,
        username=username,
        display_name=payload.display_name.strip(),
        password_hash=hash_password(payload.password),
        is_demo_user=True,
        is_active=False,
        email_verified=False,
        email_verification_token=secrets.token_urlsafe(32),
        verification_sent_at=datetime.utcnow(),
        demo_balance=100000.0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    _send_verification(user)

    return {
        "status": "success",
        "data": {
            "message": "Account created. Verify your email before logging in.",
            "verification_required": True,
            "email": user.email,
        },
    }


@router.post("/verify-email")
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email_verification_token == payload.token).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")

    user.email_verified = True
    user.is_active = True
    user.email_verification_token = None
    db.commit()
    db.refresh(user)

    return {
        "status": "success",
        "data": {
            "message": "Email verified. You can now log in.",
            "user": _serialize_user(user),
        },
    }


@router.post("/resend-verification")
def resend_verification(payload: ResendVerificationRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.email_verified:
        raise HTTPException(status_code=400, detail="Email is already verified")

    user.email_verification_token = secrets.token_urlsafe(32)
    user.verification_sent_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    _send_verification(user)

    return {"status": "success", "data": {"message": "Verification email sent"}}


@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.email_verified:
        raise HTTPException(status_code=403, detail="Verify your email before logging in")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account is inactive")

    return {
        "token": create_token(user.email),
        "user": _serialize_user(user),
    }


@router.post("/logout")
def logout():
    return {"status": "success", "data": {"message": "Logged out"}}


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"status": "success", "data": _serialize_user(user)}
