import jwt
import datetime
import os
import secrets
import warnings
from hmac import compare_digest
from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import APIKeyCookie
from pydantic import BaseModel
from typing import Optional

load_dotenv()

# Security Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(32)
    warnings.warn(
        "JWT_SECRET_KEY is not configured; sessions will be invalidated on restart.",
        RuntimeWarning,
        stacklevel=2,
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

cookie_sec = APIKeyCookie(name="session_token", auto_error=False)

class LoginData(BaseModel):
    username: str
    password: str

# Credentials must be explicitly configured; an empty default prevents accidental
# deployment with a publicly known password.
VALID_USERNAME = os.getenv("ADMIN_USERNAME", "")
VALID_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

def create_access_token(data: dict[str, str]) -> str:
    to_encode = data.copy()
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": datetime.datetime.now(datetime.timezone.utc)})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user_optional(token: Optional[str] = Security(cookie_sec)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        subject = payload.get("sub")
        return subject if isinstance(subject, str) and subject else None
    except jwt.PyJWTError:
        return None

def get_current_user(token: Optional[str] = Security(cookie_sec)):
    username = get_current_user_optional(token)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return username


def credentials_match(username: str, password: str) -> bool:
    """Compare credentials without leaking timing information."""

    return compare_digest(username, VALID_USERNAME) and compare_digest(password, VALID_PASSWORD)
