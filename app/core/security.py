from datetime import datetime, timedelta
from typing import Optional
import hashlib
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.config import settings

# Password hashing - using SHA256 pre-hash to avoid bcrypt 72-byte limit
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _prehash_password(password: str) -> str:
    """
    Pre-hash password with SHA256 to avoid bcrypt's 72-byte limit
    SHA256 always produces 64-character hex output
    """
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def get_password_hash(password: str) -> str:
    """Hash a password using SHA256 + bcrypt"""
    prehashed = _prehash_password(password)
    return pwd_context.hash(prehashed)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    prehashed = _prehash_password(plain_password)
    return pwd_context.verify(prehashed, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Optional[dict]:
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None