from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Dict, Any

router = APIRouter()


class UserRegister(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: str = None
    institution: str = None


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


@router.post("/register", response_model=Token)
async def register_user(user_data: UserRegister):
    """Register a new user"""
    # Mock implementation for demo
    return Token(
        access_token="mock_access_token_12345",
        user={
            "id": 1,
            "email": user_data.email,
            "username": user_data.username,
            "full_name": user_data.full_name,
            "institution": user_data.institution,
            "is_active": True
        }
    )


@router.post("/login", response_model=Token)
async def login_user(credentials: UserLogin):
    """Login user and return access token"""
    # Mock implementation for demo
    if credentials.username == "demo" and credentials.password == "demo":
        return Token(
            access_token="mock_access_token_demo_user",
            user={
                "id": 1,
                "email": "demo@example.com",
                "username": "demo",
                "full_name": "Demo User",
                "institution": "Demo University",
                "is_active": True
            }
        )
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")


@router.get("/me")
async def get_current_user():
    """Get current user information"""
    # Mock implementation
    return {
        "id": 1,
        "email": "demo@example.com", 
        "username": "demo",
        "full_name": "Demo User",
        "institution": "Demo University",
        "research_interests": ["AI", "Machine Learning", "Scientific Discovery"],
        "is_active": True
    } 