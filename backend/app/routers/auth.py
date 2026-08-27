"""회원가입 / 로그인 라우터."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models, api_schemas
from app.auth_guard import hash_password, verify_password, create_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=api_schemas.UserOut)
def signup(req: api_schemas.SignupRequest, db: Session = Depends(get_db)):
    if db.query(db_models.User).filter(
            db_models.User.username == req.username).first():
        raise HTTPException(status_code=400, detail="이미 사용 중인 아이디입니다.")
    if req.email and db.query(db_models.User).filter(
            db_models.User.email == req.email).first():
        raise HTTPException(status_code=400, detail="이미 사용 중인 이메일입니다.")

    user = db_models.User(
        username=req.username,
        password_hash=hash_password(req.password),
        name=req.name,
        email=req.email,
        role="USER",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=api_schemas.LoginResponse)
def login(req: api_schemas.LoginRequest, db: Session = Depends(get_db)):
    user = db.query(db_models.User).filter(
        db_models.User.username == req.username).first()
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        )
    return api_schemas.LoginResponse(token=create_token(user), user=user)
