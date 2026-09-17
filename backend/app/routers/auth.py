"""회원가입 / 로그인 라우터."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models, api_schemas
from app.auth_guard import (
    hash_password,
    verify_password,
    create_token,
    get_current_user,
)

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


@router.get("/me", response_model=api_schemas.UserOut)
def get_me(current_user: db_models.User = Depends(get_current_user)):
    """현재 로그인한 사용자의 정보를 반환합니다."""
    return current_user


@router.put("/me", response_model=api_schemas.UserOut)
def update_me(
    req: api_schemas.UpdateUserRequest,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """현재 로그인한 사용자의 프로필(이름, 이메일, 연락처)을 수정합니다."""
    if req.email and req.email != current_user.email:
        existing = db.query(db_models.User).filter(
            db_models.User.email == req.email,
            db_models.User.id != current_user.id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="이미 사용 중인 이메일 주소입니다.")
        current_user.email = req.email

    if req.name is not None:
        current_user.name = req.name.strip()
    if req.phone is not None:
        current_user.phone = req.phone.strip()

    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/change-password")
def change_password(
    req: api_schemas.ChangePasswordRequest,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """현재 로그인한 사용자의 비밀번호를 변경합니다."""
    if not verify_password(req.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=400,
            detail="현재 비밀번호가 일치하지 않습니다.",
        )
    if len(req.new_password) < 4:
        raise HTTPException(
            status_code=400,
            detail="새 비밀번호는 최소 4자 이상이어야 합니다.",
        )
    current_user.password_hash = hash_password(req.new_password)
    db.commit()
    return {"message": "비밀번호가 성공적으로 변경되었습니다."}
