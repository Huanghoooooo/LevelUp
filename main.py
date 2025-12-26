from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Date,
    JSON,
    Boolean,
    DateTime,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, ConfigDict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import bcrypt
import jwt
import secrets

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ==================== 配置 ====================
JWT_SECRET = secrets.token_hex(32)  # 生产环境应使用环境变量
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7天有效期

# ==================== 数据库配置 ====================
DATABASE_URL = "sqlite:///./kanban.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ==================== 数据模型设计 ====================
class User(Base):
    """用户表"""

    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    nickname = Column(String, default="用户")
    avatar = Column(Text, nullable=True)  # Base64 编码的头像或头像URL
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class Board(Base):
    """看板表：存储看板基本信息"""

    __tablename__ = "boards"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, index=True, nullable=True
    )  # 关联用户（可为空，兼容旧数据）
    name = Column(String, index=True, nullable=False)  # 看板名称，如"边界感六维"
    dimensions = Column(JSON)  # 维度列表，如 ["维度1", "维度2", ...]
    is_deleted = Column(Boolean, default=False)  # 软删除标记
    deleted_at = Column(DateTime, nullable=True)  # 删除时间


class DailyRecord(Base):
    """每日记录表：存储每天的维度数据"""

    __tablename__ = "daily_records"
    id = Column(Integer, primary_key=True, index=True)
    board_id = Column(Integer, index=True)  # 关联到看板
    record_date = Column(Date, index=True)  # 记录日期
    values = Column(JSON)  # 各维度的数值，格式：{"维度1": 5, "维度2": 3, ...}


# 创建数据库表
Base.metadata.create_all(bind=engine)

# ==================== FastAPI 应用初始化 ====================
app = FastAPI(title="多维度追踪看板")

# 允许跨域请求（开发时可能需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP Bearer 认证
security = HTTPBearer(auto_error=False)


# ==================== 密码和JWT工具函数 ====================
def hash_password(password: str) -> str:
    """对密码进行哈希"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """验证密码"""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_token(user_id: int) -> str:
    """创建JWT令牌"""
    now = datetime.now()
    payload = {
        "user_id": user_id,
        "exp": now + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": now,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """解码JWT令牌"""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ==================== Pydantic 模型（数据验证）====================
class UserCreate(BaseModel):
    email: str
    password: str
    nickname: Optional[str] = "用户"


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    nickname: str
    avatar: Optional[str] = None
    created_at: datetime


class UserUpdate(BaseModel):
    nickname: Optional[str] = None
    avatar: Optional[str] = None  # Base64 编码的头像


class PasswordChange(BaseModel):
    old_password: str
    new_password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class BoardCreate(BaseModel):
    name: str
    dimensions: List[str]  # 维度名称列表


class BoardResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    dimensions: List[str]
    is_deleted: Optional[bool] = False
    deleted_at: Optional[datetime] = None


class DailyRecordCreate(BaseModel):
    board_id: int
    record_date: date
    values: Dict[str, float]  # 维度名称 -> 数值


class DailyRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    board_id: int
    record_date: date
    values: Dict[str, float]


class WeeklyStatsResponse(BaseModel):
    """每周统计数据"""

    week_start: date
    week_end: date
    averages: Dict[str, float]  # 各维度的平均值
    last_week_averages: Optional[Dict[str, float]]  # 上周的平均值
    changes: Optional[Dict[str, float]]  # 与上周的变化（本周 - 上周）


# ==================== 数据库连接依赖 ====================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==================== 认证依赖 ====================
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """获取当前登录用户（可选认证）"""
    if credentials is None:
        return None

    token = credentials.credentials
    payload = decode_token(token)
    if payload is None:
        return None

    user = db.query(User).filter(User.id == payload["user_id"]).first()
    return user


async def require_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """要求必须登录"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == payload["user_id"]).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ==================== 认证 API ====================
@app.post("/auth/register", response_model=TokenResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """用户注册"""
    # 检查邮箱是否已存在
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="该邮箱已被注册")

    # 验证密码长度
    if len(user_data.password) < 6:
        raise HTTPException(status_code=400, detail="密码长度至少6位")

    # 创建用户
    user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        nickname=user_data.nickname or "用户",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # 为新用户创建默认看板
    init_boards_for_user(db, user.id)

    # 生成令牌
    token = create_token(user.id)
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@app.post("/auth/login", response_model=TokenResponse)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """用户登录"""
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")

    token = create_token(user.id)
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@app.get("/auth/me", response_model=UserResponse)
def get_me(user: User = Depends(require_user)):
    """获取当前用户信息"""
    return user


@app.put("/auth/me", response_model=UserResponse)
def update_me(
    update_data: UserUpdate,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """更新用户信息"""
    if update_data.nickname is not None:
        user.nickname = update_data.nickname
    if update_data.avatar is not None:
        user.avatar = update_data.avatar

    db.commit()
    db.refresh(user)
    return user


@app.put("/auth/password")
def change_password(
    password_data: PasswordChange,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """修改密码"""
    if not verify_password(password_data.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="原密码错误")

    if len(password_data.new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码长度至少6位")

    user.password_hash = hash_password(password_data.new_password)
    db.commit()
    return {"status": "success", "message": "密码修改成功"}


@app.put("/auth/email")
def change_email(
    new_email: str,
    password: str,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """修改邮箱"""
    if not verify_password(password, user.password_hash):
        raise HTTPException(status_code=400, detail="密码错误")

    existing = (
        db.query(User).filter(User.email == new_email, User.id != user.id).first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="该邮箱已被使用")

    user.email = new_email
    db.commit()
    return {"status": "success", "message": "邮箱修改成功"}


# ==================== 初始化看板数据 ====================
def init_boards_for_user(db: Session, user_id: int):
    """为新用户初始化默认看板"""
    default_boards = [
        {
            "name": "边界感六维",
            "dimensions": [
                "一次只专注一件事",
                "远离消耗你的人和事",
                "他人无需知道你做什么",
                "舍弃无意义的事情",
                "接纳自己,活在当下",
                "专注自身而非他人生活",
            ],
        },
        {
            "name": "2026找工作八维",
            "dimensions": [
                "Python深度应用能力",
                "Go语言后端开发能力",
                "LLM应用与提示词工程",
                "RAG系统开发能力",
                "Agent框架实战能力",
                "后端中间件与数据库技能",
                "计算机科学基础",
                "软件工程与落地规范",
            ],
        },
    ]

    for board_data in default_boards:
        # 检查该用户是否已有同名看板
        existing = (
            db.query(Board)
            .filter(Board.user_id == user_id, Board.name == board_data["name"])
            .first()
        )
        if not existing:
            board = Board(
                user_id=user_id,
                name=board_data["name"],
                dimensions=board_data["dimensions"],
            )
            db.add(board)

    db.commit()


def init_boards(db: Session):
    """初始化两个看板（兼容无用户的旧数据）"""
    # 检查是否已存在无用户的看板
    if db.query(Board).filter(Board.user_id.is_(None)).count() > 0:
        return

    # 创建"边界感六维"看板
    board1 = Board(
        name="边界感六维",
        dimensions=[
            "一次只专注一件事",
            "远离消耗你的人和事",
            "他人无需知道你做什么",
            "舍弃无意义的事情",
            "接纳自己,活在当下",
            "专注自身而非他人生活",
        ],
    )

    # 创建"2026找工作八维"看板
    board2 = Board(
        name="2026找工作八维",
        dimensions=[
            "Python深度应用能力",
            "Go语言后端开发能力",
            "LLM应用与提示词工程",
            "RAG系统开发能力",
            "Agent框架实战能力",
            "后端中间件与数据库技能",
            "计算机科学基础",
            "软件工程与落地规范",
        ],
    )

    db.add(board1)
    db.add(board2)
    db.commit()


# ==================== API 路由 ====================


@app.get("/boards", response_model=List[BoardResponse])
def get_boards(
    user: Optional[User] = Depends(get_current_user), db: Session = Depends(get_db)
):
    """获取当前用户的所有未删除看板"""
    if user:
        return (
            db.query(Board)
            .filter(Board.user_id == user.id, Board.is_deleted.is_(False))
            .all()
        )
    else:
        # 未登录用户，返回无归属的看板（兼容旧数据）
        init_boards(db)
        return (
            db.query(Board)
            .filter(Board.user_id.is_(None), Board.is_deleted.is_(False))
            .all()
        )


@app.get("/boards/deleted/list", response_model=List[BoardResponse])
def get_deleted_boards(
    user: Optional[User] = Depends(get_current_user), db: Session = Depends(get_db)
):
    """获取所有已删除的看板"""
    if user:
        return (
            db.query(Board)
            .filter(Board.user_id == user.id, Board.is_deleted.is_(True))
            .all()
        )
    else:
        return (
            db.query(Board)
            .filter(Board.user_id.is_(None), Board.is_deleted.is_(True))
            .all()
        )


@app.get("/boards/{board_id}", response_model=BoardResponse)
def get_board(
    board_id: int,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取单个看板信息"""
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")

    # 检查权限
    if user and board.user_id != user.id:
        raise HTTPException(status_code=403, detail="无权访问此看板")
    if not user and board.user_id is not None:
        raise HTTPException(status_code=403, detail="无权访问此看板")

    return board


@app.post("/boards", response_model=BoardResponse)
def create_board(
    board: BoardCreate,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """创建新看板"""
    user_id = user.id if user else None

    # 检查名称是否已存在
    existing = (
        db.query(Board)
        .filter(Board.name == board.name, Board.user_id == user_id)
        .first()
    )
    if existing:
        if existing.is_deleted:
            # 如果是已删除的同名看板，恢复并更新维度
            existing.is_deleted = False
            existing.deleted_at = None
            existing.dimensions = board.dimensions
            db.commit()
            db.refresh(existing)
            return existing
        raise HTTPException(status_code=400, detail="看板名称已存在")

    db_board = Board(
        user_id=user_id, name=board.name, dimensions=board.dimensions, is_deleted=False
    )
    db.add(db_board)
    db.commit()
    db.refresh(db_board)
    return db_board


@app.delete("/boards/{board_id}")
def delete_board(
    board_id: int,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """软删除看板"""
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")

    # 检查权限
    if user and board.user_id != user.id:
        raise HTTPException(status_code=403, detail="无权操作此看板")
    if not user and board.user_id is not None:
        raise HTTPException(status_code=403, detail="无权操作此看板")

    if board.is_deleted:
        raise HTTPException(status_code=400, detail="看板已被删除")

    board.is_deleted = True
    board.deleted_at = datetime.now()
    db.commit()
    return {"status": "success", "message": "看板已移至最近删除"}


@app.post("/boards/{board_id}/restore")
def restore_board(
    board_id: int,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """恢复已删除的看板"""
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")

    # 检查权限
    if user and board.user_id != user.id:
        raise HTTPException(status_code=403, detail="无权操作此看板")
    if not user and board.user_id is not None:
        raise HTTPException(status_code=403, detail="无权操作此看板")

    if not board.is_deleted:
        raise HTTPException(status_code=400, detail="看板未被删除")

    board.is_deleted = False
    board.deleted_at = None
    db.commit()
    return {"status": "success", "message": "看板已恢复"}


@app.delete("/boards/{board_id}/permanent")
def permanent_delete_board(
    board_id: int,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """永久删除看板及其所有记录"""
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")

    # 检查权限
    if user and board.user_id != user.id:
        raise HTTPException(status_code=403, detail="无权操作此看板")
    if not user and board.user_id is not None:
        raise HTTPException(status_code=403, detail="无权操作此看板")

    # 删除该看板的所有记录
    db.query(DailyRecord).filter(DailyRecord.board_id == board_id).delete()
    # 删除看板
    db.delete(board)
    db.commit()
    return {"status": "success", "message": "看板已永久删除"}


@app.post("/records", response_model=DailyRecordResponse)
def create_record(
    record: DailyRecordCreate,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """创建每日记录"""
    # 验证看板是否存在
    board = db.query(Board).filter(Board.id == record.board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")

    # 检查权限
    if user and board.user_id != user.id:
        raise HTTPException(status_code=403, detail="无权操作此看板")
    if not user and board.user_id is not None:
        raise HTTPException(status_code=403, detail="无权操作此看板")

    # 检查该日期是否已有记录
    existing = (
        db.query(DailyRecord)
        .filter(
            DailyRecord.board_id == record.board_id,
            DailyRecord.record_date == record.record_date,
        )
        .first()
    )

    if existing:
        # 更新已有记录
        existing.values = record.values
        db.commit()
        db.refresh(existing)
        return existing
    else:
        # 创建新记录
        db_record = DailyRecord(
            board_id=record.board_id,
            record_date=record.record_date,
            values=record.values,
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        return db_record


@app.get("/records", response_model=List[DailyRecordResponse])
def get_records(
    board_id: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取记录列表，支持按看板和日期筛选"""
    query = db.query(DailyRecord)

    if board_id:
        # 检查看板权限
        board = db.query(Board).filter(Board.id == board_id).first()
        if board:
            if user and board.user_id != user.id:
                raise HTTPException(status_code=403, detail="无权访问此看板")
            if not user and board.user_id is not None:
                raise HTTPException(status_code=403, detail="无权访问此看板")
        query = query.filter(DailyRecord.board_id == board_id)

    if start_date:
        query = query.filter(DailyRecord.record_date >= start_date)
    if end_date:
        query = query.filter(DailyRecord.record_date <= end_date)

    return query.order_by(DailyRecord.record_date.desc()).all()


@app.get("/records/{record_id}", response_model=DailyRecordResponse)
def get_record(record_id: int, db: Session = Depends(get_db)):
    """获取单条记录"""
    record = db.query(DailyRecord).filter(DailyRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    return record


@app.delete("/records/{record_id}")
def delete_record(
    record_id: int,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """删除记录"""
    record = db.query(DailyRecord).filter(DailyRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")

    # 检查权限
    board = db.query(Board).filter(Board.id == record.board_id).first()
    if board:
        if user and board.user_id != user.id:
            raise HTTPException(status_code=403, detail="无权操作此记录")
        if not user and board.user_id is not None:
            raise HTTPException(status_code=403, detail="无权操作此记录")

    db.delete(record)
    db.commit()
    return {"status": "success"}


@app.get("/boards/{board_id}/weekly-stats", response_model=WeeklyStatsResponse)
def get_weekly_stats(
    board_id: int,
    week_start: Optional[date] = None,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取每周统计数据"""
    # 验证看板是否存在
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")

    # 检查权限
    if user and board.user_id != user.id:
        raise HTTPException(status_code=403, detail="无权访问此看板")
    if not user and board.user_id is not None:
        raise HTTPException(status_code=403, detail="无权访问此看板")

    # 如果没有指定周开始日期，使用当前周
    if not week_start:
        today = date.today()
        # 计算本周一（ISO周从周一开始）
        week_start = today - timedelta(days=today.weekday())

    week_end = week_start + timedelta(days=6)

    # 获取本周的记录
    week_records = (
        db.query(DailyRecord)
        .filter(
            DailyRecord.board_id == board_id,
            DailyRecord.record_date >= week_start,
            DailyRecord.record_date <= week_end,
        )
        .all()
    )

    # 计算本周平均值
    if not week_records:
        averages = {dim: 0.0 for dim in board.dimensions}
    else:
        # 初始化累加器
        sums = {dim: 0.0 for dim in board.dimensions}
        counts = {dim: 0 for dim in board.dimensions}

        # 累加所有记录的值
        for record in week_records:
            for dim, value in record.values.items():
                if dim in sums:
                    sums[dim] += value
                    counts[dim] += 1

        # 计算平均值
        averages = {
            dim: sums[dim] / counts[dim] if counts[dim] > 0 else 0.0
            for dim in board.dimensions
        }

    # 获取上周的数据
    last_week_start = week_start - timedelta(days=7)
    last_week_end = week_end - timedelta(days=7)

    last_week_records = (
        db.query(DailyRecord)
        .filter(
            DailyRecord.board_id == board_id,
            DailyRecord.record_date >= last_week_start,
            DailyRecord.record_date <= last_week_end,
        )
        .all()
    )

    last_week_averages = None
    changes = None

    if last_week_records:
        # 计算上周平均值
        last_sums = {dim: 0.0 for dim in board.dimensions}
        last_counts = {dim: 0 for dim in board.dimensions}

        for record in last_week_records:
            for dim, value in record.values.items():
                if dim in last_sums:
                    last_sums[dim] += value
                    last_counts[dim] += 1

        last_week_averages = {
            dim: last_sums[dim] / last_counts[dim] if last_counts[dim] > 0 else 0.0
            for dim in board.dimensions
        }

        # 计算变化
        changes = {
            dim: averages[dim] - last_week_averages[dim] for dim in board.dimensions
        }

    return WeeklyStatsResponse(
        week_start=week_start,
        week_end=week_end,
        averages=averages,
        last_week_averages=last_week_averages,
        changes=changes,
    )


@app.get("/boards/{board_id}/dimension-history")
def get_dimension_history(
    board_id: int,
    dimension: str,
    weeks: int = 8,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取某个维度的周平均值历史数据"""
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")

    # 检查权限
    if user and board.user_id != user.id:
        raise HTTPException(status_code=403, detail="无权访问此看板")
    if not user and board.user_id is not None:
        raise HTTPException(status_code=403, detail="无权访问此看板")

    if dimension not in board.dimensions:
        raise HTTPException(status_code=400, detail="维度不存在")

    today = date.today()
    current_week_start = today - timedelta(days=today.weekday())

    history = []
    for i in range(weeks):
        week_start = current_week_start - timedelta(weeks=i)
        week_end = week_start + timedelta(days=6)

        records = (
            db.query(DailyRecord)
            .filter(
                DailyRecord.board_id == board_id,
                DailyRecord.record_date >= week_start,
                DailyRecord.record_date <= week_end,
            )
            .all()
        )

        if records:
            values = [
                r.values.get(dimension, 0) for r in records if dimension in r.values
            ]
            avg = sum(values) / len(values) if values else 0
        else:
            avg = 0

        history.append(
            {
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "week_label": f"{week_start.month}/{week_start.day}",
                "average": round(avg, 1),
            }
        )

    return {"dimension": dimension, "history": list(reversed(history))}


# ==================== 静态文件服务 ====================
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ==================== 启动服务器 ====================
if __name__ == "__main__":
    import os
    import platform
    import subprocess
    import sys

    import uvicorn

    # 从环境变量获取端口（Railway 会自动设置 PORT）
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    # 仅在 Windows 本地开发时检测端口占用
    if platform.system() == "Windows" and "RAILWAY_ENVIRONMENT" not in os.environ:

        def is_port_in_use(check_port: int) -> int | None:
            """检查端口是否被占用，返回占用进程的 PID，未占用返回 None"""
            try:
                result = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                for line in result.stdout.splitlines():
                    if f"127.0.0.1:{check_port}" in line and "LISTENING" in line:
                        parts = line.split()
                        return int(parts[-1])
            except Exception:
                pass
            return None

        def kill_process(pid: int) -> bool:
            """杀死指定 PID 的进程"""
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                return True
            except Exception:
                return False

        pid = is_port_in_use(port)
        if pid:
            print(f"⚠️  端口 {port} 被进程 {pid} 占用，正在尝试释放...")
            if kill_process(pid):
                print(f"✅ 已终止进程 {pid}")
                import time

                time.sleep(0.5)
            else:
                print(f"❌ 无法终止进程 {pid}，请手动处理")
                sys.exit(1)

    print(f"启动服务器: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
