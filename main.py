from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Date, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, ConfigDict
from datetime import date, timedelta
from typing import Dict, List, Optional

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ==================== 数据库配置 ====================
DATABASE_URL = "sqlite:///./kanban.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ==================== 数据模型设计 ====================
class Board(Base):
    """看板表：存储看板基本信息"""

    __tablename__ = "boards"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)  # 看板名称，如"边界感六维"
    dimensions = Column(JSON)  # 维度列表，如 ["维度1", "维度2", ...]


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


# ==================== Pydantic 模型（数据验证）====================
class BoardCreate(BaseModel):
    name: str
    dimensions: List[str]  # 维度名称列表


class BoardResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    dimensions: List[str]


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


# ==================== 初始化看板数据 ====================
def init_boards(db: Session):
    """初始化两个看板：边界感六维 和 2026找工作八维"""
    # 检查是否已存在
    if db.query(Board).count() > 0:
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
def get_boards(db: Session = Depends(get_db)):
    """获取所有看板"""
    init_boards(db)  # 确保看板已初始化
    return db.query(Board).all()


@app.get("/boards/{board_id}", response_model=BoardResponse)
def get_board(board_id: int, db: Session = Depends(get_db)):
    """获取单个看板信息"""
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")
    return board


@app.post("/records", response_model=DailyRecordResponse)
def create_record(record: DailyRecordCreate, db: Session = Depends(get_db)):
    """创建每日记录"""
    # 验证看板是否存在
    board = db.query(Board).filter(Board.id == record.board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")

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
    db: Session = Depends(get_db),
):
    """获取记录列表，支持按看板和日期筛选"""
    query = db.query(DailyRecord)

    if board_id:
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
def delete_record(record_id: int, db: Session = Depends(get_db)):
    """删除记录"""
    record = db.query(DailyRecord).filter(DailyRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    db.delete(record)
    db.commit()
    return {"status": "success"}


@app.get("/boards/{board_id}/weekly-stats", response_model=WeeklyStatsResponse)
def get_weekly_stats(
    board_id: int, week_start: Optional[date] = None, db: Session = Depends(get_db)
):
    """获取每周统计数据"""
    # 验证看板是否存在
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")

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
    db: Session = Depends(get_db),
):
    """获取某个维度的周平均值历史数据"""
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")

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
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
