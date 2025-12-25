from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Date,
    JSON,
    Boolean,
    DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, ConfigDict
from datetime import date, datetime, timedelta
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


# ==================== Pydantic 模型（数据验证）====================
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
    """获取所有未删除的看板"""
    init_boards(db)  # 确保看板已初始化
    return db.query(Board).filter(Board.is_deleted.is_(False)).all()


@app.get("/boards/deleted/list", response_model=List[BoardResponse])
def get_deleted_boards(db: Session = Depends(get_db)):
    """获取所有已删除的看板"""
    return db.query(Board).filter(Board.is_deleted.is_(True)).all()


@app.get("/boards/{board_id}", response_model=BoardResponse)
def get_board(board_id: int, db: Session = Depends(get_db)):
    """获取单个看板信息"""
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")
    return board


@app.post("/boards", response_model=BoardResponse)
def create_board(board: BoardCreate, db: Session = Depends(get_db)):
    """创建新看板"""
    # 检查名称是否已存在
    existing = db.query(Board).filter(Board.name == board.name).first()
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

    db_board = Board(name=board.name, dimensions=board.dimensions, is_deleted=False)
    db.add(db_board)
    db.commit()
    db.refresh(db_board)
    return db_board


@app.delete("/boards/{board_id}")
def delete_board(board_id: int, db: Session = Depends(get_db)):
    """软删除看板"""
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")
    if board.is_deleted:
        raise HTTPException(status_code=400, detail="看板已被删除")

    board.is_deleted = True
    board.deleted_at = datetime.now()
    db.commit()
    return {"status": "success", "message": "看板已移至最近删除"}


@app.post("/boards/{board_id}/restore")
def restore_board(board_id: int, db: Session = Depends(get_db)):
    """恢复已删除的看板"""
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")
    if not board.is_deleted:
        raise HTTPException(status_code=400, detail="看板未被删除")

    board.is_deleted = False
    board.deleted_at = None
    db.commit()
    return {"status": "success", "message": "看板已恢复"}


@app.delete("/boards/{board_id}/permanent")
def permanent_delete_board(board_id: int, db: Session = Depends(get_db)):
    """永久删除看板及其所有记录"""
    board = db.query(Board).filter(Board.id == board_id).first()
    if not board:
        raise HTTPException(status_code=404, detail="看板不存在")

    # 删除该看板的所有记录
    db.query(DailyRecord).filter(DailyRecord.board_id == board_id).delete()
    # 删除看板
    db.delete(board)
    db.commit()
    return {"status": "success", "message": "看板已永久删除"}


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
    import subprocess
    import sys

    import uvicorn

    def is_port_in_use(port: int) -> int | None:
        """检查端口是否被占用，返回占用进程的 PID，未占用返回 None"""
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for line in result.stdout.splitlines():
                if f"127.0.0.1:{port}" in line and "LISTENING" in line:
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

    port = 8000
    pid = is_port_in_use(port)
    if pid:
        print(f"⚠️  端口 {port} 被进程 {pid} 占用，正在尝试释放...")
        if kill_process(pid):
            print(f"✅ 已终止进程 {pid}")
            import time

            time.sleep(0.5)  # 等待端口释放
        else:
            print(f"❌ 无法终止进程 {pid}，请手动处理")
            sys.exit(1)

    uvicorn.run(app, host="0.0.0.0", port=port)
