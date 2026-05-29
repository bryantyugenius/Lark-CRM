from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./crm.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,       # 每次使用前检测连接是否存活
        pool_recycle=300,         # 5 分钟后强制回收连接（Neon 超时为 5min）
        pool_size=3,              # 少量连接（免费版够用）
        max_overflow=2,
    )

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Contact(Base):
    """客户/合作伙伴联系人"""
    __tablename__ = "contacts"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    company = Column(String(200))
    role = Column(String(100))
    type = Column(String(20), default="customer")
    phone = Column(String(50))
    email = Column(String(200))
    notes = Column(Text)
    custom_fields = Column(JSON)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Interaction(Base):
    """跟进记录"""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True)
    contact_id = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    raw_text = Column(Text)
    intent = Column(String(50))
    next_action = Column(Text)
    next_action_at = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100))


class Relationship(Base):
    """联系人之间的关系"""
    __tablename__ = "relationships"

    id = Column(Integer, primary_key=True)
    from_contact_id = Column(Integer, nullable=False)
    to_contact_id = Column(Integer, nullable=False)
    relation_type = Column(String(50))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
