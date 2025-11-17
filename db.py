# db.py
"""
Lightweight SQL layer for AgriSphere using SQLAlchemy.

Features:
- SQLite by default (sqlite:///agrisphere.db)
- Models: Request, Recommendation, Price
- init_db() to create tables
- save_recommendation_record(...) to persist a recommendation event
- export_tables_to_csv(path) helper to dump tables (useful for Power BI Option A)

Usage:
    from db import init_db, save_recommendation_record, export_tables_to_csv
    init_db()
    rec_id = save_recommendation_record(session_info, top_results, price_info_list)
    export_tables_to_csv("exports")
"""

import os
import csv
from typing import List, Dict, Optional
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Use environment variable AGRI_DB_URL to override (e.g. postgres://user:pass@host/db)
DATABASE_URL = os.getenv("AGRI_DB_URL", "sqlite:///agrisphere.db")

# For SQLite we need check_same_thread=False for multi-threaded environments like Streamlit
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

# Create engine and session factory
# `future=True` gives more predictable SQLAlchemy 2.x style behavior while keeping compatibility
engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class Request(Base):
    __tablename__ = "requests"
    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, default=datetime.utcnow, nullable=False)

    # location / context
    user_lat = Column(Float, nullable=True)
    user_lon = Column(Float, nullable=True)
    state = Column(String(128), nullable=True)
    profile_used = Column(String(128), nullable=True)

    # input features
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    n = Column(Float, nullable=True)
    p = Column(Float, nullable=True)
    k = Column(Float, nullable=True)
    ph = Column(Float, nullable=True)
    rainfall = Column(Float, nullable=True)

    # relationships
    recommendations = relationship("Recommendation", back_populates="request", cascade="all, delete-orphan")
    prices = relationship("Price", back_populates="request", cascade="all, delete-orphan")


class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("requests.id", ondelete="CASCADE"))
    rank = Column(Integer, nullable=True)            # 1,2,3...
    crop = Column(String(128), nullable=True)
    suitability = Column(Float, nullable=True)       # model score only
    final_score = Column(Float, nullable=True)       # combined with price

    request = relationship("Request", back_populates="recommendations")


class Price(Base):
    __tablename__ = "prices"
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("requests.id", ondelete="CASCADE"))
    crop = Column(String(128), nullable=True)
    mandi_source = Column(String(64), nullable=True)    # 'agmarknet','napanta','commodityonline','manual'
    price_per_quintal = Column(Float, nullable=True)    # canonical numeric value in ₹/quintal
    raw_display = Column(String(256), nullable=True)    # raw string shown on UI

    request = relationship("Request", back_populates="prices")

# ------------------------------------------------------------------
# Init / helpers
# ------------------------------------------------------------------

def init_db() -> None:
    """
    Create database tables if they don't exist.
    Call this once at app startup (safe to call multiple times).
    """
    Base.metadata.create_all(bind=engine)


def _float_or_none(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def save_recommendation_record(
    session_info: Dict,
    top_results: List[Dict],
    price_info_list: List[Dict]
) -> int:
    """
    Persist a recommendation event (request + top results + price rows).

    session_info keys expected:
      lat, lon, state, profile, temperature, humidity, N, P, K, ph, rainfall

    top_results: list of dicts like:
      [{'rank':1,'crop':'rice','suitability':0.63,'final_score':0.55}, ...]

    price_info_list: list of dicts like:
      [{'crop':'rice','source':'agmarknet','price_q':1500.0,'raw':'₹1500 / Quintal'}, ...]

    Returns:
      request.id (int)

    Raises:
      Exception on DB error (caller should handle)
    """
    db = SessionLocal()
    try:
        req = Request(
            user_lat = _float_or_none(session_info.get("lat")),
            user_lon = _float_or_none(session_info.get("lon")),
            state = session_info.get("state"),
            profile_used = session_info.get("profile"),
            temperature = _float_or_none(session_info.get("temperature")),
            humidity = _float_or_none(session_info.get("humidity")),
            n = _float_or_none(session_info.get("N") or session_info.get("n")),
            p = _float_or_none(session_info.get("P") or session_info.get("p")),
            k = _float_or_none(session_info.get("K") or session_info.get("k")),
            ph = _float_or_none(session_info.get("ph")),
            rainfall = _float_or_none(session_info.get("rainfall"))
        )
        db.add(req)
        db.flush()  # assign req.id

        # save recommendations
        for r in top_results:
            rec = Recommendation(
                request_id = req.id,
                rank = int(r.get("rank")) if r.get("rank") is not None else None,
                crop = r.get("crop"),
                suitability = _float_or_none(r.get("suitability")),
                final_score = _float_or_none(r.get("final_score"))
            )
            db.add(rec)

        # save prices
        for p in price_info_list:
            price_q = p.get("price_q")
            try:
                price_q_f = float(price_q) if price_q is not None else None
            except Exception:
                price_q_f = None
            price_row = Price(
                request_id = req.id,
                crop = p.get("crop"),
                mandi_source = p.get("source"),
                price_per_quintal = price_q_f,
                raw_display = p.get("raw")
            )
            db.add(price_row)

        db.commit()
        return int(req.id)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ------------------------------------------------------------------
# CSV export helpers (useful to feed Power BI Option A)
# ------------------------------------------------------------------

def export_tables_to_csv(out_dir: str = "exports") -> None:
    """
    Export requests, recommendations, prices tables to CSV files in `out_dir`.
    Files: requests.csv, recommendations.csv, prices.csv

    Compatible with SQLAlchemy 2.x by using sqlalchemy.text(...)
    """
    os.makedirs(out_dir, exist_ok=True)
    db = SessionLocal()
    try:
        # Requests
        sql_reqs = text(
            "SELECT id, ts, user_lat, user_lon, state, profile_used, temperature, humidity, n, p, k, ph, rainfall "
            "FROM requests ORDER BY ts DESC"
        )
        res = db.execute(sql_reqs)
        rows = res.fetchall()
        req_path = os.path.join(out_dir, "requests.csv")
        with open(req_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id","ts","user_lat","user_lon","state","profile_used","temperature","humidity","n","p","k","ph","rainfall"])
            for r in rows:
                row_list = list(r)
                # format datetime to ISO if present
                if row_list[1] is not None:
                    try:
                        row_list[1] = row_list[1].isoformat()
                    except Exception:
                        row_list[1] = str(row_list[1])
                writer.writerow(row_list)

        # Recommendations
        sql_recs = text(
            "SELECT id, request_id, rank, crop, suitability, final_score FROM recommendations ORDER BY request_id, rank"
        )
        res2 = db.execute(sql_recs)
        rows2 = res2.fetchall()
        rec_path = os.path.join(out_dir, "recommendations.csv")
        with open(rec_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id","request_id","rank","crop","suitability","final_score"])
            for r in rows2:
                writer.writerow(list(r))

        # Prices
        sql_prices = text(
            "SELECT id, request_id, crop, mandi_source, price_per_quintal, raw_display FROM prices ORDER BY request_id"
        )
        res3 = db.execute(sql_prices)
        rows3 = res3.fetchall()
        price_path = os.path.join(out_dir, "prices.csv")
        with open(price_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id","request_id","crop","mandi_source","price_per_quintal","raw_display"])
            for r in rows3:
                writer.writerow(list(r))

    finally:
        db.close()


# Convenience: run export when module executed directly (for quick testing)
if __name__ == "__main__":
    print("Initializing DB (if needed) and exporting CSVs...")
    init_db()
    try:
        export_tables_to_csv()
        print("Exported CSVs into ./exports/")
    except Exception as e:
        print("Export failed:", e)
