"""
IIT Chatbot – FastAPI Backend  v1.2
====================================
Feedback is persisted to Neon (PostgreSQL) — survives every restart.
Falls back to local feedback_log.json if DATABASE_URL is not set.
"""

from __future__ import annotations

import datetime
import zoneinfo
import json
import os
import sys
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional


from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))
from backend.es_client import get_es
from backend.orchestrator import chat_turn
from contextlib import asynccontextmanager
import asyncio
import httpx

# ── optional psycopg2 ─────────────────────────────────────────────────────────
try:
    import psycopg2
    import psycopg2.extras
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

# ── env ───────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS: List[str] = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "30"))
DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# ── rate limit table ──────────────────────────────────────────────────────────
_rate_counters: Dict[str, List[float]] = defaultdict(list)

# ── local fallback file ───────────────────────────────────────────────────────
_FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "feedback_log.json")

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(keepalive())
    yield

async def keepalive():
    await asyncio.sleep(60)
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.get("https://iit-chatbot-api.onrender.com/api/health", timeout=10)
                print("[Keepalive] pinged /api/health")
        except Exception as e:
            print(f"[Keepalive] ping failed: {e}")
        await asyncio.sleep(840)
app = FastAPI(title="IIT Chatbot API", version="1.2.0", lifespan=lifespan)



app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    memory: Optional[Dict[str, Any]] = {}
    conversation_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: str
    vote: str               # "up" or "down"
    query: str              # the user's question
    bot_answer: str         # the bot response that was rated
    comment: Optional[str] = ""

# ─────────────────────────────────────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────────────────────────────────────

def _check_rate_limit(ip: str) -> None:
    import time
    now = time.time()
    _rate_counters[ip] = [t for t in _rate_counters[ip] if now - t < 3600]
    if len(_rate_counters[ip]) >= RATE_LIMIT_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached – max {RATE_LIMIT_PER_HOUR} messages per hour.",
        )
    _rate_counters[ip].append(now)

# ─────────────────────────────────────────────────────────────────────────────
# Neon (PostgreSQL) helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_pg_conn():
    if not _PG_AVAILABLE or not DATABASE_URL:
        return None
    try:
        return psycopg2.connect(DATABASE_URL, sslmode="require")
    except Exception as exc:
        print(f"[Neon] connection failed: {exc}")
        return None


def _write_to_neon(entry: Dict[str, Any]) -> bool:
    conn = _get_pg_conn()
    if conn is None:
        return False
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback
                    (timestamp, session_id, query_id, vote, query, bot_answer, feedback_text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                      (
                          entry["timestamp"],
                          entry["conversation_id"],
                          entry["message_id"],
                          entry["vote"],
                          entry["query"],
                          entry["bot_answer"],
                          entry["comment"],
                          ),
)
        return True
    except Exception as exc:
        print(f"[Neon] write failed: {exc}")
        return False
    finally:
        conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# Local JSON fallback (always written as backup)
# ─────────────────────────────────────────────────────────────────────────────

def _write_to_file(entry: Dict[str, Any]) -> None:
    try:
        existing: List[Dict] = []
        if os.path.exists(_FEEDBACK_FILE):
            with open(_FEEDBACK_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.append(entry)
        with open(_FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[File] feedback write failed: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    _check_rate_limit(ip)

    conversation_id = req.conversation_id or str(uuid.uuid4())
    message_id      = str(uuid.uuid4())
    result          = chat_turn(req.query, memory=req.memory)

    return {**result, "conversation_id": conversation_id, "message_id": message_id}

# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat/stream
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    _check_rate_limit(ip)

    conversation_id = req.conversation_id or str(uuid.uuid4())
    message_id      = str(uuid.uuid4())
    result          = chat_turn(req.query, memory=req.memory)
    answer          = result.get("answer_markdown", "")

    def event_stream():
        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id, 'message_id': message_id, 'mode': result.get('mode'), 'topic': result.get('topic'), 'memory': result.get('memory', {}), 'full_answer': answer})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ─────────────────────────────────────────────────────────────────────────────
# POST /api/feedback
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    if req.vote not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="vote must be 'up' or 'down'")

    entry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago")).isoformat(),
        "conversation_id": req.conversation_id,
        "message_id":      req.message_id,
        "vote":            req.vote,
        "query":           req.query,
        "bot_answer":      req.bot_answer,
        "comment":         req.comment or "",
    }

    # Always write local file first (instant backup)
    _write_to_file(entry)

    # Then write to Neon (primary persistent store)
    saved_to_db = _write_to_neon(entry)

    return {
        "status":        "ok",
        "saved_to_db":   saved_to_db,
        "saved_to_file": True,
        "entry":         entry,
    }

# ─────────────────────────────────────────────────────────────────────────────
# GET /api/feedback
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/feedback")
def get_feedback():
    """Read all feedback from Neon, fall back to local file."""
    conn = _get_pg_conn()
    if conn:
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT * FROM feedback ORDER BY timestamp DESC")
                    return cur.fetchall()
        except Exception as exc:
            print(f"[Neon] read failed: {exc}")
        finally:
            conn.close()
    if os.path.exists(_FEEDBACK_FILE):
        with open(_FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ─────────────────────────────────────────────────────────────────────────────
# GET /api/health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    es_ok = idx_ok = db_ok = False
    doc_count = 0

    try:
        es = get_es()
        if es.ping():
            es_ok    = True
            idx_name = os.getenv("ES_INDEX", "iit_policy_chunks")
            if es.indices.exists(index=idx_name):
                idx_ok    = True
                doc_count = es.count(index=idx_name).get("count", 0)
    except Exception:
        pass

    conn = _get_pg_conn()
    if conn:
        db_ok = True
        conn.close()

    return {
        "status":         "ok" if (es_ok and idx_ok and doc_count > 0 and db_ok) else "degraded",
        "elasticsearch":  "ok" if es_ok  else "unreachable",
        "index":          "ok" if idx_ok else "missing",
        "document_count": doc_count,
        "database":       "ok" if db_ok  else "not configured" if not DATABASE_URL else "unreachable",
        "timestamp":      datetime.datetime.utcnow().isoformat() + "Z",
    }

# ─────────────────────────────────────────────────────────────────────────────
# GET /
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "IIT International Student Chatbot API",
        "version": "1.2.0",
        "endpoints": [
            "POST /api/chat",
            "POST /api/chat/stream",
            "POST /api/feedback",
            "GET  /api/feedback",
            "GET  /api/health",
        ],
    }
