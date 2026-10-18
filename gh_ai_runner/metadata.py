from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .types import InferenceResult, JobRecord

DEFAULT_METADATA_PATH = Path.home() / ".gh_ai_runner" / "jobs.sqlite3"


class MetadataStore:
    """Thread-safe SQLite metadata store. Prompts, outputs, and secrets are excluded."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        # Do not change permissions on an existing user-supplied directory.
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._initialize()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    correlation_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    batch_id TEXT,
                    owner TEXT NOT NULL,
                    repo_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    run_id INTEGER,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    output_sha256 TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    duration_seconds REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_job_id_idx ON jobs(job_id, attempt DESC)"
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "batch_id" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN batch_id TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_batch_id_idx ON jobs(batch_id, created_at)"
            )

    def upsert(self, record: JobRecord) -> None:
        fields = asdict(record)
        columns = list(fields)
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(
            f"{column}=excluded.{column}" for column in columns if column != "correlation_id"
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                f"INSERT INTO jobs ({','.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(correlation_id) DO UPDATE SET {updates}",
                [fields[column] for column in columns],
            )

    def get(self, job_id: str, attempt: Optional[int] = None) -> Optional[JobRecord]:
        query = "SELECT * FROM jobs WHERE job_id = ?"
        values = [job_id]
        if attempt is not None:
            query += " AND attempt = ?"
            values.append(attempt)
        query += " ORDER BY attempt DESC LIMIT 1"
        with self._lock, self._connect() as connection:
            row = connection.execute(query, values).fetchone()
        return JobRecord(**dict(row)) if row else None

    def list(self, limit: int = 100) -> list[JobRecord]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [JobRecord(**dict(row)) for row in rows]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def result_fields(result: Optional[InferenceResult]) -> dict:
    if result is None:
        return {}
    return {
        "output_sha256": result.integrity.output_sha256,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "total_tokens": result.usage.total_tokens,
        "duration_seconds": result.resources.duration_seconds,
    }
