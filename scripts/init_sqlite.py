"""Create the local SQLite client-deliverables database with illustrative rows."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.config import DATA_DIR, SQLITE_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS client_deliverables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client TEXT NOT NULL,
    project TEXT NOT NULL,
    deliverable TEXT NOT NULL,
    workstream TEXT NOT NULL,
    owner TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('Not Started','In Progress','Blocked','In Review','Delivered')),
    priority TEXT NOT NULL CHECK(priority IN ('Critical','High','Medium','Low')),
    due_date TEXT,
    progress_pct INTEGER NOT NULL DEFAULT 0 CHECK(progress_pct BETWEEN 0 AND 100),
    blocker TEXT NOT NULL DEFAULT '',
    next_action TEXT NOT NULL DEFAULT '',
    last_update TEXT,
    is_sample INTEGER NOT NULL DEFAULT 1 CHECK(is_sample IN (0,1)),
    UNIQUE(client, project, deliverable)
)
"""


def initialize(path: Path = SQLITE_DB_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(SCHEMA)
        count = conn.execute("SELECT COUNT(*) FROM client_deliverables").fetchone()[0]
        if count == 0:
            seed_path = DATA_DIR / "client_deliverables.json"
            rows = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.exists() else []
            conn.executemany(
                """INSERT OR IGNORE INTO client_deliverables
                (client, project, deliverable, workstream, owner, status, priority, due_date,
                 progress_pct, blocker, next_action, last_update, is_sample)
                VALUES (:client,:project,:deliverable,:workstream,:owner,:status,:priority,:due_date,
                        :progress_pct,:blocker,:next_action,:last_update,1)""",
                rows,
            )
        conn.commit()
    finally:
        conn.close()
    return path


if __name__ == "__main__":
    print(initialize())
