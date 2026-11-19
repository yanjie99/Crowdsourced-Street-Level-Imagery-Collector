"""Run ledger: one SQLite file per project recording every step run (status, times, inputs, outputs, counts).

A step is 'done' when its last run finished with status ok; its input fingerprint (sizes and modification
times of its input files, plus the settings it reads) tells whether a rerun is needed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
create table if not exists runs (
  id integer primary key autoincrement,
  step text not null,
  started_utc text not null,
  finished_utc text,
  status text not null,            -- running | ok | failed
  fingerprint text,
  inputs text,                     -- json list of input paths
  outputs text,                    -- json list of output paths
  summary text,                    -- json dict of counts
  error text
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fingerprint(paths: list[str], settings: dict) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        if os.path.isdir(p):
            for root, _, files in sorted(os.walk(p)):
                for f in sorted(files):
                    st = os.stat(os.path.join(root, f))
                    h.update(f"{os.path.join(root, f)}|{st.st_size}|{int(st.st_mtime)}".encode())
        elif os.path.exists(p):
            st = os.stat(p)
            h.update(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode())
        else:
            h.update(f"{p}|missing".encode())
    h.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


class Ledger:
    def __init__(self, path: str):
        self.path = path
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=60)   # parallel lanes write too: wait for the lock
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def start(self, step: str, fp: str, inputs: list[str]) -> int:
        with self._db() as db:
            cur = db.execute("insert into runs (step, started_utc, status, fingerprint, inputs) values (?,?,?,?,?)",
                             (step, now(), "running", fp, json.dumps(inputs)))
            return cur.lastrowid

    def finish(self, run_id: int, outputs: list[str], summary: dict):
        with self._db() as db:
            db.execute("update runs set finished_utc=?, status='ok', outputs=?, summary=? where id=?",
                       (now(), json.dumps(outputs), json.dumps(summary, default=str), run_id))

    def fail(self, run_id: int, error: str):
        with self._db() as db:
            db.execute("update runs set finished_utc=?, status='failed', error=? where id=?", (now(), error[:4000], run_id))

    def last(self, step: str) -> dict | None:
        with self._db() as db:
            db.row_factory = sqlite3.Row
            r = db.execute("select * from runs where step=? order by id desc limit 1", (step,)).fetchone()
            return dict(r) if r else None

    def all_last(self) -> list[dict]:
        with self._db() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("select * from runs where id in (select max(id) from runs group by step) order by step").fetchall()
            return [dict(r) for r in rows]
