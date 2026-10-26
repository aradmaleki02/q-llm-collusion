from contextlib import contextmanager
import io
import json
import math
from pathlib import Path
import sqlite3
import time

import torch


class Paused(Exception):
    """An actionable, resumable stop, not a failed experiment."""


@contextmanager
def single_writer(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".writer.lock").open("a+b")
    handle.seek(0)
    if not handle.read(1):
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if __import__("os").name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise Paused("Another command is using this runs directory. Close it before resuming.")
    try:
        yield
    finally:
        handle.close()  # OS releases locks even if a process dies.


class Store:
    def __init__(self, root="runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "experiment.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiments(
                name TEXT PRIMARY KEY, config TEXT NOT NULL, state BLOB NOT NULL,
                status TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS requests(
                id INTEGER PRIMARY KEY, experiment TEXT NOT NULL, position TEXT NOT NULL,
                attempt INTEGER NOT NULL, body TEXT NOT NULL, status TEXT NOT NULL,
                reserved REAL NOT NULL, charged REAL NOT NULL DEFAULT 0,
                response TEXT, error TEXT, retry_output_tokens INTEGER, created REAL NOT NULL,
                UNIQUE(experiment,position,attempt));
            CREATE TABLE IF NOT EXISTS rounds(
                experiment TEXT, task INTEGER, round INTEGER, variant TEXT, seed INTEGER,
                phase TEXT, episode INTEGER, record TEXT,
                PRIMARY KEY(experiment,task,round));
        """)
        self.db.execute("INSERT OR IGNORE INTO meta VALUES('budget','50')")
        self.db.commit()

    def close(self):
        self.db.close()

    def budget(self, amount=None):
        if amount is not None:
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError("Budget must be finite and positive")
            with self.db:
                self.db.execute("UPDATE meta SET value=? WHERE key='budget'", (str(amount),))
        return float(self.db.execute("SELECT value FROM meta WHERE key='budget'").fetchone()[0])

    def spending(self):
        billed, reserved = self.db.execute("""SELECT COALESCE(SUM(charged),0),
            COALESCE(SUM(CASE WHEN status IN ('pending','uncertain') THEN reserved ELSE 0 END),0)
            FROM requests""").fetchone()
        return {"budget": self.budget(), "charged": billed, "reserved": reserved,
                "available": self.budget() - billed - reserved}

    def save(self, name, config, state, status="running", row=None):
        buffer = io.BytesIO()
        torch.save(state, buffer)
        with self.db:
            self.db.execute("""INSERT INTO experiments VALUES(?,?,?,?,?) ON CONFLICT(name)
                DO UPDATE SET state=excluded.state,status=excluded.status,updated=excluded.updated""",
                (name, json.dumps(config.dictionary(), sort_keys=True), buffer.getvalue(), status, time.time()))
            if row:
                task_index, task, record = row
                self.db.execute("INSERT INTO rounds VALUES(?,?,?,?,?,?,?,?)",
                    (name, task_index, record["round"], task["variant"], task["seed"], task["phase"],
                     task["episode"], json.dumps(record)))

    def load(self, name):
        row = self.db.execute("SELECT * FROM experiments WHERE name=?", (name,)).fetchone()
        if not row:
            raise ValueError(f"No experiment named {name!r}")
        # Checkpoints contain our Python learner objects: load only your own files.
        state = torch.load(io.BytesIO(row["state"]), map_location="cpu", weights_only=False)
        return json.loads(row["config"]), state, row["status"]

    def exists(self, name):
        return self.db.execute("SELECT 1 FROM experiments WHERE name=?", (name,)).fetchone() is not None

    def reserve(self, name, position, body, reserve, spend_ceiling=None):
        with self.db:
            # Serialize budget check + reservation across same-round HTTP workers.
            self.db.execute("BEGIN IMMEDIATE")
            spending = self.spending()
            ceiling = min(spending['budget'], spend_ceiling) if spend_ceiling is not None else spending['budget']
            if reserve > ceiling - spending['charged'] - spending['reserved'] + 1e-12:
                raise Paused("Stage spending limit reached; progress is saved. Review results before extending it."
                             if ceiling < spending['budget'] else
                             "Shared allowance reached; progress is saved. Review usage before changing the budget.")
            attempt = self.db.execute("SELECT COALESCE(MAX(attempt),0)+1 FROM requests "
                                      "WHERE experiment=? AND position=?", (name, position)).fetchone()[0]
            cursor = self.db.execute("""INSERT INTO requests(experiment,position,attempt,body,status,reserved,created)
                VALUES(?,?,?,?,?,?,?)""", (name, position, attempt, json.dumps(body), "pending", reserve, time.time()))
        return cursor.lastrowid

    def request(self, name, position):
        return self.db.execute("SELECT * FROM requests WHERE experiment=? AND position=? ORDER BY attempt DESC LIMIT 1",
                               (name, position)).fetchone()

    def received(self, request_id, response, charged):
        with self.db:
            self.db.execute("UPDATE requests SET status='received',response=?,charged=? WHERE id=?",
                            (json.dumps(response), charged, request_id))

    def mark(self, request_id, status, error=""):
        with self.db:
            self.db.execute("UPDATE requests SET status=?,error=? WHERE id=?", (status, error, request_id))

    def recover_inflight(self, name):
        with self.db:
            self.db.execute("UPDATE requests SET status='uncertain',error='Process stopped during request' "
                            "WHERE experiment=? AND status='pending'", (name,))

    def retry_request(self, request_id, accept_duplicate, max_output_tokens=None):
        row = self.db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
        if not row or row["status"] not in ("uncertain", "invalid"):
            raise ValueError("Only uncertain or invalid requests can be released for retry")
        if row["status"] == "uncertain" and not accept_duplicate:
            raise ValueError("An uncertain call may already be billed. Supply --accept-possible-duplicate "
                             "only after checking the API dashboard. Its reservation becomes estimated spend.")
        if max_output_tokens is not None and max_output_tokens < 256:
            raise ValueError("max_output_tokens must be at least 256")
        with self.db:
            # Never make a possibly charged call disappear from the budget ledger.
            charged = max(row["charged"], row["reserved"]) if row["status"] == "uncertain" else row["charged"]
            self.db.execute("UPDATE requests SET status='retry',charged=?,retry_output_tokens=? WHERE id=?",
                            (charged, max_output_tokens, request_id))

    def records(self, name, task=None):
        query = "SELECT * FROM rounds WHERE experiment=?"
        args = [name]
        if task is not None:
            query += " AND task=?"
            args.append(task)
        return self.db.execute(query + " ORDER BY task,round", args).fetchall()

    def extend_baseline_horizon(self, name, rounds):
        """Audited extension of an unannounced cutoff, preserving paid responses.

        This exception to immutable configs is only for one continuing all-LLM
        market: no finite endpoint or trained network can be changed this way.
        Caller must hold single_writer; the config and amendment commit together.
        """
        from .config import Config
        values, state, status = self.load(name)
        old_rounds = values['rounds']
        if rounds == old_rounds:
            return
        if type(rounds) is not int or rounds <= old_rounds:
            raise ValueError('Only an increase of the unannounced round cap is supported')
        if values['gamma'] == 1 or state['learners'] or len(state['tasks']) != 1 or \
           state['tasks'][0]['variant'] != 'all_llm' or status.startswith('archived'):
            raise ValueError('Only a single continuing four-LLM baseline can be extended')
        values['rounds'] = rounds
        config = Config(**values).validate()
        state.setdefault('horizon_amendments', []).append(dict(
            previous=old_rounds, new=rounds, after_committed_round=state['committed_rounds'],
            timestamp=time.time(), reason='User requested 1000-round defaults; unannounced cap, unchanged objective.'))
        if state['market'] is None:
            history = [json.loads(row['record']) for row in self.records(name)]
            state['market'] = dict(history=history, strategies=history[-1]['strategies'] if history else ['']*4,
                                   pending=None)
            state['task_index'] = 0
        buffer = io.BytesIO()
        torch.save(state, buffer)
        with self.db:
            self.db.execute('UPDATE experiments SET config=?,state=?,status=?,updated=? WHERE name=?',
                (json.dumps(config.dictionary(), sort_keys=True), buffer.getvalue(),
                 'ready: extended unannounced horizon', time.time(), name))
