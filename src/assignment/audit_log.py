"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """TODO: store input + start timestamp keyed by request_id/user_id."""
        key = request_id or user_id
        self._open[key] = {
            "request_id": key,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "started_monotonic": time.monotonic(),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """TODO: store output, layer decision, latency; append to self.logs."""
        key = request_id or user_id
        opened = self._open.pop(key, None)
        if opened is None:
            # Keep the audit trail complete even if a caller omitted record_input.
            opened = {
                "request_id": request_id or key,
                "user_id": user_id,
                "input": "",
                "started_at": utc_now_iso(),
                "started_monotonic": time.monotonic(),
            }

        elapsed_ms = max(
            0.0, (time.monotonic() - opened.pop("started_monotonic")) * 1000
        )
        opened.update(
            {
                "output": text,
                "blocked": bool(blocked),
                "layer": layer,
                "finished_at": utc_now_iso(),
                "latency_ms": round(elapsed_ms, 3),
            }
        )
        self.logs.append(opened)
        return opened

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        # TODO: path = filepath or default_audit_log_path()
        #       ensure parent dirs exist, dump self.logs with indent=2
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
