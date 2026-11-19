"""Logging for step runs: plain lines to the console and to <project>/logs/<step>.log. Any registered secret
(the Mapillary token) is replaced by *** before a line is written anywhere."""
from __future__ import annotations

import logging
import os

_SECRETS: set[str] = set()


def register_secret(value: str | None):
    if value:
        _SECRETS.add(value)


def redact(text: str) -> str:
    for s in _SECRETS:
        text = text.replace(s, "***")
    return text


class _Redact(logging.Filter):
    def filter(self, record):
        record.msg = redact(str(record.msg))
        if record.args:
            record.args = tuple(redact(str(a)) for a in record.args)
        return True


def get_logger(step: str, log_dir: str | None = None) -> logging.Logger:
    lg = logging.getLogger(f"collect.{step}")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S")
    hs = [logging.StreamHandler()]
    if log_dir:
        hs.append(logging.FileHandler(os.path.join(log_dir, f"{step}.log"), encoding="utf8"))
    for h in hs:
        h.setFormatter(fmt)
        h.addFilter(_Redact())
        lg.addHandler(h)
    lg.propagate = False
    return lg
