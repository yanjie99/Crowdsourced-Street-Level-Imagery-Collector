"""The Mapillary token: kept in Windows Credential Manager (generic credential "collect:mapillary"),
read only when a request needs it, sent only in the header Authorization: OAuth <token>, and registered
with the logger so it is masked if it ever reaches a log line. Never written to project files.

Fallback (other systems, or a token already set for the old scripts): the MAPILLARY_TOKEN environment variable.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

from .log import register_secret

TARGET = "collect:mapillary"
_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2


class _CREDENTIAL(ctypes.Structure):
    _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD), ("TargetName", wintypes.LPWSTR),
                ("Comment", wintypes.LPWSTR), ("LastWritten", wintypes.FILETIME),
                ("CredentialBlobSize", wintypes.DWORD), ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
                ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p),
                ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR)]


def _advapi():
    if sys.platform != "win32":
        return None
    a = ctypes.WinDLL("advapi32", use_last_error=True)
    a.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(_CREDENTIAL))]
    a.CredWriteW.argtypes = [ctypes.POINTER(_CREDENTIAL), wintypes.DWORD]
    a.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    a.CredFree.argtypes = [ctypes.c_void_p]
    return a


def store_token(token: str) -> None:
    a = _advapi()
    if a is None:
        raise RuntimeError("Credential Manager is Windows only; set MAPILLARY_TOKEN instead")
    blob = token.encode("utf-16-le")
    buf = (ctypes.c_byte * len(blob)).from_buffer_copy(blob)
    c = _CREDENTIAL(Type=_CRED_TYPE_GENERIC, TargetName=TARGET, CredentialBlobSize=len(blob),
                    CredentialBlob=ctypes.cast(buf, ctypes.POINTER(ctypes.c_byte)),
                    Persist=_CRED_PERSIST_LOCAL_MACHINE, UserName="mapillary")
    if not a.CredWriteW(ctypes.byref(c), 0):
        raise OSError(ctypes.get_last_error(), "CredWriteW failed")


def delete_token() -> bool:
    a = _advapi()
    return bool(a and a.CredDeleteW(TARGET, _CRED_TYPE_GENERIC, 0))


def get_token(required: bool = True) -> str | None:
    tok = None
    a = _advapi()
    if a is not None:
        p = ctypes.POINTER(_CREDENTIAL)()
        if a.CredReadW(TARGET, _CRED_TYPE_GENERIC, 0, ctypes.byref(p)):
            try:
                c = p.contents
                tok = ctypes.string_at(c.CredentialBlob, c.CredentialBlobSize).decode("utf-16-le")
            finally:
                a.CredFree(p)
    tok = tok or os.environ.get("MAPILLARY_TOKEN")
    if not tok and required:
        raise RuntimeError("No Mapillary token: run `collect token set` (or set MAPILLARY_TOKEN)")
    register_secret(tok)
    return tok


def token_source() -> str:
    a = _advapi()
    if a is not None:
        p = ctypes.POINTER(_CREDENTIAL)()
        if a.CredReadW(TARGET, _CRED_TYPE_GENERIC, 0, ctypes.byref(p)):
            a.CredFree(p)
            return "Windows Credential Manager"
    return "MAPILLARY_TOKEN environment variable" if os.environ.get("MAPILLARY_TOKEN") else "none"


def auth_header() -> dict:
    return {"Authorization": f"OAuth {get_token()}"}


def _graph() -> str:
    from .config import defaults
    return str(defaults().get("services", {}).get("mapillary_graph", "https://graph.mapillary.com")).rstrip("/")


def test_token(bbox) -> tuple[bool, str]:
    """One small Graph API search inside the area; returns (ok, message without the token)."""
    import requests
    b = ",".join(str(v) for v in bbox)
    try:
        r = requests.get(_graph() + "/images", params={"bbox": b, "limit": 1, "fields": "id"},
                         headers=auth_header(), timeout=30)
    except requests.RequestException as e:
        return False, f"request failed: {type(e).__name__}"
    if r.status_code == 200:
        return True, f"token works ({len(r.json().get('data', []))} image found in a 1-image test)"
    return False, f"HTTP {r.status_code}"
