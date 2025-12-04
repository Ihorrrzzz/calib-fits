# bhtom_api.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, List

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BHTOM_BASE_URL = "https://bh-tom2.astrolabs.pl"
TOKEN_AUTH_ENDPOINT = "/api/token-auth/"
UPLOAD_URL = "https://uploadsvc2.astrolabs.pl/upload/"

# We'll store credentials (username + token) in a small JSON file
# next to this module. You can move this wherever you prefer.
CREDENTIALS_FILE = Path(__file__).with_name("bhtom_credentials.json")


# ---------------------------------------------------------------------------
# Low-level API
# ---------------------------------------------------------------------------

class BHTOMAuthError(Exception):
    """Raised when authentication with BHTOM fails."""


class BHTOMUploadError(Exception):
    """Raised when upload of calibrated frames fails."""


def get_auth_token(username: str, password: str) -> str:
    """
    Perform real login request to BHTOM and return the auth token.

    Raises BHTOMAuthError on bad credentials or HTTP errors.
    """
    url = BHTOM_BASE_URL + TOKEN_AUTH_ENDPOINT
    payload = {"username": username, "password": password}

    try:
        resp = requests.post(url, json=payload, timeout=15)
    except requests.RequestException as exc:
        raise BHTOMAuthError(f"Network error while contacting BHTOM: {exc}") from exc

    if resp.status_code != 200:
        # BHTOM returns 400 on invalid credentials, etc.
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise BHTOMAuthError(
            f"Login failed (HTTP {resp.status_code}). Response: {detail}"
        )

    try:
        data = resp.json()
    except Exception as exc:
        raise BHTOMAuthError(f"Unexpected response from BHTOM: {exc}") from exc

    token = data.get("token")
    if not token:
        raise BHTOMAuthError("Login succeeded but no token was returned.")

    return token


# ---------------------------------------------------------------------------
# Upload of calibrated frames
# ---------------------------------------------------------------------------

def upload_calibrated_files(
    files: List[Path],
    token: str,
    target: str,
    observatory: str,
    filter_name: str = "GaiaSP/any",
) -> None:
    """
    Upload a list of calibrated FITS files to BHTOM.

    This is a simplified uploader: it assumes the target already exists in BHTOM.
    Raises BHTOMUploadError on first failure.
    """
    if not files:
        return

    headers = {"Authorization": f"Token {token}"}

    for p in files:
        data = {
            "target": target,
            "filter": filter_name,
            "data_product_type": "fits_file",
            "dry_run": "False",
            "observatory": observatory,
        }

        try:
            with p.open("rb") as f:
                resp = requests.post(
                    UPLOAD_URL,
                    headers=headers,
                    data=data,
                    files={"files": f},
                    timeout=60,
                )
        except requests.RequestException as exc:
            raise BHTOMUploadError(
                f"Network error while uploading {p.name}: {exc}"
            ) from exc

        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text}

        if resp.status_code != 200:
            raise BHTOMUploadError(
                f"Upload failed for {p.name}: HTTP {resp.status_code}, response={payload}"
            )

        # If API encodes logical errors inside JSON, we can add extra checks here
        if isinstance(payload, dict) and "non_field_errors" in payload:
            raise BHTOMUploadError(
                f"Upload failed for {p.name}: {payload.get('non_field_errors')}"
            )


# ---------------------------------------------------------------------------
# Persistent "remember me" storage
# ---------------------------------------------------------------------------

def save_credentials(username: str, token: str) -> None:
    """
    Persist username + token to disk (for Remember me).
    There is deliberately **no** expiry – user stays logged in until logout.
    """
    CREDENTIALS_FILE.write_text(
        json.dumps({"username": username, "token": token}, indent=2),
        encoding="utf-8",
    )


def load_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Load username + token from disk. Returns (username, token) or (None, None).
    """
    if not CREDENTIALS_FILE.exists():
        return None, None

    try:
        data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    username = data.get("username")
    token = data.get("token")
    if not username or not token:
        return None, None
    return username, token


def clear_credentials() -> None:
    """Forget any saved BHTOM session."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()


def has_saved_session() -> bool:
    """Utility so the GUI can cheaply ask “do we have a session?”"""
    username, token = load_credentials()
    return bool(username and token)