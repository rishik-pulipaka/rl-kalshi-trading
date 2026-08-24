"""The load-bearing safety test: this project can never place a real order.

The credentials this system uses belong to a real, funded Kalshi account. The
agents trade simulated money against real prices (PRD 1, 2) and no code path may
ever reach Kalshi's trading endpoints.

That guarantee is only worth anything if it is mechanically enforced. A comment
saying "read-only" survives exactly until someone adds a convenient helper. So
this test reads the source of every module in the project and fails if it finds
either an HTTP verb that can mutate state or a reference to a trading endpoint.

If this test ever fails, do not weaken it. Either the offending code is a genuine
mistake, or the project's purpose has changed and that decision belongs to a
human, not to a passing test suite.
"""

import io
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories that hold our own source. Excludes tests/ (this file names the
# forbidden strings on purpose) and data/ (runtime state, not code).
SOURCE_DIRS = ("kalshi", "sim", "agent", "store", "dashboard", "tools")

# HTTP verbs that can change state on Kalshi's side. Matched as method calls on
# any object -- `requests.post(...)`, `session.put(...)`, `self._client.delete(...)`.
MUTATING_VERBS = re.compile(
    r"\.(post|put|patch|delete)\s*\(", re.IGNORECASE)

# Kalshi's trading surface. Reading market data never touches these.
TRADING_ENDPOINTS = re.compile(
    r"/portfolio/orders"
    r"|/portfolio/positions"
    r"|trade-api/v2/portfolio"
    r"|create_order"
    r"|place_order"
    r"|cancel_order"
    r"|batch_create_orders"
    r"|decrease_order",
    re.IGNORECASE)

# WebSocket commands that would act on the account rather than observe markets.
TRADING_WS_COMMANDS = re.compile(
    r"[\"']subscribe[\"']\s*:\s*[\"']fills[\"']"
    r"|[\"']fills[\"']"
    r"|[\"']market_positions[\"']",
    re.IGNORECASE)


def _source_files():
    for directory in SOURCE_DIRS:
        base = os.path.join(ROOT, directory)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                if name.endswith(".py"):
                    yield os.path.join(dirpath, name)
    for name in ("run.py",):
        path = os.path.join(ROOT, name)
        if os.path.exists(path):
            yield path


def _strip_noise(text):
    """Drop comments and docstrings so prose about safety isn't a violation.

    This file and several module docstrings legitimately discuss the forbidden
    endpoints; only executable code should be scanned.
    """
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    text = re.sub(r"'''(?:.|\n)*?'''", "", text)
    text = re.sub(r"#.*", "", text)
    return text


def _violations(pattern):
    found = []
    for path in _source_files():
        code = _strip_noise(io.open(path, encoding="utf-8").read())
        for match in pattern.finditer(code):
            line = code[:match.start()].count("\n") + 1
            found.append(f"{os.path.relpath(path, ROOT)}:{line}  {match.group(0)!r}")
    return found


def test_no_mutating_http_verbs():
    """No POST/PUT/PATCH/DELETE anywhere in the project's own source."""
    found = _violations(MUTATING_VERBS)
    assert not found, (
        "This project must only ever issue GET requests to Kalshi.\n"
        "Found state-changing HTTP calls:\n  " + "\n  ".join(found))


def test_no_trading_endpoints():
    """No reference to Kalshi's order or portfolio endpoints."""
    found = _violations(TRADING_ENDPOINTS)
    assert not found, (
        "This project must never reference Kalshi's trading surface.\n"
        "Found:\n  " + "\n  ".join(found))


def test_no_account_websocket_channels():
    """No subscription to private account channels (fills, positions)."""
    found = _violations(TRADING_WS_COMMANDS)
    assert not found, (
        "This project must only subscribe to public market-data channels.\n"
        "Found:\n  " + "\n  ".join(found))


def test_rest_client_exposes_only_get():
    """`kalshi.rest` must not grow a mutating helper.

    Checked by name as well as by verb, so a future `rest.submit()` that wraps a
    POST somewhere else still trips the alarm.
    """
    from kalshi import rest

    forbidden = ("post", "put", "patch", "delete", "order", "buy", "sell",
                 "cancel", "submit")
    offenders = [name for name in dir(rest)
                 if not name.startswith("__")
                 and any(word in name.lower() for word in forbidden)]
    assert not offenders, f"kalshi.rest exposes mutating-sounding names: {offenders}"


def test_private_key_is_not_inside_the_repo():
    """The .pem must live outside the project directory (PRD: public repo).

    The repo is public. Even with .gitignore in place, a key file sitting in the
    working tree is one `git add -f` or one archive-of-the-folder away from
    exposure. Keeping it out of the tree entirely removes that class of mistake.
    """
    strays = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for name in filenames:
            if name.endswith((".pem", ".key", ".p8")):
                strays.append(os.path.relpath(os.path.join(dirpath, name), ROOT))
    assert not strays, (
        "Private key material found inside the repository: " + ", ".join(strays) +
        "\nMove it outside the project and point KALSHI_PRIVATE_KEY_PATH at it.")


@pytest.mark.parametrize("name", [".env", ".env.local"])
def test_env_files_are_gitignored(name):
    """A committed .env would leak the key id on a public repo."""
    gitignore = os.path.join(ROOT, ".gitignore")
    assert os.path.exists(gitignore), "no .gitignore -- secrets are unprotected"
    patterns = io.open(gitignore, encoding="utf-8").read().split()
    assert ".env" in patterns, ".gitignore must exclude .env"
