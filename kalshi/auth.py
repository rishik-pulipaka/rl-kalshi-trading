"""Kalshi WebSocket authentication: load the RSA private key and build the
signed auth headers. The WS signed message is `{timestamp_ms}GET/trade-api/ws/v2`
(no query string), signed with RSA-PSS / SHA256."""

import time
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key


def load_private_key(path):
    with open(path, "rb") as f:
        return load_pem_private_key(f.read(), password=None)


def ws_auth_headers(key_id, private_key):
    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}GET/trade-api/ws/v2"
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
    }
