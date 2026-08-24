import base64, time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from kalshi import auth


def _make_key(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    p = tmp_path / "k.pem"
    p.write_bytes(pem)
    return key, str(p)


def test_load_private_key(tmp_path):
    key, path = _make_key(tmp_path)
    loaded = auth.load_private_key(path)
    assert loaded.key_size == 2048


def test_ws_auth_headers_shape_and_signature(tmp_path):
    key, path = _make_key(tmp_path)
    loaded = auth.load_private_key(path)
    before = int(time.time() * 1000)
    h = auth.ws_auth_headers("mykey", loaded)
    after = int(time.time() * 1000)

    assert h["KALSHI-ACCESS-KEY"] == "mykey"
    ts = int(h["KALSHI-ACCESS-TIMESTAMP"])
    assert before <= ts <= after

    # Signature must verify against the public key for exactly this message.
    message = f"{ts}GET/trade-api/ws/v2".encode()
    sig = base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"])
    key.public_key().verify(
        sig, message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )  # raises InvalidSignature on failure
