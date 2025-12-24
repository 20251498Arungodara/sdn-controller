import hashlib, json, requests, time, base64
from pathlib import Path
from ecdsa import SigningKey, NIST256p

# ================= CONFIG =================
PRIVATE_KEY_PATH = "controller_private.pem"
GATEWAY_URL = "http://localhost:3000"
CONTROLLER_ID = "ryu-controller-A"
# =========================================

def sha256_hex(data: bytes):
    return hashlib.sha256(data).hexdigest()

def load_or_create_key():
    if Path(PRIVATE_KEY_PATH).exists():
        return SigningKey.from_pem(Path(PRIVATE_KEY_PATH).read_bytes())
    sk = SigningKey.generate(curve=NIST256p)
    Path(PRIVATE_KEY_PATH).write_bytes(sk.to_pem())
    return sk

def main():
    sk = load_or_create_key()
    pubkey_pem = sk.verifying_key.to_pem().decode()

    # 1️⃣ Register controller (safe if already exists)
    r = requests.post(
        f"{GATEWAY_URL}/registerController",
        json={"controller_id": CONTROLLER_ID, "pubkey_pem": pubkey_pem}
    )
    print("controller:", r.text)

    # 2️⃣ Metrics
    metrics = {"avg_latency_ms": 12}
    metrics_bytes = json.dumps(metrics, sort_keys=True).encode()
    digest = sha256_hex(metrics_bytes)

    signature = base64.b64encode(
        sk.sign(bytes.fromhex(digest))
    ).decode()

    epoch = {
        "epoch_id": f"epoch_{int(time.time())}",
        "controller_id": CONTROLLER_ID,
        "digest": digest,
        "signature": signature,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    r = requests.post(f"{GATEWAY_URL}/submitEpoch", json=epoch)
    print("epoch:", r.text)

if __name__ == "__main__":
    main()
