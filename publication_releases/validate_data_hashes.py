from pathlib import Path
import hashlib

EXPECTED = {
    "data/processed.cleveland.data": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    "data/grid3_nga_-_health_facilities_-1.xlsx": "c438adbeec7577515a945b5dffebe6b97af52a4a291c5f110b9088feb36d60a2",
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

ok = True
for name, expected in EXPECTED.items():
    p = Path(name)
    if not p.exists():
        print(f"MISSING: {name}")
        ok = False
        continue
    got = sha256(p)
    status = "OK" if got == expected else "MISMATCH"
    print(f"{status}: {name}\n  expected: {expected}\n  got:      {got}")
    ok &= got == expected

raise SystemExit(0 if ok else 1)
