from langchain.agents.middleware._redaction import (
    detect_email,
    detect_ip,
    detect_mac_address,
)

CASES = {
    "email": (detect_email, "alice@example.com"),
    "ip": (detect_ip, "192.168.1.100"),
    "mac_address": (detect_mac_address, "00:1A:2B:3C:4D:5E"),
}
NEIGHBORS = {
    "ascii": "contact ",
    "chinese": "联系 ",
    "japanese": "メール ",
    "korean": "이메일 ",
    "cyrillic": "почта ",
    "arabic": "بريد ",
    "accented": "café ",
}

print("Testing PII Detection with Non-ASCII Characters:\n")
for name, (detector, value) in CASES.items():
    for script, prefix in NEIGHBORS.items():
        result = bool(detector(prefix + value))
        status = "✅" if result else "❌"
        print(f"{name:<12} {script:<9} detected={result} {status}")
    print()
