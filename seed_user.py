"""
Run this once per user to create their account and issue an API key.

Usage:
    python seed_user.py "Alice"

The plaintext key is printed ONCE. It is not recoverable afterward —
only its hash is stored. If lost, generate a new one (and revoke the old row).
"""
import hashlib
import secrets
import sys

from src.database.db import get_db_cursor


def create_user(name: str) -> str:
    plaintext_key = f"dk_live_{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()

    with get_db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (name, api_key_hash)
            VALUES (%s, %s)
            RETURNING id
            """,
            (name, key_hash),
        )
        user_id = cur.fetchone()["id"]

    return user_id, plaintext_key


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python seed_user.py \"User Name\"")
        sys.exit(1)

    name = sys.argv[1]
    user_id, plaintext_key = create_user(name)

    print(f"\nUser created: {name}")
    print(f"user_id: {user_id}")
    print(f"\nAPI key (save this now, it will not be shown again):")
    print(f"  {plaintext_key}\n")