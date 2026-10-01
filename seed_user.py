"""
<<<<<<< HEAD
DocuVortex — Seed User & API Key Generator
Usage:
    python seed_user.py "my_username"
    python seed_user.py "test_key"

This script creates or updates a user in the Supabase PostgreSQL database,
generates a secure `dk_live_...` API key, and prints it out so you can paste
it into the frontend "Enter API Key" popup.
"""

import hashlib
import os
import secrets
import sys
import uuid
from dotenv import load_dotenv

load_dotenv()

from src.database.db import get_db_cursor, init_db, test_connection


def seed_user(username_arg: str = "test_user"):
    username_clean = str(username_arg).strip()
    if not username_clean:
        username_clean = "test_user"

    print("=" * 65)
    print("DocuVortex — Seeding User & Generating API Key")
    print("=" * 65)

    # 1. Test database connection
    print("\n[1] Verifying connection to Supabase...")
    if not test_connection():
        print("❌ Cannot connect to Supabase PostgreSQL. Check DATABASE_URL in .env.")
        sys.exit(1)

    # 2. Ensure all tables (users, sessions, messages, attachments) are ready
    print("\n[2] Ensuring database schema & tables exist...")
    init_db("database/init.sql")

    # 3. Generate raw API key & its SHA-256 hash
    raw_api_key = f"dk_live_{secrets.token_hex(24)}"
    api_key_hash = hashlib.sha256(raw_api_key.encode()).hexdigest()
    email_clean = f"{username_clean.lower()}@docuvortex.local"

    print(f"\n[3] Upserting user '{username_clean}' into database...")
    user_id = None
    try:
        with get_db_cursor(commit=True) as cur:
            # Check if user already exists by username or email
            cur.execute(
                """
                SELECT id, username, email FROM users
                WHERE lower(username) = %s OR lower(email) = %s
                LIMIT 1;
                """,
                (username_clean.lower(), email_clean.lower()),
            )
            existing = cur.fetchone()

            if existing:
                user_id = str(existing["id"])
                cur.execute(
                    """
                    UPDATE users
                    SET api_key_hash = %s,
                        is_active = true,
                        is_verified = true
                    WHERE id = %s;
                    """,
                    (api_key_hash, user_id),
                )
                print(f"    • Updated existing user ID: {user_id}")
            else:
                user_id = str(uuid.uuid4())
                cur.execute(
                    """
                    INSERT INTO users (id, username, email, api_key_hash, name, is_verified, is_active)
                    VALUES (%s, %s, %s, %s, %s, true, true);
                    """,
                    (user_id, username_clean, email_clean, api_key_hash, username_clean),
                )
                print(f"    • Created new user ID: {user_id}")
    except Exception as e:
        print(f"❌ Failed to seed user: {e}")
        sys.exit(1)

    # 4. Display the generated API key clearly
    print("\n" + "=" * 65)
    print("✅ SUCCESS! User successfully created / updated.")
    print("=" * 65)
    print(f"\nUser Name : {username_clean}")
    print(f"User ID   : {user_id}")
    print(f"Email     : {email_clean}")
    print("\n" + "─" * 65)
    print("YOUR API KEY (copy and paste this into the browser popup):")
    print(f"\n    {raw_api_key}\n")
    print("─" * 65)
    print("\nInstructions:")
    print("1. Open http://localhost:8000 in your browser.")
    print("2. When prompted by the 'Enter API Key' popup, paste the key above.")
    print("3. Click 'Save Key'.")
    print("4. Your chat history, sessions, and uploaded documents will now persist")
    print("   permanently across browser refreshes and between new chats!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    target_username = sys.argv[1] if len(sys.argv) > 1 else "test_user"
    seed_user(target_username)
=======
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
>>>>>>> origin/main
