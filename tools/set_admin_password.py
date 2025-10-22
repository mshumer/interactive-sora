from __future__ import annotations

import argparse
import getpass
import sys

from argon2 import PasswordHasher

from database import init_db, session_scope
from models import AdminSecret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set or update the admin dashboard password.")
    parser.add_argument(
        "--password",
        help="Password to set. If omitted, you will be prompted securely.",
    )
    return parser.parse_args()


def prompt_password() -> str:
    for _ in range(3):
        first = getpass.getpass("New admin password: ")
        if not first:
            print("Password may not be empty.", file=sys.stderr)
            continue
        confirm = getpass.getpass("Confirm password: ")
        if first != confirm:
            print("Passwords do not match. Try again.", file=sys.stderr)
            continue
        return first
    raise SystemExit("Failed to confirm password after 3 attempts.")


def main() -> None:
    args = parse_args()
    password = args.password or prompt_password()

    init_db()

    hasher = PasswordHasher()
    password_hash = hasher.hash(password)

    with session_scope() as session:
        secret = session.get(AdminSecret, 1)
        if secret is None:
            secret = AdminSecret(id=1, password_hash=password_hash)
            session.add(secret)
        else:
            secret.password_hash = password_hash

    print("Admin password updated.")


if __name__ == "__main__":
    main()
