"""
Simple officer authentication for AgroNex.
Passwords are hashed with SHA-256 before storage -- never stored in plain text.
"""

import hashlib
from db import get_connection


def _hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_officer(officer_id, password, name):
    """Creates a new officer account. Returns True on success, False if the ID already exists."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT officer_id FROM officers WHERE officer_id = ?", (officer_id,))
    if c.fetchone():
        conn.close()
        return False

    c.execute(
        "INSERT INTO officers (officer_id, password_hash, name) VALUES (?, ?, ?)",
        (officer_id, _hash_password(password), name)
    )
    conn.commit()
    conn.close()
    return True


def verify_login(officer_id, password):
    """Returns the officer's name if login is correct, otherwise None."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT password_hash, name FROM officers WHERE officer_id = ?", (officer_id,))
    row = c.fetchone()
    conn.close()

    if row is None:
        return None
    if row["password_hash"] == _hash_password(password):
        return row["name"]
    return None


if __name__ == "__main__":
    # Create one default test officer so you can log in immediately.
    success = register_officer("officer1", "password123", "Test Officer")
    if success:
        print("Default officer created: officer1 / password123")
    else:
        print("Officer 'officer1' already exists -- skipping.")