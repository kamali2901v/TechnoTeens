"""
SQLite storage layer for AgroNex.
Replaces the flat audit_log.csv with structured, queryable storage.
Generates readable IDs (BATCH-0001, ON-00001) instead of hashes.
"""

import sqlite3
from datetime import datetime

DB_PATH = "agronex.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY,
            supplier_name TEXT,
            procurement_centre TEXT,
            date TEXT,
            onion_variety TEXT,
            quantity_received REAL,
            unit TEXT,
            intended_use TEXT,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            sample_id TEXT PRIMARY KEY,
            batch_id TEXT,
            timestamp TEXT,
            image_path TEXT,
            ai_prediction TEXT,
            ai_confidence REAL,
            grade TEXT,
            human_decision TEXT,
            human_agreed INTEGER,
            correction_reason TEXT,
            defect_tags TEXT,
            size_assessment TEXT DEFAULT 'Not Assessed',
            FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT,
            action TEXT,
            details TEXT,
            timestamp TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS officers (
            officer_id TEXT PRIMARY KEY,
            password_hash TEXT,
            name TEXT
        )
    """)

    migrations = [
        "ALTER TABLE batches ADD COLUMN status TEXT DEFAULT 'Pending'",
        "ALTER TABLE batches ADD COLUMN farmer_id TEXT",
        "ALTER TABLE samples ADD COLUMN mechanical_damage TEXT DEFAULT 'Not Assessed'",
        "ALTER TABLE samples ADD COLUMN final_grade TEXT",
        "ALTER TABLE samples ADD COLUMN grade_reason TEXT",
        "ALTER TABLE samples ADD COLUMN confidence_level TEXT",
    ]
    for sql in migrations:
        try:
            c.execute(sql)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()


def log_audit(batch_id, action, details=""):
    conn = get_connection()
    conn.execute(
        "INSERT INTO audit_logs (batch_id, action, details, timestamp) VALUES (?, ?, ?, ?)",
        (batch_id, action, details, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def _next_id(prefix, table, id_column):
    """Generates the next readable ID, e.g. BATCH-0001, ON-00001."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(f"SELECT {id_column} FROM {table} ORDER BY rowid DESC LIMIT 1")
    row = c.fetchone()
    conn.close()

    if row is None:
        next_num = 1
    else:
        last_id = row[0]
        last_num = int(last_id.split("-")[1])
        next_num = last_num + 1

    width = 4 if prefix == "BATCH" else 5
    return f"{prefix}-{str(next_num).zfill(width)}"


def create_batch(supplier_name, procurement_centre, onion_variety,
                  quantity_received, unit, intended_use):
    batch_id = _next_id("BATCH", "batches", "batch_id")
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO batches
        (batch_id, supplier_name, procurement_centre, date, onion_variety,
         quantity_received, unit, intended_use, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        batch_id, supplier_name, procurement_centre,
        datetime.now().strftime("%Y-%m-%d"), onion_variety,
        quantity_received, unit, intended_use, datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
    log_audit(batch_id, "BATCH_CREATED", f"Supplier: {supplier_name}, Variety: {onion_variety}")
    return batch_id


def create_sample(batch_id, image_path, ai_prediction, ai_confidence, grade, confidence_level=None):
    sample_id = _next_id("ON", "samples", "sample_id")
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO samples
        (sample_id, batch_id, timestamp, image_path, ai_prediction,
         ai_confidence, grade, human_decision, human_agreed,
         correction_reason, defect_tags, size_assessment, confidence_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 'Not Assessed', ?)
    """, (
        sample_id, batch_id, datetime.now().isoformat(), image_path,
        ai_prediction, ai_confidence, grade, confidence_level
    ))
    conn.commit()
    conn.close()
    log_audit(batch_id, "SAMPLE_ADDED", f"Sample {sample_id}: AI predicted {ai_prediction}")
    return sample_id


def record_human_decision(sample_id, human_decision, correction_reason=None, defect_tags=None):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT ai_prediction, batch_id FROM samples WHERE sample_id = ?", (sample_id,))
    row = c.fetchone()
    ai_prediction = row["ai_prediction"] if row else None
    batch_id = row["batch_id"] if row else None
    human_agreed = 1 if human_decision == ai_prediction else 0

    c.execute("""
        UPDATE samples
        SET human_decision = ?, human_agreed = ?, correction_reason = ?, defect_tags = ?
        WHERE sample_id = ?
    """, (human_decision, human_agreed, correction_reason, defect_tags, sample_id))
    conn.commit()
    conn.close()

    action = "HUMAN_CONFIRMED" if human_agreed else "HUMAN_CORRECTED"
    log_audit(batch_id, action, f"Sample {sample_id}: {ai_prediction} -> {human_decision}")


def record_size_assessment(sample_id, size_assessment):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT batch_id FROM samples WHERE sample_id = ?", (sample_id,))
    row = c.fetchone()
    batch_id = row["batch_id"] if row else None

    c.execute("UPDATE samples SET size_assessment = ? WHERE sample_id = ?",
              (size_assessment, sample_id))
    conn.commit()
    conn.close()
    log_audit(batch_id, "SIZE_ASSESSED", f"Sample {sample_id}: {size_assessment}")


def get_batch_summary(batch_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM samples WHERE batch_id = ?", (batch_id,))
    rows = c.fetchall()
    conn.close()

    total = len(rows)
    counts = {"healthy": 0, "damaged": 0, "rotten": 0}
    corrected = 0
    size_counts = {"Normal": 0, "Undersized": 0, "Not Assessed": 0}

    for r in rows:
        final_class = (r["human_decision"] or r["ai_prediction"] or "").lower()
        if final_class in counts:
            counts[final_class] += 1
        if r["human_agreed"] == 0:
            corrected += 1
        size = r["size_assessment"] or "Not Assessed"
        if size in size_counts:
            size_counts[size] += 1

    pct = {k: round(v / total * 100, 1) if total else 0 for k, v in counts.items()}

    return {
        "batch_id": batch_id,
        "total_samples": total,
        "healthy": counts["healthy"],
        "damaged": counts["damaged"],
        "rotten": counts["rotten"],
        "human_corrections": corrected,
        "healthy_pct": pct["healthy"],
        "damaged_pct": pct["damaged"],
        "rotten_pct": pct["rotten"],
        "size_normal": size_counts["Normal"],
        "size_undersized": size_counts["Undersized"],
        "size_not_assessed": size_counts["Not Assessed"],
    }


def get_all_batches():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM batches ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_batch_samples(batch_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM samples WHERE batch_id = ? ORDER BY timestamp", (batch_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]
def search_samples(query="", class_filter="All"):
    conn = get_connection()
    c = conn.cursor()
    sql = """
        SELECT s.*, b.supplier_name, b.procurement_centre
        FROM samples s
        LEFT JOIN batches b ON s.batch_id = b.batch_id
    """
    conditions = []
    params = []
    if query:
        like = f"%{query}%"
        conditions.append("(s.sample_id LIKE ? OR s.batch_id LIKE ? OR b.supplier_name LIKE ?)")
        params += [like, like, like]
    if class_filter and class_filter != "All":
        conditions.append("(COALESCE(s.human_decision, s.ai_prediction) = ?)")
        params.append(class_filter)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY s.timestamp DESC"
    c.execute(sql, params)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == "__main__":
    init_db()
    print("Database initialized: agronex.db")