import sqlite3
import numpy as np
import os
from config import DATABASE_PATH

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB,
            image_path TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS unknown_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB,
            image_path TEXT,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            seen_count INTEGER DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()

def insert_face(name, embedding, image_path):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO faces (name, embedding, image_path) VALUES (?, ?, ?)",
              (name, embedding.tobytes(), image_path))
    conn.commit()
    conn.close()

def get_all_faces():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM faces")
    rows = c.fetchall()
    conn.close()
    return [(name, np.frombuffer(embed, dtype=np.float32)) for name, embed in rows]


# Unknown face helpers
def insert_unknown(embedding, image_path):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO unknown_faces (embedding, image_path) VALUES (?, ?)",
        (embedding.tobytes(), image_path),
    )
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def get_all_unknowns():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT id, embedding, image_path, first_seen, last_seen, seen_count FROM unknown_faces ORDER BY last_seen DESC")
    rows = c.fetchall()
    conn.close()
    return [
        (
            row[0],
            np.frombuffer(row[1], dtype=np.float32),
            row[2],
            row[3],
            row[4],
            row[5],
        )
        for row in rows
    ]


def touch_unknown(unk_id):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE unknown_faces SET last_seen=CURRENT_TIMESTAMP, seen_count=seen_count+1 WHERE id=?",
        (unk_id,),
    )
    conn.commit()
    conn.close()


def delete_unknown(unk_id):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM unknown_faces WHERE id=?", (unk_id,))
    conn.commit()
    conn.close()


def promote_unknown(unk_id, name):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT embedding, image_path FROM unknown_faces WHERE id=?", (unk_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    embedding, image_path = row
    c.execute("INSERT INTO faces (name, embedding, image_path) VALUES (?, ?, ?)", (name, embedding, image_path))
    c.execute("DELETE FROM unknown_faces WHERE id=?", (unk_id,))
    conn.commit()
    conn.close()
    return True
