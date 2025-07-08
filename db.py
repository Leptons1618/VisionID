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
