"""Локальная база данных проекта (SQLite) на диске D.

Хранит камеры, координаты, историю алертов и опасные точки.
Путь БД по умолчанию: D:\\antiterror_data\\antiterror.db
(папка создаётся автоматически, если её нет).
"""
import os
import sqlite3
import time
from pathlib import Path

DB_PATH = r"D:\antiterror_data\antiterror.db"


def get_conn(path: str = DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str = DB_PATH) -> None:
    """Создаёт таблицы, если их нет."""
    conn = get_conn(path)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_index INT NOT NULL UNIQUE,  -- индекс камеры (0,1,2...)
            name TEXT,
            lat REAL,
            lon REAL
        );

        CREATE TABLE IF NOT EXISTS map_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            lat REAL,
            lon REAL,
            level TEXT DEFAULT 'danger',   -- danger | warning
            created_at REAL
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id INTEGER,                -- camera index (может быть NULL)
            cls_name TEXT,
            conf REAL,
            level TEXT,
            img_path TEXT,                 -- куда сохранён кадр алерта
            lat REAL,
            lon REAL,
            created_at REAL
        );
    """)
    conn.commit()
    conn.close()


def seed_cameras(config: dict, path: str = DB_PATH) -> None:
    """Заполняет таблицу cameras из config['map']['cameras'] (upsert)."""
    conn = get_conn(path)
    cur = conn.cursor()
    for c in config.get("map", {}).get("cameras", []):
        cur.execute("""
            INSERT INTO cameras (cam_index, name, lat, lon)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cam_index) DO UPDATE SET
                name=excluded.name, lat=excluded.lat, lon=excluded.lon
        """, (c.get("index", 0), c.get("name", ""), c.get("lat", 0.0), c.get("lon", 0.0)))
    conn.commit()
    conn.close()


def get_cameras(path: str = DB_PATH) -> list[dict]:
    conn = get_conn(path)
    rows = conn.execute("SELECT * FROM cameras ORDER BY cam_index").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_alert(cls_name: str, conf: float, level: str, img_path: str = "",
              cam_id: int | None = None, lat: float | None = None,
              lon: float | None = None, path: str = DB_PATH) -> int:
    """Сохраняет событие детекции. Возвращает id записи."""
    conn = get_conn(path)
    cur = conn.execute("""
        INSERT INTO alerts (cam_id, cls_name, conf, level, img_path, lat, lon, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (cam_id, cls_name, conf, level, img_path, lat, lon, time.time()))
    conn.commit()
    aid = cur.lastrowid
    conn.close()
    return aid


def add_map_point(lat: float, lon: float, label: str = "Опасное место",
                  level: str = "danger", path: str = DB_PATH) -> int:
    """Отмечает опасную точку для карты."""
    conn = get_conn(path)
    cur = conn.execute("""
        INSERT INTO map_points (label, lat, lon, level, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (label, lat, lon, level, time.time()))
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid


def get_danger_points(path: str = DB_PATH) -> list[dict]:
    """Все отмеченные опасные точки для отрисовки карты."""
    conn = get_conn(path)
    rows = conn.execute(
        "SELECT * FROM map_points ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_alerts(limit: int = 50, path: str = DB_PATH) -> list[dict]:
    conn = get_conn(path)
    rows = conn.execute(
        "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    seed_cameras(__import__("yaml").safe_load(
        open(r"C:\Users\MSI KATANA\Documents\Default Project\anti_terror\config.yaml",
             encoding="utf-8")))
    print("БД:", os.path.abspath(DB_PATH))
    print("Камеры:", get_cameras())