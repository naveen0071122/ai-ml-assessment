"""
init_db.py
Creates a small dummy company database (employees, projects, issues) with
realistic FK relationships so multi-table / JOIN reasoning is possible.

Run:
    python init_db.py            # creates company.db in this folder
"""
import sqlite3
import os
from datetime import date, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "company.db")

SCHEMA = """
CREATE TABLE employees (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    department    TEXT NOT NULL,      -- e.g. 'AI', 'Platform', 'Design'
    title         TEXT NOT NULL,
    hire_date     DATE NOT NULL,
    email         TEXT UNIQUE NOT NULL
);

CREATE TABLE projects (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    department    TEXT NOT NULL,
    lead_id       INTEGER NOT NULL REFERENCES employees(id),
    status        TEXT NOT NULL CHECK (status IN ('active','on_hold','completed')),
    start_date    DATE NOT NULL
);

CREATE TABLE issues (
    id            INTEGER PRIMARY KEY,
    project_id    INTEGER NOT NULL REFERENCES projects(id),
    assignee_id   INTEGER NOT NULL REFERENCES employees(id),
    title         TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('open','in_progress','closed')),
    priority      TEXT NOT NULL CHECK (priority IN ('low','medium','high','critical')),
    created_date  DATE NOT NULL
);
"""

EMPLOYEES = [
    (1, "Ananya Rao",     "AI",       "ML Engineer",        "2022-03-01", "ananya.rao@corp.com"),
    (2, "Karthik Iyer",   "AI",       "Senior ML Engineer",  "2021-06-15", "karthik.iyer@corp.com"),
    (3, "Priya Menon",    "AI",       "Data Scientist",      "2023-01-10", "priya.menon@corp.com"),
    (4, "Rahul Verma",    "Platform", "Backend Engineer",    "2020-11-20", "rahul.verma@corp.com"),
    (5, "Sneha Pillai",   "Platform", "DevOps Engineer",     "2022-08-05", "sneha.pillai@corp.com"),
    (6, "Arjun Nair",     "Design",   "Product Designer",    "2021-02-14", "arjun.nair@corp.com"),
    (7, "Divya Krishnan", "AI",       "ML Engineer",         "2023-07-01", "divya.krishnan@corp.com"),
    (8, "Vikram Shah",    "Platform", "Engineering Manager", "2019-09-09", "vikram.shah@corp.com"),
]

PROJECTS = [
    (1, "Project X - Doc Intelligence",  "AI",       2, "active",    "2024-01-15"),
    (2, "Project Nova - Search Revamp",  "Platform", 4, "active",    "2024-03-01"),
    (3, "Project Aria - Chatbot",        "AI",       1, "on_hold",   "2023-11-01"),
    (4, "Project Halo - Design System",  "Design",   6, "completed", "2023-05-01"),
]

ISSUES = [
    (1, 1, 1, "OCR fails on scanned tables",              "open",        "high",     "2024-04-01"),
    (2, 1, 7, "Add reranking step to retrieval pipeline",  "in_progress", "medium",   "2024-04-10"),
    (3, 1, 2, "Latency spike on multimodal queries",       "open",        "critical", "2024-04-18"),
    (4, 2, 4, "Search index rebuild job flaky",            "closed",      "medium",   "2024-03-05"),
    (5, 2, 5, "Add read replica for search DB",            "open",        "high",     "2024-04-02"),
    (6, 3, 1, "Chatbot hallucinates on edge-case intents",  "open",        "medium",   "2024-02-20"),
    (7, 3, 3, "Evaluate smaller quantized model",           "closed",      "low",      "2024-01-30"),
    (8, 4, 6, "Update button component tokens",             "closed",      "low",      "2023-04-10"),
]


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?)", EMPLOYEES)
    cur.executemany("INSERT INTO projects VALUES (?,?,?,?,?,?)", PROJECTS)
    cur.executemany("INSERT INTO issues VALUES (?,?,?,?,?,?,?)", ISSUES)
    conn.commit()
    conn.close()
    print(f"Created {DB_PATH} with {len(EMPLOYEES)} employees, "
          f"{len(PROJECTS)} projects, {len(ISSUES)} issues.")


if __name__ == "__main__":
    main()
