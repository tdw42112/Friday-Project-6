import sqlite3
import pandas as pd

# connect to your SQLite database
conn = sqlite3.connect("feedback.db")

# list all tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in database:")
print(tables)

# choose one to inspect
table_name = tables.iloc[0, 0]  # first table name
print(f"\nPreview of table '{table_name}':")
df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5;", conn)
print(df)

# show column names
print("\nColumns:", df.columns.tolist())

conn.close()
