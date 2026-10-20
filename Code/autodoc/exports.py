"""CSV / JSON export helpers."""
import json

import pandas as pd


def to_csv_bytes(rows) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")  # utf-8-sig opens cleanly in Excel


def to_json_bytes(rows) -> bytes:
    return json.dumps(rows, indent=2, ensure_ascii=False, default=str).encode("utf-8")
