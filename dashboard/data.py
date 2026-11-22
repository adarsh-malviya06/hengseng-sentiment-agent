"""Small dashboard data access helpers, independent of Streamlit UI startup."""
from pathlib import Path
import pandas as pd


def load_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
