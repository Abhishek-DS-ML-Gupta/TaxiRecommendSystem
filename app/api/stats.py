"""
stats.py — GET /api/stats
Returns model performance summary from the GCN-LSTM experiment results CSV.

Note: The CSV has list-formatted cells like "[27, 27]" which causes pandas to
miscount fields. We use Python engine + quoting to handle this correctly.
"""

from fastapi import APIRouter
from typing import Dict, Any
import os
import pandas as pd
import numpy as np

router = APIRouter()

_RESULTS_CSV = os.path.join(
    os.path.dirname(__file__), '..', '..', 'results',
    'summary_result_of_gcn_lstm_split.csv'
)


def _load_results_df() -> pd.DataFrame:
    """
    Robustly loads the GCN-LSTM results CSV.
    The file has list-valued columns like "[27, 27]" wrapped in quotes,
    which causes field-count mismatches with the C parser. We use the
    Python engine with quotechar handling and skip bad lines as a fallback.
    """
    csv_path = os.path.abspath(_RESULTS_CSV)

    # Try with Python engine first (handles quoted lists correctly)
    try:
        df = pd.read_csv(
            csv_path,
            engine='python',
            quotechar='"',
            on_bad_lines='skip',
        )
        return df
    except Exception:
        pass

    # Final fallback: read raw lines and parse manually
    try:
        import csv, io
        rows = []
        header = None
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, quotechar='"')
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                else:
                    if len(row) >= len(header):
                        rows.append(row[:len(header)])
        df = pd.DataFrame(rows, columns=header)
        # Convert numeric columns
        for col in ['mae', 'learning_rate', 'dropout', 'prior_days', 'prior_hours']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        raise RuntimeError(f"Could not parse results CSV: {e}")


@router.get("/stats")
async def get_model_stats() -> Dict[str, Any]:
    """
    Returns aggregated model performance statistics from the
    GCN-LSTM hyperparameter search results (246 experiment runs).
    """
    csv_path = os.path.abspath(_RESULTS_CSV)
    if not os.path.exists(csv_path):
        return {"available": False, "message": f"Results CSV not found at {csv_path}"}

    try:
        df = _load_results_df()
    except Exception as e:
        return {"available": False, "message": str(e)}

    if df is None or len(df) == 0:
        return {"available": False, "message": "Results CSV is empty"}

    # Ensure mae is numeric
    if 'mae' not in df.columns:
        return {"available": False, "message": f"'mae' column not found. Columns: {df.columns.tolist()}"}

    df['mae'] = pd.to_numeric(df['mae'], errors='coerce')
    df = df.dropna(subset=['mae'])

    # Aggregate by date — best MAE per date
    best_by_date_rows = []
    if 'date' in df.columns:
        for date, grp in df.groupby('date'):
            best_idx = grp['mae'].idxmin()
            best_by_date_rows.append(grp.loc[best_idx])
    else:
        best_by_date_rows = []

    # Overall best
    best_row = df.loc[df['mae'].idxmin()]

    # MAE values list
    mae_values = df['mae'].dropna().tolist()

    def safe_str(val):
        if isinstance(val, float) and np.isnan(val):
            return ''
        return str(val)

    def safe_float(val, default=0.0):
        try:
            f = float(val)
            return 0.0 if np.isnan(f) else f
        except Exception:
            return default

    def safe_int(val, default=0):
        try:
            return int(float(val))
        except Exception:
            return default

    experiments_by_date = []
    for row in best_by_date_rows:
        experiments_by_date.append({
            "date": safe_str(row.get('date', '')),
            "mae": round(safe_float(row.get('mae', 0)), 6),
            "gc_dims": safe_str(row.get('gc_dims', '')),
            "lstm_dims": safe_str(row.get('lstm_dims', '')),
            "learning_rate": round(safe_float(row.get('learning_rate', 0)), 6),
            "prior_days": safe_int(row.get('prior_days', 3)),
            "prior_hours": safe_int(row.get('prior_hours', 4)),
        })

    # MAE histogram buckets (10 buckets) for bar chart
    mae_arr = np.array(mae_values)
    hist_counts, hist_edges = np.histogram(mae_arr, bins=15)
    mae_histogram = [
        {"bucket": round(float(hist_edges[i]), 5), "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    return {
        "available": True,
        "total_experiments": len(df),
        "best_overall": {
            "date": safe_str(best_row.get('date', '')),
            "mae": round(safe_float(best_row.get('mae', 0)), 6),
            "learning_rate": safe_float(best_row.get('learning_rate', 0)),
            "gc_dims": safe_str(best_row.get('gc_dims', '')),
            "lstm_dims": safe_str(best_row.get('lstm_dims', '')),
            "prior_days": safe_int(best_row.get('prior_days', 3)),
            "prior_hours": safe_int(best_row.get('prior_hours', 4)),
            "dropout": safe_float(best_row.get('dropout', 0)),
        },
        "mae_distribution": {
            "min": round(float(min(mae_values)), 6),
            "max": round(float(max(mae_values)), 6),
            "mean": round(float(np.mean(mae_values)), 6),
            "median": round(float(np.median(mae_values)), 6),
            "histogram": mae_histogram,
            "all_values": [round(float(v), 5) for v in mae_values],
        },
        "experiments_by_date": experiments_by_date,
        "paper": {
            "title": "A Cost-Effective Sequential Route Recommender System for Taxi Drivers",
            "authors": "Liu, Teng, Chen, Xiong",
            "journal": "INFORMS Journal on Computing",
            "year": 2023,
            "doi": "10.1287/ijoc.2021.0112"
        }
    }
