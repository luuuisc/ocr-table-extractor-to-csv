from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class NumericMetrics:
    column: str
    mse: float
    rmse: float
    r2: float
    n: int


@dataclass
class TableEvaluation:
    numeric_by_column: List[NumericMetrics]
    numeric_overall: Optional[NumericMetrics]
    text_accuracy: float
    total_cells: int
    matched_cells: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "numeric_by_column": [metric.__dict__ for metric in self.numeric_by_column],
            "numeric_overall": self.numeric_overall.__dict__ if self.numeric_overall else None,
            "text_accuracy": self.text_accuracy,
            "total_cells": self.total_cells,
            "matched_cells": self.matched_cells,
        }


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # normalizar espacios
    df = df.map(lambda x: (x or "").strip())
    return df


def _coerce_numeric(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    clean = series.replace({"": np.nan, "-": np.nan})
    numeric = pd.to_numeric(clean, errors="coerce")
    mask = ~numeric.isna()
    return numeric.to_numpy(), mask.to_numpy()


def _compute_numeric_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    column: str,
) -> Optional[NumericMetrics]:
    if len(y_true) == 0 or len(y_pred) == 0:
        return None
    length = min(len(y_true), len(y_pred))
    y_true = y_true[:length]
    y_pred = y_pred[:length]

    errors = y_pred - y_true
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum(errors ** 2) / denom)

    return NumericMetrics(column=column, mse=mse, rmse=rmse, r2=r2, n=length)


def evaluate_tables(
    reference_csv: str,
    predicted_csv: str,
    *,
    numeric_columns: Optional[List[str]] = None,
) -> TableEvaluation:
    df_ref = _read_csv(reference_csv)
    df_pred = _read_csv(predicted_csv)

    # igualar número de columnas rellenando faltantes
    max_cols = max(df_ref.shape[1], df_pred.shape[1])
    while df_ref.shape[1] < max_cols:
        df_ref[f"_ref_dummy_{df_ref.shape[1]}"] = ""
    while df_pred.shape[1] < max_cols:
        df_pred[f"_pred_dummy_{df_pred.shape[1]}"] = ""

    max_rows = max(len(df_ref), len(df_pred))
    if len(df_ref) < max_rows:
        pad_rows = max_rows - len(df_ref)
        padding = pd.DataFrame([[""] * df_ref.shape[1]] * pad_rows, columns=df_ref.columns)
        df_ref = pd.concat([df_ref, padding], ignore_index=True)
    if len(df_pred) < max_rows:
        pad_rows = max_rows - len(df_pred)
        padding = pd.DataFrame([[""] * df_pred.shape[1]] * pad_rows, columns=df_pred.columns)
        df_pred = pd.concat([df_pred, padding], ignore_index=True)

    # text accuracy
    total_cells = int(df_ref.shape[0] * df_ref.shape[1])
    matches = int((df_ref.values == df_pred.values).sum())
    text_accuracy = matches / total_cells if total_cells else 0.0

    # numeric metrics
    numeric_metrics: List[NumericMetrics] = []
    all_true: List[float] = []
    all_pred: List[float] = []

    for idx, column in enumerate(df_ref.columns):
        col_name = column if column not in df_ref.columns[df_ref.columns.str.startswith("_ref_dummy_")] else f"col_{idx}"
        if numeric_columns and column not in numeric_columns and col_name not in numeric_columns:
            continue

        true_values, mask_true = _coerce_numeric(df_ref.iloc[:, idx])
        pred_values, mask_pred = _coerce_numeric(df_pred.iloc[:, idx])

        mask = mask_true & mask_pred
        if not mask.any():
            continue

        filtered_true = true_values[mask]
        filtered_pred = pred_values[mask]
        metric = _compute_numeric_metrics(filtered_true, filtered_pred, column=col_name)
        if metric:
            numeric_metrics.append(metric)
            all_true.extend(filtered_true.tolist())
            all_pred.extend(filtered_pred.tolist())

    overall_metric = None
    if all_true and all_pred:
        overall_metric = _compute_numeric_metrics(np.array(all_true), np.array(all_pred), column="overall")

    return TableEvaluation(
        numeric_by_column=numeric_metrics,
        numeric_overall=overall_metric,
        text_accuracy=text_accuracy,
        total_cells=total_cells,
        matched_cells=matches,
    )


def write_report(evaluation: TableEvaluation, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Metric", "Column", "Value", "N"])
        writer.writerow(["text_accuracy", "-", f"{evaluation.text_accuracy:.4f}", evaluation.total_cells])
        for metric in evaluation.numeric_by_column:
            writer.writerow(["mse", metric.column, f"{metric.mse:.6f}", metric.n])
            writer.writerow(["rmse", metric.column, f"{metric.rmse:.6f}", metric.n])
            writer.writerow(["r2", metric.column, f"{metric.r2:.6f}", metric.n])
        if evaluation.numeric_overall:
            writer.writerow(["overall_mse", evaluation.numeric_overall.column, f"{evaluation.numeric_overall.mse:.6f}", evaluation.numeric_overall.n])
            writer.writerow(["overall_rmse", evaluation.numeric_overall.column, f"{evaluation.numeric_overall.rmse:.6f}", evaluation.numeric_overall.n])
            writer.writerow(["overall_r2", evaluation.numeric_overall.column, f"{evaluation.numeric_overall.r2:.6f}", evaluation.numeric_overall.n])
