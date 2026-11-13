from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TRUE_PARAMS = ROOT / "Data/Synthetic_Stock/heston_params.csv"
RESULTS = ROOT / "Results/Synthetic_Stock"

METHOD_FILES = {
    "P": {
        "garch_mcmc": ROOT / "Output/Baseline_Synthetic_Stock/garch_calibration_params.csv",
        "pmcmc": ROOT / "Output/Baseline_Synthetic_Stock/pmcmc_calibration_params.csv",
        "joint_eape": ROOT / "Output/Synthetic_Stock/EAPE/parameters.csv",
        "joint_mape": ROOT / "Output/Synthetic_Stock/MAPE/parameters.csv",
        "joint_eape_lms": ROOT / "Output/Synthetic_Stock_LMS/EAPE/parameters.csv",
        "joint_mape_lms": ROOT / "Output/Synthetic_Stock_LMS/MAPE/parameters.csv",
    },
    "Q": {
        "single_state": ROOT / "Output/Baseline_Synthetic_Stock/single_state_calibration_params.csv",
        "multi_state": ROOT / "Output/Baseline_Synthetic_Stock/multi_state_calibration_params.csv",
        "joint_eape": ROOT / "Output/Synthetic_Stock/EAPE/parameters.csv",
        "joint_mape": ROOT / "Output/Synthetic_Stock/MAPE/parameters.csv",
        "joint_eape_lms": ROOT / "Output/Synthetic_Stock_LMS/EAPE/parameters.csv",
        "joint_mape_lms": ROOT / "Output/Synthetic_Stock_LMS/MAPE/parameters.csv",
    },
}

P_COLUMNS = ["mu", "kappa_P", "theta_P", "xi", "rho", "lambda", "v_t"]
Q_COLUMNS = ["kappa_Q", "theta_Q", "xi", "rho", "v_t"]


def normalize_date(value: object) -> pd.Timestamp:
    return pd.to_datetime(str(value).strip(), errors="raise").normalize()


def read_repo_csv(path: Path) -> pd.DataFrame:
    """Read both valid CSV dates and legacy unquoted 'Month day, year' dates."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = [item.strip() for item in rows[0]]
    records = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not row or not any(item.strip() for item in row):
            continue
        row = [item.strip() for item in row]
        if len(row) == len(header) + 1:
            row = [f"{row[0]}, {row[1]}", *row[2:]]
        if len(row) != len(header):
            raise ValueError(
                f"{path}:{line_number}: expected {len(header)} fields, got {len(row)}"
            )
        records.append(row)
    frame = pd.DataFrame(records, columns=header)
    frame["date"] = frame["date"].map(normalize_date)
    return frame


def load_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.read_csv(TRUE_PARAMS)
    truth["date"] = truth["date"].map(normalize_date)
    truth = truth.rename(columns={"volatility": "v_t"})
    for column in P_COLUMNS:
        truth[column] = pd.to_numeric(truth[column], errors="raise")
    p_truth = truth[["date", *P_COLUMNS]].copy()
    q_truth = truth[["date", "xi", "rho", "v_t"]].copy()
    q_truth["kappa_Q"] = truth["kappa_P"] + truth["lambda"]
    q_truth["theta_Q"] = truth["kappa_P"] * truth["theta_P"] / q_truth["kappa_Q"]
    return p_truth, q_truth[["date", *Q_COLUMNS]]


def estimate_columns(path: Path, method: str, space: str) -> pd.DataFrame:
    frame = read_repo_csv(path)
    if method in {"garch_mcmc", "pmcmc"}:
        mapping = {
            "mu_mean": "mu", "kappa_mean": "kappa_P", "theta_mean": "theta_P",
            "vol-of-vol_mean": "xi", "rho_mean": "rho",
            "vproxy": "v_t", "v_t_mean": "v_t",
        }
    elif method in {"single_state", "multi_state"}:
        mapping = {
            "kappa": "kappa_Q", "theta": "theta_Q", "vol-of-vol": "xi",
            "rho": "rho", "v0": "v_t",
        }
    else:
        mapping = {
            "mu_mean": "mu", "mu": "mu",
            "vol-of-vol_mean": "xi", "vol-of-vol": "xi",
            "kappa_mean": "kappa_P", "kappa": "kappa_P",
            "theta_mean": "theta_P", "theta": "theta_P",
            "rho_mean": "rho", "rho": "rho",
            "lambda_mean": "lambda", "lambda": "lambda",
            "v_t_mean": "v_t", "v_t": "v_t",
        }
    frame = frame.rename(columns={key: value for key, value in mapping.items() if key in frame})

    if space == "Q" and method.startswith("joint_"):
        required_p = set(P_COLUMNS)
        missing = required_p.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing joint P-space columns: {sorted(missing)}")
        frame["kappa_Q"] = pd.to_numeric(frame["kappa_P"]) + pd.to_numeric(frame["lambda"])
        frame["theta_Q"] = (
            pd.to_numeric(frame["kappa_P"]) * pd.to_numeric(frame["theta_P"])
            / frame["kappa_Q"]
        )

    columns = P_COLUMNS if space == "P" else Q_COLUMNS
    available = [column for column in columns if column in frame.columns]
    if not available:
        raise ValueError(f"{path} has no recognized {space}-space parameter columns")
    result = frame[["date", *available]].copy()
    for column in available:
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def compare_method(method: str, estimate: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameters = [column for column in truth.columns if column != "date" and column in estimate]
    merged = truth[["date", *parameters]].merge(
        estimate[["date", *parameters]], on="date", suffixes=("_true", "_estimated"),
        how="inner", validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(f"{method}: no dates overlap with the synthetic truth")
    long_rows = []
    for parameter in parameters:
        true = merged[f"{parameter}_true"]
        estimated = merged[f"{parameter}_estimated"]
        error = estimated - true
        denominator = true.abs().replace(0.0, np.nan)
        for date, true_value, estimate_value, signed, relative in zip(
            merged["date"], true, estimated, error, error.abs() / denominator
        ):
            long_rows.append({
                "date": date, "method": method, "parameter": parameter,
                "true_value": true_value, "estimated_value": estimate_value,
                "signed_error": signed, "absolute_error": abs(signed),
                "absolute_percentage_error": relative * 100.0,
            })
    details = pd.DataFrame(long_rows)
    summary = details.groupby(["method", "parameter"], as_index=False).agg(
        matched_dates=("date", "size"),
        bias=("signed_error", "mean"),
        mae=("absolute_error", "mean"),
        rmse=("signed_error", lambda values: np.sqrt(np.mean(np.square(values)))),
        mape_percent=("absolute_percentage_error", "mean"),
    )
    return details, summary


def run_space(space: str, truth: pd.DataFrame, result_root: Path) -> None:
    output_dir = result_root / f"{space}_errors"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method, path in METHOD_FILES[space].items():
        if not path.is_file():
            print(f"Skipping {method}: missing {path}")
            continue
        try:
            estimate = estimate_columns(path, method, space)
            details, summary = compare_method(method, estimate, truth)
        except ValueError as exc:
            print(f"Skipping {method}: {exc}")
            continue
        details.to_csv(output_dir / f"{method}_parameter_errors.csv", index=False, date_format="%Y-%m-%d")
        summaries.append(summary)
    if not summaries:
        raise RuntimeError(f"No {space}-space methods could be analyzed")
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "report.txt").write_text(
        f"Synthetic Heston {space}-space parameter comparison\n"
        + "=" * (44 + len(space)) + "\n\n"
        + combined.to_string(index=False, float_format=lambda value: f"{value:.8g}")
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {space}-space parameter comparisons to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare synthetic Heston estimates with their generating parameters.")
    parser.add_argument("--results", type=Path, default=RESULTS, help="Result directory")
    args = parser.parse_args()
    p_truth, q_truth = load_truth()
    run_space("P", p_truth, args.results)
    run_space("Q", q_truth, args.results)


if __name__ == "__main__":
    main()
