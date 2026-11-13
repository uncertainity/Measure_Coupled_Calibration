from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REQUIRED_ERROR_COLUMNS = {
    "date",
    "strike",
    "maturity",
    "true_price",
    "computed_price",
    "abs_error",
}
CONTRACT_COLUMNS = ["date", "strike", "maturity", "true_price"]


def load_stock_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing stock path: {path}")
    stock_df = pd.read_csv(path).rename(
        columns={
            "Date": "date",
            "Underlying Value": "stock_price",
            "underlying": "stock_price",
        }
    )
    required = {"date", "stock_price"}
    missing = required.difference(stock_df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    stock_df["date"] = pd.to_datetime(stock_df["date"], errors="raise")
    if stock_df["date"].duplicated().any():
        raise ValueError(f"{path} contains duplicate dates")
    return stock_df[["date", "stock_price"]]


def load_error_data(method: str, path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {method} errors: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_ERROR_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[list(REQUIRED_ERROR_COLUMNS)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    numeric_columns = REQUIRED_ERROR_COLUMNS.difference({"date"})
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df[list(REQUIRED_ERROR_COLUMNS)].isna().any().any():
        raise ValueError(f"{path} contains missing required values")

    recomputed_abs_error = (df["true_price"] - df["computed_price"]).abs()
    # computed_price and abs_error are serialized at slightly different precision.
    if not np.allclose(df["abs_error"], recomputed_abs_error, rtol=1e-4, atol=6e-4):
        raise ValueError(f"{path} has inconsistent abs_error values")
    df["method"] = method
    return df


def add_moneyness(df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(stock_df, on="date", how="left", validate="many_to_one")
    if merged["stock_price"].isna().any():
        dates = merged.loc[merged["stock_price"].isna(), "date"].dt.strftime("%Y-%m-%d")
        raise ValueError(f"Missing stock prices for dates: {sorted(dates.unique())}")
    merged["moneyness_value"] = merged["strike"] / merged["stock_price"]
    merged["moneyness"] = np.select(
        [
            merged["moneyness_value"] < 0.97,
            merged["moneyness_value"].between(0.97, 1.03, inclusive="both"),
            merged["moneyness_value"] > 1.03,
        ],
        ["ITM", "ATM", "OTM"],
        default="Unknown",
    )
    return merged


def metric_row(df: pd.DataFrame) -> Dict[str, Union[float, int]]:
    squared_error = (df["true_price"] - df["computed_price"]) ** 2
    return {
        "count": len(df),
        "dates": df["date"].nunique(),
        "mae": df["abs_error"].mean(),
        "median_ae": df["abs_error"].median(),
        "rmse": np.sqrt(squared_error.mean()),
        "max_ae": df["abs_error"].max(),
    }


def moneyness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name in ["ITM", "ATM", "OTM", "Unknown"]:
        group = df.loc[df["moneyness"] == group_name]
        if group.empty:
            continue
        rows.append({"moneyness": group_name, **metric_row(group)})
    return pd.DataFrame(rows)


def daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in df.groupby("date", sort=True):
        rows.append({"date": date, **metric_row(group)})
    return pd.DataFrame(rows)


def daily_summary(daily: pd.DataFrame) -> Dict[str, float]:
    return {
        "average_daily_mae": daily["mae"].mean(),
        "median_daily_mae": daily["mae"].median(),
        "average_daily_rmse": daily["rmse"].mean(),
    }


def common_contract_keys(method_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    common = None
    for df in method_frames.values():
        keys = df[CONTRACT_COLUMNS].drop_duplicates()
        common = keys if common is None else common.merge(keys, on=CONTRACT_COLUMNS, how="inner")
    if common is None or common.empty:
        raise ValueError("No common option contracts exist across all methods")
    return common


def format_section(
    method: str,
    overall: Dict[str, Union[float, int]],
    by_moneyness: pd.DataFrame,
    by_day_summary: Dict[str, float],
) -> str:
    return "\n".join(
        [
            method,
            "=" * len(method),
            "",
            "Overall",
            pd.Series(overall).to_string(float_format=lambda value: f"{value:.6f}"),
            "",
            "Moneyness metrics",
            by_moneyness.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
            "",
            "Daily metrics summary",
            pd.Series(by_day_summary).to_string(float_format=lambda value: f"{value:.6f}"),
            "",
        ]
    )


def run_analysis(
    stock_path: Path,
    baseline_output: Path,
    joint_output: Path,
    result_dir: Path,
    title: str,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    method_files = {
        "baseline_single": baseline_output / "single_state_errors.csv",
        "baseline_multi": baseline_output / "multi_state_errors.csv",
        "joint_eape": joint_output / "EAPE/Errors.csv",
        "joint_mape": joint_output / "MAPE/Errors.csv",
    }
    stock_df = load_stock_data(stock_path)
    method_frames = {
        method: add_moneyness(load_error_data(method, path), stock_df)
        for method, path in method_files.items()
    }

    overall_rows = []
    report_sections = []
    for method, df in method_frames.items():
        overall = {"method": method, **metric_row(df)}
        by_moneyness = moneyness_metrics(df)
        by_day = daily_metrics(df)
        by_day_summary = daily_summary(by_day)

        overall_rows.append({**overall, **by_day_summary})
        by_moneyness.insert(0, "method", method)
        by_moneyness.to_csv(result_dir / f"{method}_moneyness.csv", index=False)
        by_day.to_csv(result_dir / f"{method}_daily.csv", index=False, date_format="%Y-%m-%d")
        report_sections.append(format_section(method, overall, by_moneyness, by_day_summary))

    overall_summary = pd.DataFrame(overall_rows)
    overall_summary.to_csv(result_dir / "overall_summary.csv", index=False)

    common_keys = common_contract_keys(method_frames)
    common_rows = []
    for method, df in method_frames.items():
        common_df = df.merge(common_keys, on=CONTRACT_COLUMNS, how="inner", validate="one_to_one")
        common_rows.append({"method": method, **metric_row(common_df)})
    common_summary = pd.DataFrame(common_rows)
    common_summary.to_csv(result_dir / "common_contract_summary.csv", index=False)

    report_header = "\n".join(
        [
            title,
            "=" * len(title),
            "",
            f"Common contracts across all methods: {len(common_keys)}",
            "",
            "Strict common-contract comparison",
            common_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
            "",
        ]
    )
    (result_dir / "Q_errors_report.txt").write_text(
        report_header + "\n" + "\n".join(report_sections), encoding="utf-8"
    )
    print(f"Wrote Q-space results to {result_dir}")


def main() -> None:
    runs = [
        (ROOT / "Output/Real_Stock", ROOT / "Results/Real_Stock/Q_errors", "Heston Q-space option-pricing error report"),
        (ROOT / "Output/Real_Stock_LMS", ROOT / "Results/Real_Stock_LMS/Q_errors", "Heston LMS Q-space option-pricing error report"),
    ]
    stock_path = ROOT / "Data/Real_Stock/S_path.csv"
    baseline_output = ROOT / "Output/Baseline_Stock"

    for joint_output, result_dir, title in runs:
        run_analysis(stock_path, baseline_output, joint_output, result_dir, title)


if __name__ == "__main__":
    main()
