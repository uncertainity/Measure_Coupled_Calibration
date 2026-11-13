import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union


ROOT = Path(__file__).resolve().parent.parent
CONTRACT_COLUMNS = ["date", "hour", "strike", "maturity", "true_price"]

ERROR_COLUMNS = [
    "date",
    "hour",
    "strike",
    "maturity",
    "true_price",
    "computed_price",
    "abs_error",
]


def read_bates_errors(path: Path, method: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    # Baseline Bates rows include hour even though their current header omits it.
    df = (pd.read_csv(path, names=ERROR_COLUMNS, skiprows=1)
          if "hour" not in header else pd.read_csv(path))
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["hour"].astype(int)

    numeric_columns = [
        "strike",
        "maturity",
        "true_price",
        "computed_price",
        "abs_error",
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    recomputed = (df["true_price"] - df["computed_price"]).abs()
    if not np.allclose(df["abs_error"], recomputed, rtol=1e-4, atol=6e-4):
        raise ValueError(f"{path} has inconsistent abs_error values")
    df["method"] = method
    return df


def add_moneyness(df, stock_df):
    state_df = df.merge(
        stock_df[["date", "hour", "stock_price"]],
        on=["date", "hour"],
        how="left",
    )

    missing_stock = state_df["stock_price"].isna().sum()
    if missing_stock:
        raise ValueError(f"Missing stock_price for {missing_stock} option error rows")

    state_df["Moneyness_Val"] = state_df["strike"] / state_df["stock_price"]
    conditions = [
        state_df["Moneyness_Val"] < 0.97,
        state_df["Moneyness_Val"].between(0.97, 1.03),
        state_df["Moneyness_Val"] > 1.03,
    ]
    choices = ["ITM", "ATM", "OTM"]
    state_df["Moneyness"] = np.select(conditions, choices, default="Unknown")

    return state_df


def metric_row(df: pd.DataFrame) -> Dict[str, Union[float, int]]:
    squared_error = (df["true_price"] - df["computed_price"]) ** 2
    return {"count": len(df), "date_hours": df[["date", "hour"]].drop_duplicates().shape[0],
            "mae": df["abs_error"].mean(), "median_ae": df["abs_error"].median(),
            "rmse": np.sqrt(squared_error.mean()), "max_ae": df["abs_error"].max()}


def get_moneyness_metrics(state_df):
    state_df = state_df.copy()
    squared_error = (state_df["true_price"] - state_df["computed_price"]) ** 2
    state_df["squared_error"] = squared_error

    moneyness_metrics = (
        state_df.groupby("Moneyness")
        .agg(
            MAE=("abs_error", "mean"),
            **{"Median AE": ("abs_error", "median")},
            RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
            count=("abs_error", "size"),
        )
    )

    overall = metric_row(state_df)

    time_metrics = (
        state_df.groupby(["date", "hour"])
        .agg(
            MAE=("abs_error", "mean"),
            RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
        )
    )

    time_metrics_summary = {
        "Average Date-Hour MAE": time_metrics["MAE"].mean(),
        "Median Date-Hour MAE": time_metrics["MAE"].median(),
        "Average Date-Hour RMSE": time_metrics["RMSE"].mean(),
    }

    return moneyness_metrics, overall, time_metrics, time_metrics_summary


def format_report_section(title, moneyness_metrics, overall, time_metrics_summary):
    overall_df = pd.Series(overall).to_frame("value")
    time_summary_df = pd.Series(time_metrics_summary).to_frame("value")

    return "\n".join([
        title,
        "=" * len(title),
        "",
        "Overall",
        overall_df.to_string(float_format=lambda x: f"{x:.6f}"),
        "",
        "Moneyness Metrics",
        moneyness_metrics.to_string(float_format=lambda x: f"{x:.6f}"),
        "",
        "Date-Hour Metrics Summary",
        time_summary_df.to_string(float_format=lambda x: f"{x:.6f}"),
        "",
    ])


def run_analysis(joint_output: Path, result_dir: Path, title: str) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    files = {"baseline_single": ROOT / "Output/Baseline_Crypto/single_state_errors.csv",
             "baseline_multi": ROOT / "Output/Baseline_Crypto/multi_state_errors.csv",
             "joint_eape": joint_output / "EAPE/Errors.csv",
             "joint_mape": joint_output / "MAPE/Errors.csv"}
    stock = pd.read_csv(ROOT / "Data/Real_Crypto/S_path.csv").rename(
        columns={"Underlying Value": "stock_price"})
    stock["date"] = pd.to_datetime(stock["date"])
    stock["hour"] = stock["hour"].astype(int)
    frames = {name: add_moneyness(read_bates_errors(path, name), stock)
              for name, path in files.items()}
    rows, sections = [], []
    for name, df in frames.items():
        by_money, overall, by_time, time_summary = get_moneyness_metrics(df)
        rows.append({"method": name, **overall, **time_summary})
        by_money.to_csv(result_dir / f"{name}_moneyness.csv")
        by_time.to_csv(result_dir / f"{name}_date_hour.csv")
        sections.append(format_report_section(name, by_money, overall, time_summary))
    pd.DataFrame(rows).to_csv(result_dir / "overall_summary.csv", index=False)

    common = None
    for df in frames.values():
        keys = df[CONTRACT_COLUMNS].drop_duplicates()
        common = keys if common is None else common.merge(keys, on=CONTRACT_COLUMNS)
    if common is None or common.empty:
        raise ValueError("No common option contracts exist across all methods")
    common_rows = [{"method": name, **metric_row(df.merge(common, on=CONTRACT_COLUMNS))}
                   for name, df in frames.items()]
    common_summary = pd.DataFrame(common_rows)
    common_summary.to_csv(result_dir / "common_contract_summary.csv", index=False)
    header = (f"{title}\n{'=' * len(title)}\n\nCommon contracts across all methods: {len(common)}"
              f"\n\nStrict common-contract comparison\n{common_summary.to_string(index=False)}\n\n")
    (result_dir / "Q_errors_report.txt").write_text(header + "\n".join(sections), encoding="utf-8")
    print(f"Wrote Q-space results to {result_dir}")


def main() -> None:
    for output, result, title in [
        (ROOT / "Output/Real_Crypto", ROOT / "Results/Real_Crypto/Q_errors", "Bates Q-space option-pricing error report"),
        (ROOT / "Output/Real_Crypto_LMS", ROOT / "Results/Real_Crypto_LMS/Q_errors", "Bates LMS Q-space option-pricing error report")]:
        run_analysis(output, result, title)


if __name__ == "__main__":
    main()
