"""End-to-end backtest pipeline for a simple daily equity strategy.

The script loads price data, engineers a handful of technical features,
trains a regression model to predict the next day's return, converts the
predictions into fractional positions, and then runs a backtest that
includes trading costs.

Example usage::

    python scripts/backtest_pipeline.py --data data/sample_prices.csv \
        --trading-cost 0.0005

The input CSV must contain at least two columns: ``date`` and ``close``.
Dates should be parseable by :mod:`pandas` and the prices must be
numerical. The script splits the data chronologically into a training
window (70%) and an evaluation window (30%), ensuring that the model is
not trained on future information.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


@dataclass
class BacktestResult:
    """Container for the backtest output series and summary metrics."""

    history: pd.DataFrame
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float


def load_price_data(path: str) -> pd.DataFrame:
    """Load and prepare price data from ``path``.

    Parameters
    ----------
    path:
        CSV file containing ``date`` and ``close`` columns.

    Returns
    -------
    :class:`pandas.DataFrame`
        Data frame sorted by date with an additional ``return`` column.
    """

    df = pd.read_csv(path, parse_dates=["date"])
    missing_cols = {"date", "close"} - set(df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Input file is missing required columns: {missing}")

    df = df.sort_values("date").reset_index(drop=True)
    df["return"] = df["close"].pct_change()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a feature matrix and supervised learning target.

    The function engineers a few simple technical indicators including
    moving averages, momentum, and realized volatility. The prediction
    target is the following day's simple return.
    """

    features = df.copy()

    for window in (5, 10, 21):
        features[f"sma_{window}"] = features["close"].rolling(window).mean()
        features[f"ema_{window}"] = (
            features["close"].ewm(span=window, adjust=False).mean()
        )

    features["momentum_5"] = features["close"].pct_change(5)
    features["momentum_21"] = features["close"].pct_change(21)
    features["volatility_10"] = features["return"].rolling(10).std()

    # Z-score the closing price to help the model capture regime changes.
    features["close_z"] = (
        features["close"] - features["close"].rolling(63).mean()
    ) / features["close"].rolling(63).std()

    features["target"] = features["return"].shift(-1)
    features = features.dropna().reset_index(drop=True)
    return features


def split_train_test(features: pd.DataFrame, test_size: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the feature matrix chronologically into train and test windows."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_idx = int(len(features) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(features):
        raise ValueError("Not enough observations to split train/test")

    train = features.iloc[:split_idx].copy()
    test = features.iloc[split_idx:].copy()
    return train, test


def train_model(train: pd.DataFrame, feature_cols: Iterable[str]) -> RandomForestRegressor:
    """Fit a random forest regressor on the training data."""

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[list(feature_cols)], train["target"])
    return model


def generate_positions(predictions: pd.Series, scale: float = 0.5) -> pd.Series:
    """Transform predictions into fractional portfolio weights."""

    std = predictions.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=predictions.index)

    raw_signal = predictions / std
    positions = np.clip(scale * raw_signal, -1.0, 1.0)
    return pd.Series(positions, index=predictions.index)


def run_backtest(
    data: pd.DataFrame,
    predictions: pd.Series,
    trading_cost: float,
    position_scale: float,
) -> BacktestResult:
    """Apply positions to realized returns and compute summary metrics."""

    history = data.copy()
    history["prediction"] = predictions
    history["position"] = generate_positions(predictions, scale=position_scale)
    history["prev_position"] = history["position"].shift(1).fillna(0.0)
    history["turnover"] = history["position"].diff().abs().fillna(history["position"].abs())

    history["gross_return"] = history["prev_position"] * history["return"]
    history["cost"] = trading_cost * history["turnover"]
    history["net_return"] = history["gross_return"] - history["cost"]
    history["equity_curve"] = (1 + history["net_return"]).cumprod()

    ann_factor = np.sqrt(252)
    annualized_return = history["net_return"].mean() * 252
    annualized_volatility = history["net_return"].std(ddof=0) * ann_factor
    sharpe_ratio = (
        annualized_return / annualized_volatility if annualized_volatility else 0.0
    )

    running_max = history["equity_curve"].cummax()
    drawdowns = history["equity_curve"] / running_max - 1
    max_drawdown = drawdowns.min()

    return BacktestResult(history, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown)


def summarize(result: BacktestResult, mse: float) -> None:
    """Print human-readable output for the pipeline run."""

    print("Model evaluation:")
    print(f"  Mean squared error: {mse:.6f}")
    print("Backtest summary (evaluation window):")
    print(f"  Annualized return:   {result.annualized_return: .4%}")
    print(f"  Annualized vol:      {result.annualized_volatility: .4%}")
    print(f"  Sharpe ratio:        {result.sharpe_ratio: .2f}")
    print(f"  Max drawdown:        {result.max_drawdown: .2%}")
    print(f"  Final equity:        {result.history['equity_curve'].iloc[-1]: .4f}")


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV file containing daily price data.",
    )
    parser.add_argument(
        "--trading-cost",
        type=float,
        default=0.0005,
        help="Per-unit trading cost applied to position changes (default: 0.0005).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of observations reserved for backtest/evaluation.",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=0.5,
        help="Multiplier applied to normalized predictions when sizing positions.",
    )
    return parser.parse_args(args)


def main(cli_args: List[str] | None = None) -> None:
    args = parse_args(cli_args)

    df = load_price_data(args.data)
    features = engineer_features(df)
    train, test = split_train_test(features, test_size=args.test_size)

    feature_cols = [
        col
        for col in test.columns
        if col
        not in {
            "date",
            "close",
            "return",
            "target",
        }
    ]

    model = train_model(train, feature_cols)

    predictions = pd.Series(
        model.predict(test[feature_cols]), index=test.index, name="prediction"
    )

    mse = mean_squared_error(test["target"], predictions)

    backtest_slice = test[["date", "return"]].copy()
    result = run_backtest(
        backtest_slice,
        predictions,
        trading_cost=args.trading_cost,
        position_scale=args.position_scale,
    )

    summarize(result, mse)


if __name__ == "__main__":
    main()
