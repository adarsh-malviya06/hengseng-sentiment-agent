"""Chronological per-asset price-only models and evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Open", "High", "Low", "Close", "Volume", "daily_return", "momentum_5d", "momentum_10d", "sma_5", "sma_10", "sma_20", "volatility_5d", "volatility_10d", "daily_range", "volume_change"]


def metrics(actual, predicted, prior_close):
    actual, predicted, prior_close = np.asarray(actual), np.asarray(predicted), np.asarray(prior_close)
    error = predicted - actual
    mape_mask = np.isfinite(actual) & (actual != 0) & np.isfinite(predicted)
    valid_direction = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(prior_close) & (prior_close > 0)
    actual_return = actual[valid_direction] / prior_close[valid_direction] - 1
    predicted_return = predicted[valid_direction] / prior_close[valid_direction] - 1
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape_pct": float(np.mean(np.abs(error[mape_mask] / actual[mape_mask])) * 100) if mape_mask.any() else None,
        # Both directions are computed as returns from the same current close.
        # A naive unchanged-price forecast has a zero return and is only a hit
        # when the actual next-session return is also exactly zero.
        "directional_accuracy_pct": float(np.mean(np.sign(predicted_return) == np.sign(actual_return)) * 100) if valid_direction.any() else None,
    }


def train_asset(df, models=None):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import TimeSeriesSplit

    df = df.sort_values("date").reset_index(drop=True)
    trainable = df.dropna(subset=FEATURES + ["target_next_close"])
    if len(trainable) < 40:
        return {"status": "insufficient_history", "rows": len(trainable)}
    split = max(1, int(len(trainable) * 0.8))
    tr, te = trainable.iloc[:split], trainable.iloc[split:]
    if te.empty:
        return {"status": "insufficient_test_rows", "rows": len(trainable)}
    models = models or {
        "Linear Regression": LinearRegression(),
        # Single process keeps this small per-asset dataset portable on Windows.
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=1),
    }
    Xtr, ytr = tr[FEATURES], tr["target_next_close"]
    Xte, yte = te[FEATURES], te["target_next_close"]
    results, predictions = [], []
    baseline = te["Close"].to_numpy()
    results.append({"model": "Naive (today close)", **metrics(yte, baseline, te["Close"]), "test_rows": len(te), "evaluation_start": str(te["date"].min()), "evaluation_end": str(te["next_trading_date"].max())})
    predictions.append(_backtest_frame(te, yte, baseline, "Naive (today close)"))
    model_instances = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        results.append({"model": name, **metrics(yte, pred, te["Close"]), "test_rows": len(te), "evaluation_start": str(te["date"].min()), "evaluation_end": str(te["next_trading_date"].max())})
        predictions.append(_backtest_frame(te, yte, pred, name))
        model_instances[name] = model
    # TimeSeriesSplit is a supplementary validation check, always ordered.
    folds = min(5, max(2, len(tr) // 30))
    if len(tr) >= folds + 1:
        tscv = TimeSeriesSplit(n_splits=folds)
        cv_errors = {"Naive (today close)": []}
        cv_errors.update({name: [] for name in model_instances})
        for a, b in tscv.split(Xtr):
            fold_actual = ytr.iloc[b].to_numpy()
            fold_prior = tr["Close"].iloc[b].to_numpy()
            naive = fold_prior
            cv_errors["Naive (today close)"].append(float(np.mean(np.abs(naive - fold_actual))))
            for name, model in model_instances.items():
                from sklearn.base import clone
                fold_model = clone(model).fit(Xtr.iloc[a], ytr.iloc[a])
                fold_pred = fold_model.predict(Xtr.iloc[b])
                cv_errors[name].append(float(np.mean(np.abs(fold_pred - fold_actual))))
        for name, fold_mae in cv_errors.items():
            next(x for x in results if x["model"] == name)["timeseries_cv_mae"] = float(np.mean(fold_mae))

    # Use training-only rolling validation to choose the model used for the
    # latest estimate. The holdout winner is reported separately for honest
    # evaluation and is not fed back into training/model selection.
    selection_scores = [x for x in results if x.get("timeseries_cv_mae") is not None]
    selected_model = min(selection_scores, key=lambda x: x["timeseries_cv_mae"])["model"] if selection_scores else "Naive (today close)"
    holdout_best_model = min(results, key=lambda x: x["mae"])["model"]
    for row in results:
        row["selected_model_by_cv"] = selected_model
        row["holdout_best_model_by_mae"] = holdout_best_model
        row["is_holdout_best_by_mae"] = row["model"] == holdout_best_model

    all_valid = df.dropna(subset=FEATURES)
    latest = all_valid.iloc[-1]
    forecast = []
    historical_returns = (trainable["target_next_close"] / trainable["Close"] - 1) * 100
    historical_min, historical_max = float(historical_returns.min()), float(historical_returns.max())
    for name, model in [("Naive (today close)", None), *model_instances.items()]:
        if model is None:
            value = float(latest.Close)
        else:
            fit_model = model.fit(trainable[FEATURES], trainable["target_next_close"])
            value = float(fit_model.predict(latest[FEATURES].to_frame().T)[0])
        predicted_return = (value / latest.Close - 1) * 100 if latest.Close else None
        outlier = predicted_return is not None and not historical_min <= predicted_return <= historical_max
        forecast.append({"ticker": latest.ticker, "asset_name": latest.asset_name, "exchange": latest.exchange, "latest_date": str(latest.date), "latest_close": float(latest.Close), "predicted_next_close": value, "predicted_return_pct": predicted_return, "actual_next_close": None, "model": name, "status": "price_only_estimate", "selected_model_by_cv": selected_model, "selected_for_prediction": name == selected_model, "prediction_warning": "outside_historical_daily_return_range" if outlier else None})
    return {"status": "ok", "metrics": results, "predictions": pd.concat(predictions, ignore_index=True), "forecast": forecast, "training_rows": len(tr), "test_rows": len(te)}


def _backtest_frame(current_rows, actual_next_close, predicted_close, model_name):
    """Represent actual and predicted returns on identical t to t+1 rows."""
    prior = current_rows["Close"].to_numpy(dtype=float)
    actual = np.asarray(actual_next_close, dtype=float)
    predicted = np.asarray(predicted_close, dtype=float)
    return pd.DataFrame({
        "date": current_rows["date"].to_numpy(),
        "next_trading_date": current_rows["next_trading_date"].to_numpy(),
        "prior_close": prior,
        "actual_next_close": actual,
        "predicted_close": predicted,
        "actual_return_pct": (actual / prior - 1) * 100,
        "predicted_return_pct": (predicted / prior - 1) * 100,
        "model": model_name,
    })


def support_resistance(df, window=60):
    """Recent 3-point swing lows/highs, returning observed levels only."""
    recent = df.sort_values("date").tail(window).reset_index(drop=True)
    supports, resistances = [], []
    for i in range(1, len(recent) - 1):
        if recent.Low.iloc[i] < recent.Low.iloc[i-1] and recent.Low.iloc[i] <= recent.Low.iloc[i+1]: supports.append(float(recent.Low.iloc[i]))
        if recent.High.iloc[i] > recent.High.iloc[i-1] and recent.High.iloc[i] >= recent.High.iloc[i+1]: resistances.append(float(recent.High.iloc[i]))
    return {"support": sorted(set(supports))[-5:], "resistance": sorted(set(resistances))[:5], "current_price": float(recent.Close.iloc[-1]) if len(recent) else None, "trend": "up" if len(recent) > 1 and recent.Close.iloc[-1] > recent.Close.iloc[0] else "down"}


def run_predictions(input_file=None):
    path = Path(input_file) if input_file else ROOT / "data" / "processed" / "all_assets_processed.csv"
    data = pd.read_csv(path)
    forecasts, metric_rows, pred_frames = [], [], []
    for _, group in data.groupby("ticker"):
        result = train_asset(group)
        if result["status"] != "ok":
            metric_rows.append({"ticker": group.ticker.iloc[0], "status": result["status"], "rows": result.get("rows")})
            continue
        forecasts.extend(result["forecast"])
        metric_rows.extend([{"ticker": group.ticker.iloc[0], "asset_name": group.asset_name.iloc[0], "exchange": group.exchange.iloc[0], **m} for m in result["metrics"]])
        pred = result["predictions"].copy(); pred["ticker"] = group.ticker.iloc[0]; pred_frames.append(pred)
    outdir = ROOT / "data" / "predictions"; outdir.mkdir(parents=True, exist_ok=True)
    latest_file = outdir / "latest_predictions.csv"
    history_file = outdir / "prediction_history.csv"
    current = pd.DataFrame(forecasts)
    if latest_file.exists():
        previous = pd.read_csv(latest_file)
        for i, forecast in previous.iterrows():
            if pd.notna(forecast.get("actual_next_close")):
                continue
            group = data[data.ticker == forecast.ticker].copy()
            group["_date"] = pd.to_datetime(group.date, errors="coerce")
            next_rows = group[group["_date"] > pd.Timestamp(forecast.latest_date)].sort_values("_date")
            if not next_rows.empty:
                previous.at[i, "actual_next_close"] = float(next_rows.iloc[0].Close)
                previous.at[i, "status"] = "actual_observed"
        history = pd.concat([pd.read_csv(history_file), previous, current], ignore_index=True) if history_file.exists() else pd.concat([previous, current], ignore_index=True)
    else:
        history = current.copy()
    history = history.drop_duplicates(subset=["ticker", "model", "latest_date"], keep="last")
    history.to_csv(history_file, index=False)
    current.to_csv(latest_file, index=False)
    pd.DataFrame(metric_rows).to_csv(outdir / "model_metrics.csv", index=False)
    if pred_frames: pd.concat(pred_frames).to_csv(outdir / "backtest_predictions.csv", index=False)
    print(f"Wrote price-only estimates for {len(forecasts)} assets to {outdir}")
    return forecasts, metric_rows


if __name__ == "__main__":
    run_predictions()
