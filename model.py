from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def train_stock_model(symbol="TCS.NS"):
    # Download historical stock data
    data = yf.download(
        symbol,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(f"Unable to download data for {symbol}")

    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Create price features
    df = pd.DataFrame({"close": close})
        # Returns and moving averages
    df["return_1d"] = df["close"].pct_change(fill_method=None)
    df["return_5d"] = df["close"].pct_change(5, fill_method=None)
    df["return_10d"] = df["close"].pct_change(10, fill_method=None)
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["volatility_20"] = df["return_1d"].rolling(20).std()

    # Relative position to moving averages
    df["price_sma5_ratio"] = df["close"] / df["sma_5"] - 1
    df["price_sma20_ratio"] = df["close"] / df["sma_20"] - 1
    df["price_sma50_ratio"] = df["close"] / df["sma_50"] - 1

    # RSI indicator
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD indicators
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    # Predict whether the next session closes higher
    df["target"] = (
        df["close"].shift(-1) > df["close"]
    ).astype(int)

    features = [
        "return_1d",
        "return_5d",
        "return_10d",
        "volatility_20",
        "price_sma5_ratio",
        "price_sma20_ratio",
        "price_sma50_ratio",
        "rsi",
        "macd",
        "macd_signal"
    ]
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=features)

    # Keep the latest features for the next-session prediction
    latest_features = df[features].iloc[[-1]].copy()

    # The final row has no known next-session outcome
    training_data = df.iloc[:-1].copy()

    if len(training_data) < 100:
        raise ValueError("Not enough historical data to train.")

    # Chronological train/test split
    split = int(len(training_data) * 0.8)
    train = training_data.iloc[:split]
    test = training_data.iloc[split:]

    X_train = train[features]
    y_train = train["target"]
    X_test = test[features]
    y_test = test["target"]

      # Majority baseline
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )

    # Logistic Regression model
    logistic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )

    logistic_model.fit(X_train, y_train)
    logistic_predictions = logistic_model.predict(X_test)

    logistic_accuracy = accuracy_score(
        y_test, logistic_predictions
    )
        # Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )

    # Train and evaluate the model
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, test_predictions)

    # Model comparison
    print("\n--- Model Comparison ---")
    print(f"Majority Baseline: {baseline_accuracy:.2%}")
    print(f"Logistic Regression: {logistic_accuracy:.2%}")
    print(f"Random Forest: {test_accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")    
    # Predict the next session
    direction_value = int(model.predict(latest_features)[0])
    probabilities = model.predict_proba(latest_features)[0]
    probability_by_class = dict(zip(model.classes_, probabilities))
    up_probability = float(probability_by_class.get(1, 0.0))
    result = {
        "symbol": symbol,
        "latest_date": str(data.index[-1].date()),
        "latest_price": float(close.iloc[-1]),
        "direction": "UP" if direction_value == 1 else "DOWN",
        "up_probability": up_probability,
        "accuracy": float(test_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
        "logistic_accuracy": float(logistic_accuracy),
        "balanced_accuracy": float(balanced_accuracy)
    }

    print("\nStock:", symbol)
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Majority baseline accuracy: {baseline_accuracy:.2%}")
    print("Latest session:", result["latest_date"])
    print("Predicted direction:", result["direction"])
    print(f"Estimated UP probability: {up_probability:.2%}")

    return result


if __name__ == "__main__":
    train_stock_model("TCS.NS")
