import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from model import train_stock_model
from datetime import datetime
from prophet import Prophet

# ------------------------------------------------
# Page configuration
# ------------------------------------------------
st.set_page_config(
    page_title="AI Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0A0A0F;
    color: #FFFFFF;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #11111A;
    border-right: 1px solid #292936;
}

/* Main headings */
h1, h2, h3 {
    color: #FFFFFF !important;
}

/* Paragraphs and labels */
p, label, .stMarkdown {
    color: #D0D0DF;
}

/* Buttons */
.stButton > button {
    background-color: #6633CC;
    color: #FFFFFF;
    border: 1px solid #8055E8;
    border-radius: 10px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #8055E8;
    border-color: #00E5FF;
    color: #FFFFFF;
}

/* Input fields and dropdowns */
.stTextInput input,
.stNumberInput input {
    background-color: #15151F;
    color: #FFFFFF;
    border: 1px solid #393948;
    border-radius: 8px;
}

/* Cards */
[data-testid="stMetric"] {
    background-color: #15151F;
    border: 1px solid #292936;
    padding: 18px;
    border-radius: 14px;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #00E5FF;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #C3C3D3;
}

.stTabs [aria-selected="true"] {
    color: #00E5FF !important;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid #393948;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #292936;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #15151F;
    border: 1px solid #292936;
    border-radius: 10px;
}

/* Links and accent text */
a {
    color: #00E5FF !important;
}

/* Hide Streamlit's default footer */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Global dark theme for all Plotly charts
# (replaces the old fig.update_layout, which crashed
#  because `fig` did not exist yet)
# ------------------------------------------------
nexus_template = go.layout.Template(pio.templates["plotly_dark"])
nexus_template.layout.update(
    paper_bgcolor="#0A0A0F",
    plot_bgcolor="#15151F",
    font=dict(color="#D0D0DF"),
    title_font=dict(color="#FFFFFF"),
    margin=dict(l=10, r=10, t=50, b=10),
)
pio.templates["nexus"] = nexus_template
pio.templates.default = "nexus"

# ------------------------------------------------
# Shared constants
# ------------------------------------------------
HISTORY_FILE = Path("data/prediction_history.csv")

TRACKER_COLUMNS = [
    "recorded_at",
    "symbol",
    "latest_session",
    "latest_price",
    "predicted_direction",
    "up_probability",
    "test_accuracy",
    "balanced_accuracy",
    "baseline_accuracy",
    "actual_price",
    "actual_direction",
    "evaluation_date",
    "evaluation_status",
]

# ------------------------------------------------
# Dashboard heading
# ------------------------------------------------
st.title("AI-Powered Stock Market Prediction")
st.caption("NSE Stock Analysis | AI Prediction | News Sentiment | Forecasting")

# Available stocks
stocks = {
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

company = st.selectbox("Select a stock", list(stocks.keys()))
symbol = stocks[company]

if st.button("Load Stock Data"):

    with st.spinner("Loading historical stock data..."):
        data = yf.download(
            symbol,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

    if data.empty:
        st.error("Unable to retrieve stock data. Please try again.")

    else:
        close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        # Latest price and daily return
        col1, col2 = st.columns(2)

        col1.metric(
            "Latest Closing Price",
            f"₹{float(close.iloc[-1]):,.2f}"
        )

        daily_return = close.pct_change(fill_method=None).iloc[-1]

        col2.metric(
            "Latest Daily Return",
            f"{daily_return:.2%}" if pd.notna(daily_return) else "N/A"
        )

        # Historical data
        dates = pd.to_datetime(data.index)
        if dates.tz is not None:
            dates = dates.tz_localize(None)

        chart = pd.DataFrame({
            "Date": dates,
            "Closing Price": close.to_numpy()
        })

        # ------------------------------------------------
        # AI Stock Prediction
        # ------------------------------------------------
        st.divider()
        st.subheader("🤖 AI Stock Prediction")

        prediction = None

        try:
            with st.spinner("Running AI prediction..."):
                prediction = train_stock_model(symbol)

            st.write("Stock:", prediction["symbol"])
            st.write(
                "Latest Price:",
                f'₹{prediction["latest_price"]:.2f}'
            )
            st.write(
                "Predicted Direction:",
                prediction["direction"]
            )
            st.write(
                "Estimated UP Probability:",
                f'{prediction["up_probability"]:.2%}'
            )

            # Model performance
            st.subheader("📊 Model Performance")
            perf1, perf2, perf3 = st.columns(3)

            perf1.metric(
                "Test Accuracy",
                f'{prediction["accuracy"]:.2%}'
            )
            perf2.metric(
                "Baseline Accuracy",
                f'{prediction["baseline_accuracy"]:.2%}'
            )
            perf3.metric(
                "Balanced Accuracy",
                f'{prediction["balanced_accuracy"]:.2%}'
            )

            # Save prediction history (always with the full column set)
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

            history_record = pd.DataFrame([{
                "recorded_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "symbol": symbol,
                "latest_session": prediction["latest_date"],
                "latest_price": prediction["latest_price"],
                "predicted_direction": prediction["direction"],
                "up_probability": prediction["up_probability"],
                "test_accuracy": prediction["accuracy"],
                "balanced_accuracy": prediction["balanced_accuracy"],
                "baseline_accuracy": prediction["baseline_accuracy"]
            }]).reindex(columns=TRACKER_COLUMNS)
            if HISTORY_FILE.exists():
                try:
                    old_history = pd.read_csv(
                        HISTORY_FILE,
                        on_bad_lines="skip"
                    )
                    old_history.columns = old_history.columns.str.strip()

                    # Skip saving if this stock + session is already recorded
                    already_saved = (
                        (old_history["symbol"].astype(str).str.strip() == symbol)
                        & (
                            pd.to_datetime(
                                old_history["latest_session"], errors="coerce"
                            ).dt.strftime("%Y-%m-%d")
                            == pd.to_datetime(
                                prediction["latest_date"]
                            ).strftime("%Y-%m-%d")
                        )
                    ).any()

                    if already_saved:
                        combined_history = old_history
                    else:
                        combined_history = pd.concat(
                            [old_history, history_record],
                            ignore_index=True
                        )
                except Exception:
                    combined_history = history_record
            else:
                combined_history = history_record
            

            combined_history = combined_history.reindex(
                columns=TRACKER_COLUMNS
            )
            combined_history.to_csv(HISTORY_FILE, index=False)

            # Prediction summary cards
            pred_col1, pred_col2, pred_col3 = st.columns(3)

            pred_col1.metric(
                "Predicted Direction",
                prediction["direction"]
            )
            pred_col2.metric(
                "Estimated UP Probability",
                f'{prediction["up_probability"]:.2%}'
            )
            pred_col3.metric(
                "Model Test Accuracy",
                f'{prediction["accuracy"]:.2%}'
            )

            st.caption(
                f'Latest session: {prediction["latest_date"]} | '
                f'Majority baseline accuracy: '
                f'{prediction["baseline_accuracy"]:.2%}'
            )

            if prediction["accuracy"] < prediction["baseline_accuracy"]:
                st.warning(
                    "The model currently performs worse than the "
                    "majority baseline. Treat this prediction as experimental."
                )

        except Exception as e:
            st.error(f"Unable to generate prediction: {e}")

        # ------------------------------------------------
        # Model comparison
        # ------------------------------------------------
        if prediction is not None:
            st.divider()
            st.subheader("📊 AI Model Accuracy Comparison")

            comparison = pd.DataFrame({
                "Model": [
                    "Majority Baseline",
                    "Logistic Regression",
                    "Random Forest"
                ],
                "Accuracy (%)": [
                    prediction["baseline_accuracy"] * 100,
                    prediction["logistic_accuracy"] * 100,
                    prediction["accuracy"] * 100
                ]
            })

            fig_comparison = px.bar(
                comparison,
                x="Model",
                y="Accuracy (%)",
                title="Model Performance on Test Data",
                text="Accuracy (%)",
                range_y=[0, 100]
            )
            fig_comparison.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )
            st.plotly_chart(fig_comparison, width="stretch")

        # ------------------------------------------------
        # Historical price chart
        # ------------------------------------------------
        st.subheader("Historical Stock Prices")

        fig = px.line(
            chart,
            x="Date",
            y="Closing Price",
            title=f"{company} Closing Price"
        )
        st.plotly_chart(fig, width="stretch")

        # ------------------------------------------------
        # 7-DAY PROPHET FORECAST
        # ------------------------------------------------
        st.divider()
        st.subheader("🔮 7-Day Stock Price Forecast (Prophet)")
        st.caption(
            "Estimated prices for the next seven business days. "
            "Forecasts are uncertain and are not financial advice."
        )

        try:
            prophet_data = chart.rename(
                columns={
                    "Date": "ds",
                    "Closing Price": "y"
                }
            ).copy()

            prophet_data["ds"] = pd.to_datetime(prophet_data["ds"])
            prophet_data["y"] = pd.to_numeric(
                prophet_data["y"], errors="coerce"
            )

            prophet_data = (
                prophet_data.dropna(subset=["ds", "y"])
                .sort_values("ds")
                .drop_duplicates(subset=["ds"])
            )

            if len(prophet_data) < 60:
                st.warning(
                    "Not enough historical prices for a reliable "
                    "forecast. At least 60 observations are required."
                )
            else:
                with st.spinner("Training Prophet forecast model..."):
                    prophet_model = Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                        interval_width=0.80
                    )

                    prophet_model.fit(prophet_data)

                    # Next seven weekdays, excluding weekends.
                    future_dates = pd.bdate_range(
                        start=prophet_data["ds"].max()
                        + pd.offsets.BDay(1),
                        periods=7
                    )

                    future = pd.DataFrame({"ds": future_dates})
                    forecast = prophet_model.predict(future)

                forecast_view = forecast[
                    ["ds", "yhat", "yhat_lower", "yhat_upper"]
                ].rename(columns={
                    "ds": "Date",
                    "yhat": "Forecast Price",
                    "yhat_lower": "Lower Estimate",
                    "yhat_upper": "Upper Estimate"
                })

                latest_price = float(prophet_data["y"].iloc[-1])
                first_forecast = float(
                    forecast_view["Forecast Price"].iloc[0]
                )

                percentage_change = (
                    (first_forecast / latest_price) - 1
                ) * 100

                fc1, fc2, fc3 = st.columns(3)

                fc1.metric(
                    "Latest Closing Price",
                    f"₹{latest_price:,.2f}"
                )
                fc2.metric(
                    "First Forecast Price",
                    f"₹{first_forecast:,.2f}"
                )
                fc3.metric(
                    "Forecast Change",
                    f"{percentage_change:+.2f}%"
                )

                # Historical prices and forecast chart
                fig_forecast = go.Figure()

                fig_forecast.add_trace(go.Scatter(
                    x=prophet_data["ds"],
                    y=prophet_data["y"],
                    mode="lines",
                    name="Historical Price"
                ))

                # Forecast uncertainty interval
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_view["Date"],
                    y=forecast_view["Upper Estimate"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))

                fig_forecast.add_trace(go.Scatter(
                    x=forecast_view["Date"],
                    y=forecast_view["Lower Estimate"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    name="80% Forecast Interval",
                    hoverinfo="skip"
                ))

                fig_forecast.add_trace(go.Scatter(
                    x=forecast_view["Date"],
                    y=forecast_view["Forecast Price"],
                    mode="lines+markers",
                    name="Prophet Forecast"
                ))

                fig_forecast.update_layout(
                    title=f"{company} - 7-Day Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    hovermode="x unified"
                )

                st.plotly_chart(fig_forecast, width="stretch")

                st.subheader("📅 Forecast Values")
                st.dataframe(
                    forecast_view.round(2),
                    width="stretch",
                    hide_index=True
                )

                st.caption(
                    "The forecast interval represents model uncertainty. "
                    "Actual stock prices may fall outside this range. "
                    "Business days are used; exchange holidays are not "
                    "automatically excluded."
                )

        except Exception as e:
            st.error(f"Unable to generate Prophet forecast: {e}")

        # ------------------------------------------------
        # Recent market data
        # ------------------------------------------------
        st.subheader("Recent Market Data")
        st.dataframe(data.tail(10), width="stretch")

        # ------------------------------------------------
        # Financial News Sentiment
        # ------------------------------------------------
        st.divider()
        st.subheader("📰 Financial News Sentiment")

        sentiment_file = Path("data/news_sentiment.csv")

        if sentiment_file.exists():
            try:
                news = pd.read_csv(
                    sentiment_file,
                    on_bad_lines="skip"
                )

                if "symbol" in news.columns:
                    news["symbol"] = (
                        news["symbol"].astype(str).str.strip()
                    )
                    news = news[news["symbol"] == symbol].copy()

                if news.empty:
                    st.info(
                        "No saved news results for this stock. "
                        "Run sentiment.py for the selected stock."
                    )

                elif all(
                    column in news.columns
                    for column in [
                        "sentiment",
                        "headline",
                        "published_at"
                    ]
                ):
                    news["sentiment"] = (
                        news["sentiment"]
                        .astype(str)
                        .str.strip()
                        .str.title()
                    )

                    counts = (
                        news["sentiment"]
                        .value_counts()
                        .reindex(
                            ["Positive", "Neutral", "Negative"],
                            fill_value=0
                        )
                    )

                    if "sentiment_score" in news.columns:
                        scores = pd.to_numeric(
                            news["sentiment_score"],
                            errors="coerce"
                        )
                        average_score = scores.mean()
                    else:
                        average_score = float("nan")

                    s1, s2, s3, s4 = st.columns(4)

                    s1.metric("Positive Headlines", int(counts["Positive"]))
                    s2.metric("Neutral Headlines", int(counts["Neutral"]))
                    s3.metric("Negative Headlines", int(counts["Negative"]))

                    if pd.notna(average_score):
                        if average_score > 0.05:
                            sentiment_label = "Positive"
                        elif average_score < -0.05:
                            sentiment_label = "Negative"
                        else:
                            sentiment_label = "Neutral"

                        s4.metric(
                            "Average Sentiment Score",
                            f"{average_score:.3f}"
                        )

                        st.caption(
                            f"Overall news sentiment: "
                            f"**{sentiment_label}**"
                        )
                    else:
                        s4.metric("Average Sentiment Score", "N/A")

                    fig_sentiment = px.pie(
                        values=counts.values,
                        names=counts.index,
                        title="News Sentiment Distribution"
                    )

                    st.plotly_chart(fig_sentiment, width="stretch")

                    st.subheader("Latest Collected Headlines")

                    st.dataframe(
                        news[["published_at", "headline", "sentiment"]],
                        width="stretch",
                        hide_index=True
                    )

                else:
                    st.warning(
                        "The sentiment CSV does not contain the "
                        "expected columns. Please check sentiment.py."
                    )

            except Exception as e:
                st.error(f"Unable to load news sentiment: {e}")
        else:
            st.warning(
                "No sentiment file found. Run python sentiment.py first."
            )

else:
    st.info(
        "Select a stock and click 'Load Stock Data' to begin."
    )

# ------------------------------------------------
# Prediction History
# ------------------------------------------------
st.divider()
st.subheader("🕘 Prediction History")

if HISTORY_FILE.exists():
    try:
        history_view = pd.read_csv(
            HISTORY_FILE,
            on_bad_lines="skip"
        )

        history_view.columns = history_view.columns.str.strip()

        if not history_view.empty:
            if "recorded_at" in history_view.columns:
                history_view = history_view.sort_values(
                    by="recorded_at",
                    ascending=False
                )

            st.dataframe(
                history_view.reset_index(drop=True),
                width="stretch",
                hide_index=True
            )
        else:
            st.info("No valid prediction records found.")

    except Exception as e:
        st.error(f"History error: {e}")
else:
    st.info(
        "Prediction history file not found. "
        "Generate a prediction first."
    )

# ==========================================
# PREDICTION ACCURACY TRACKER
# ==========================================
st.divider()
st.header("🎯 Prediction Accuracy Tracker")

if not HISTORY_FILE.exists():
    st.info("Generate a stock prediction first.")

else:
    try:
        history = pd.read_csv(
            HISTORY_FILE,
            on_bad_lines="skip"
        )
        history.columns = history.columns.str.strip()

        required = [
            "recorded_at",
            "symbol",
            "latest_session",
            "latest_price",
            "predicted_direction",
        ]

        if not all(col in history.columns for col in required):
            st.error(
                "The prediction history CSV has an invalid header. "
                "Back up the CSV before correcting its columns."
            )

        else:
            # Add missing columns
            for col in TRACKER_COLUMNS:
                if col not in history.columns:
                    history[col] = pd.NA

            history["latest_session"] = pd.to_datetime(
                history["latest_session"],
                errors="coerce"
            )

            history["latest_price"] = pd.to_numeric(
                history["latest_price"],
                errors="coerce"
            )

            history["actual_price"] = pd.to_numeric(
                history["actual_price"],
                errors="coerce"
            )

            for col in [
                "actual_direction",
                "evaluation_date",
                "evaluation_status"
            ]:
                history[col] = history[col].astype("object")

            history["symbol"] = (
                history["symbol"].fillna("").astype(str).str.strip()
            )

            history["predicted_direction"] = (
                history["predicted_direction"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

            # Download prices once for the selected stock.
            with st.spinner(f"Checking actual prices for {symbol}..."):
                market_data = yf.download(
                    symbol,
                    period="1y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False
                )

            if not market_data.empty:
                prices_close = market_data["Close"]

                if isinstance(prices_close, pd.DataFrame):
                    prices_close = prices_close.iloc[:, 0]

                market_dates = pd.to_datetime(market_data.index)

                if market_dates.tz is not None:
                    market_dates = market_dates.tz_localize(None)

                prices = pd.DataFrame({
                    "date": market_dates.normalize(),
                    "close": prices_close.to_numpy()
                })

                prices["close"] = pd.to_numeric(
                    prices["close"], errors="coerce"
                )

                prices = (
                    prices.dropna()
                    .drop_duplicates("date")
                    .sort_values("date")
                    .reset_index(drop=True)
                )

                # Evaluate only predictions for the selected stock.
                for idx, record in history.iterrows():

                    if record["symbol"] != symbol:
                        continue

                    pred_date = record["latest_session"]
                    pred_price = record["latest_price"]
                    pred_direction = record["predicted_direction"]

                    if pd.isna(pred_date) or pd.isna(pred_price):
                        continue

                    pred_date = pd.Timestamp(pred_date).normalize()

                    # First trading day strictly after the prediction day.
                    next_prices = prices[prices["date"] > pred_date]

                    if next_prices.empty:
                        history.at[idx, "actual_price"] = float("nan")
                        history.at[idx, "actual_direction"] = ""
                        history.at[idx, "evaluation_date"] = ""
                        history.at[idx, "evaluation_status"] = "Pending"
                        continue

                    next_row = next_prices.iloc[0]
                    actual_price = float(next_row["close"])
                    actual_date = next_row["date"]

                    if actual_price > float(pred_price):
                        actual_direction = "UP"
                    elif actual_price < float(pred_price):
                        actual_direction = "DOWN"
                    else:
                        actual_direction = "FLAT"

                    if actual_direction == "FLAT":
                        status = "Flat / No Change"
                    elif pred_direction == actual_direction:
                        status = "Correct"
                    else:
                        status = "Incorrect"

                    history.at[idx, "actual_price"] = actual_price
                    history.at[idx, "actual_direction"] = actual_direction
                    history.at[idx, "evaluation_date"] = (
                        actual_date.strftime("%Y-%m-%d")
                    )
                    history.at[idx, "evaluation_status"] = status

            else:
                st.warning(
                    "Yahoo Finance returned no price data. "
                    "Predictions have not been evaluated."
                )

            # Normalize empty values and save consistent column order.
            for col in [
                "actual_direction",
                "evaluation_date",
                "evaluation_status",
            ]:
                history[col] = history[col].fillna("").astype(str)

            history = history.reindex(columns=TRACKER_COLUMNS)
            history.to_csv(HISTORY_FILE, index=False)

            # Selected-stock metrics
            selected = history[history["symbol"] == symbol].copy()

            st.subheader(f"Performance for {company}")

            if selected.empty:
                st.info("No prediction history for this stock.")

            else:
                selected["evaluation_status"] = (
                    selected["evaluation_status"].replace("", "Pending")
                )

                evaluated = selected[
                    selected["evaluation_status"].isin(
                        ["Correct", "Incorrect"]
                    )
                ]

                correct = int(
                    (evaluated["evaluation_status"] == "Correct").sum()
                )
                incorrect = int(
                    (evaluated["evaluation_status"] == "Incorrect").sum()
                )
                pending = int(
                    (selected["evaluation_status"] == "Pending").sum()
                )

                total = correct + incorrect
                success_rate = correct / total * 100 if total else None

                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Evaluated Predictions", total)
                t2.metric("Correct", correct)
                t3.metric("Incorrect", incorrect)
                t4.metric(
                    "Success Rate",
                    f"{success_rate:.2f}%"
                    if success_rate is not None else "N/A"
                )

                st.caption(f"Pending predictions: {pending}")

                st.subheader("Prediction Evaluation History")

                display_columns = [
                    "recorded_at",
                    "symbol",
                    "latest_session",
                    "latest_price",
                    "predicted_direction",
                    "evaluation_date",
                    "actual_price",
                    "actual_direction",
                    "evaluation_status",
                ]

                st.dataframe(
                    selected[display_columns].sort_values(
                        "recorded_at",
                        ascending=False
                    ),
                    width="stretch",
                    hide_index=True
                )

                if total:
                    chart_data = evaluated.copy()
                    chart_data["recorded_at"] = pd.to_datetime(
                        chart_data["recorded_at"],
                        errors="coerce"
                    )
                    chart_data = (
                        chart_data.dropna(subset=["recorded_at"])
                        .sort_values("recorded_at")
                    )
                    chart_data["correct_value"] = (
                        chart_data["evaluation_status"] == "Correct"
                    ).astype(int)
                    chart_data["running_accuracy"] = (
                        chart_data["correct_value"]
                        .expanding()
                        .mean() * 100
                    )

                    fig_accuracy = px.line(
                        chart_data,
                        x="recorded_at",
                        y="running_accuracy",
                        markers=True,
                        title="Prediction Success Rate Over Time",
                        labels={
                            "recorded_at": "Prediction Date",
                            "running_accuracy": "Success Rate (%)",
                        }
                    )
                    fig_accuracy.update_yaxes(range=[0, 100])
                    st.plotly_chart(fig_accuracy, width="stretch")

                st.caption(
                    "Predictions are evaluated using the next available "
                    "trading session's closing price. Pending predictions "
                    "are excluded from the success rate."
                )

    except Exception as e:
        st.error(f"Prediction tracker error: {e}")
