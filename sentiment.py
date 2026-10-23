
import feedparser
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import quote_plus
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path


OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

analyzer = SentimentIntensityAnalyzer()

# NSE symbols mapped to company names for news searches
COMPANIES = {
    "TCS.NS": "Tata Consultancy Services",
    "RELIANCE.NS": "Reliance Industries",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank"
}


def fetch_news(symbol):
    company = COMPANIES.get(symbol)

    if not company:
        raise ValueError(f"Unsupported stock symbol: {symbol}")

    query = quote_plus(f'"{company}" stock')
    url = (
        "https://news.google.com/rss/search?q="
        + query
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(url)

    if getattr(feed, "bozo", False) and not feed.entries:
        raise RuntimeError("Could not retrieve the news feed.")

    articles = []

    for entry in feed.entries:
        headline = entry.get("title", "").strip()

        if not headline:
            continue

        published = entry.get("published", "")

        try:
            published_at = parsedate_to_datetime(published)
            if published_at.tzinfo is None:
                published_at = published_at.replace(
                    tzinfo=timezone.utc
                )
        except (TypeError, ValueError, OverflowError):
            published_at = None

        scores = analyzer.polarity_scores(headline)
        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        articles.append({
            "symbol": symbol,
            "published_at": published_at,
            "headline": headline,
            "link": entry.get("link", ""),
            "positive_score": scores["pos"],
            "neutral_score": scores["neu"],
            "negative_score": scores["neg"],
            "sentiment_score": compound,
            "sentiment": sentiment,
            "retrieved_at": datetime.now(timezone.utc)
        })

    return pd.DataFrame(articles)


def main():
    symbol = "TCS.NS"

    print(f"Fetching news for {COMPANIES[symbol]}...")

    news = fetch_news(symbol)

    if news.empty:
        print("No news articles found. Try again later.")
        return

    output_path = OUTPUT_DIR / "news_sentiment.csv"
    news.to_csv(output_path, index=False)

    print(f"\nArticles collected: {len(news)}")

    print("\nSentiment distribution:")
    print(news["sentiment"].value_counts())

    print("\nRecent headlines:")
    print(news[["published_at", "headline", "sentiment"]].head(10)
          .to_string(index=False))

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
