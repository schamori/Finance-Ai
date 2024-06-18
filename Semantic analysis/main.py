import yfinance as yf
import pandas as pd
import finnhub
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from transformers import pipeline
from yahoo_fin import stock_info

# Finnhub API-Konfiguration
FINNHUB_API_KEY = 'cphfoqpr01qp5iv5eo2gcphfoqpr01qp5iv5eo30'
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Sentiment-Analyse-Pipeline initialisieren
nlp = pipeline("sentiment-analysis")

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.index = stock_data.index.tz_localize(None)
    return stock_data

def fetch_news_data(symbol, start_date, end_date):
    all_articles = []
    current_start_date = start_date
    current_end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=5)).strftime('%Y-%m-%d')
    
    while datetime.strptime(current_start_date, '%Y-%m-%d') < datetime.strptime(current_end_date, '%Y-%m-%d'):
        articles = finnhub_client.company_news(symbol, _from=current_start_date, to=current_end_date)
        all_articles.extend(articles)
        current_start_date = current_end_date
        current_end_date = (datetime.strptime(current_end_date, '%Y-%m-%d') + timedelta(days=5)).strftime('%Y-%m-%d')
        if datetime.strptime(current_end_date, '%Y-%m-%d') > datetime.strptime(end_date, '%Y-%m-%d'):
            current_end_date = end_date

    print(f"Anzahl der abgerufenen Artikel für {symbol}: {len(all_articles)}")
    return all_articles

def analyze_article_sentiment(articles):
    sentiments = []
    for article in articles:
        content = article.get('summary')
        if content:
            result = nlp(content[:512])
            score = result[0]['score']
            sentiment_score = score if result[0]['label'] == 'POSITIVE' else -score
            sentiments.append({'score': sentiment_score, 'publishedAt': article['datetime']})
    return sentiments

def correlate_sentiment_with_stock(ticker, stock_data, articles, sentiments):
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
    stock_data['MA30'] = stock_data['Adj Close'].rolling(window=30).mean()
    stock_data['MA100'] = stock_data['Adj Close'].rolling(window=100).mean()
    
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt'], unit='s').dt.tz_localize(None)
    sentiment_df.set_index('publishedAt', inplace=True)
    sentiment_daily_avg = sentiment_df.resample('D').mean()
    
    combined_df = pd.concat([stock_data, sentiment_daily_avg], axis=1)
    combined_df.dropna(subset=['Daily Return', 'score'], inplace=True)
    
    if len(combined_df) < 2:
        print(f"Zu wenige Datenpunkte für eine sinnvolle Korrelation für {ticker}")
        return None

    #print(f"Anzahl der Datenpunkte für {ticker}: {len(combined_df)}")

    combined_df = apply_sentiment_trading_strategy(combined_df)
    plot_sentiment_vs_return(ticker, combined_df, stock_data)

    correlation_return = combined_df['Daily Return'].corr(combined_df['score'])
    correlation_price = combined_df['Adj Close'].corr(combined_df['score'])
    return correlation_return, correlation_price, combined_df

def plot_sentiment_vs_return(ticker, combined_df, stock_data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 21))
    
    ax1_twin = ax1.twinx()
    ax1.plot(combined_df.index, combined_df['Daily Return'], 'g-', label='Daily Return')
    ax1_twin.plot(combined_df.index, combined_df['score'], 'b-', label='Sentiment Score')
    ax1.set_xlabel('Datum')
    ax1.set_ylabel('Tägliche Rendite', color='g')
    ax1_twin.set_ylabel('Sentiment Score', color='b')
    ax1.set_title(f'{ticker} Sentiment vs. Daily Return')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    ax2_twin = ax2.twinx()
    ax2.plot(stock_data.index, stock_data['Adj Close'], 'r-', label='Stock Price')
    ax2.plot(stock_data.index, stock_data['MA30'], 'orange', label='30-Day MA')
    ax2.plot(stock_data.index, stock_data['MA100'], 'purple', label='100-Day MA')
    ax2_twin.plot(combined_df.index, combined_df['score'], 'b-', label='Sentiment Score')
    ax2.set_xlabel('Datum')
    ax2.set_ylabel('Aktienkurs', color='r')
    ax2_twin.set_ylabel('Sentiment Score', color='b')
    ax2.set_title(f'{ticker} Stock Price vs. Sentiment')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    strategy_return, stock_return = calculate_strategy_and_stock_return(combined_df)
    
    ax3.plot(stock_return.index, stock_return, label='Stock Return', color='orange')
    ax3.plot(strategy_return.index, strategy_return, label='Strategy Return', color='blue')
    ax3.set_xlabel('Datum')
    ax3.set_ylabel('Kumulierte Rendite')
    ax3.set_title(f'{ticker} Strategy Return vs. Stock Return')
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    initial_investment = 1
    final_strategy_value = strategy_return.iloc[-1] * initial_investment
    final_stock_value = stock_return.iloc[-1] * initial_investment

    print(f"Endwert bei Verwendung der Strategie: {final_strategy_value:.2f} Münzen")
    print(f"Endwert bei einfachem Kauf und Halten: {final_stock_value:.2f} Münzen")

def apply_sentiment_trading_strategy(combined_df):
    signals = []
    for i in range(len(combined_df)):
        if combined_df['score'].iloc[i] > 0:
            signals.append(1)
        else:
            signals.append(-1)

    combined_df['Signal'] = signals
    combined_df['Strategy Return'] = combined_df['Signal'].shift(1) * combined_df['Daily Return']
    combined_df.dropna(subset=['Strategy Return'], inplace=True)
    return combined_df

def calculate_strategy_and_stock_return(combined_df):
    initial_value = 100
    strategy_return = [initial_value]
    stock_return = [initial_value]

    for i in range(1, len(combined_df)):
        if combined_df['Signal'].iloc[i] == 1:
            strategy_return.append(strategy_return[-1] * (1 + combined_df['Daily Return'].iloc[i]))
        else:
            strategy_return.append(strategy_return[-1] * (1 - combined_df['Daily Return'].iloc[i]))
        
        stock_return.append(stock_return[-1] * (1 + combined_df['Daily Return'].iloc[i]))

    return pd.Series(strategy_return, index=combined_df.index), pd.Series(stock_return, index=combined_df.index)

def get_trending_stock():
    from requests_html import HTMLSession
    trending_stocks = stock_info.get_day_most_active().head(1)['Symbol'].tolist()
    return trending_stocks[0] if trending_stocks else None

def main():
    ticker = input("Bitte geben Sie das Aktienkürzel ein (oder lassen Sie es leer, um die Top-Trending-Aktie zu verwenden): ").strip().upper()
    if not ticker:
        ticker = get_trending_stock()
        if not ticker:
            print("Keine Trendaktien gefunden.")
            return
        print(f"Verwende Top-Trending-Aktie: {ticker}")

    start_date = (datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d')
    end_date = (datetime.now() - timedelta(days=0)).strftime('%Y-%m-%d')

    print(f"Analysiere {ticker}")
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    articles = fetch_news_data(ticker, start_date, end_date)
    if not articles:
        print(f"Keine Artikel gefunden für {ticker}")
        return
    
    sentiments = analyze_article_sentiment(articles)
    if not sentiments:
        print(f"Keine Sentiment-Analyse-Daten für {ticker}")
        return
    
    correlation_return, correlation_price, combined_df = correlate_sentiment_with_stock(ticker, stock_data, articles, sentiments)
    if correlation_return is not None and correlation_price is not None:
        print(f"Korrelation für {ticker} (Daily Return): {correlation_return}")
        print(f"Korrelation für {ticker} (Stock Price): {correlation_price}")

        combined_df = apply_sentiment_trading_strategy(combined_df)
        # Speichern in eine CSV-Datei
        combined_df[['Adj Close', 'score', 'Signal', 'Strategy Return']].to_csv(f'{ticker}_analysis.csv')
        print(f"Daten wurden in {ticker}_analysis.csv gespeichert.")
    else:
        print(f"Zu wenige Datenpunkte für eine sinnvolle Korrelation für {ticker}")

if __name__ == "__main__":
    main()
