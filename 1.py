import yfinance as yf

# ticker4 = yf.Ticker("SPY")  # S&P 500 ETF
# info4 = ticker4.info

# # Look for price-related keys
# for key in info4.keys():
#     if 'price' in key.lower():
#         print(f"{key}: {info4.get(key)}")


# hist = yf.Ticker("AAPL").history(period="5d")
# print(hist)
# print()
# print(type(hist))
# print(hist.columns.tolist())

# Different periods
# print(yf.Ticker("AAPL").history(period="1mo").shape)
# print(yf.Ticker("AAPL").history(period="1y").shape)

# # Different intervals
# print(yf.Ticker("AAPL").history(period="5d", interval="1h").shape)

# # Hourly data for a year?
# hist = yf.Ticker("AAPL").history(period="1y", interval="1h")
# print(hist.shape)
# print(hist.index[0])  # First date
# print(hist.index[-1]) # Last date

# # 5-minute data for a month?
# hist = yf.Ticker("AAPL").history(period="1mo", interval="5m")
# print(hist.shape)
# print(hist.index[0])  # First date
# print(hist.index[-1]) # Last date

# ticker = yf.Ticker("AAPL")

# # Get the balance sheet
# bs = ticker.balance_sheet
# # print(type(bs))
# # print(bs.shape)
# # print(bs.columns.tolist())  # These are dates
# # print(bs.index.tolist()[:10])  # These are line items (first 10)

# print(bs.loc['Total Debt'])
# # See what other line items are available
# print(bs.index.tolist())

# ticker = yf.Ticker("AAPL")
# bs = ticker.balance_sheet

# # Total Debt over 5 years
# print(bs.loc['Total Debt'])
# print()

# # All available line items
# print(bs.index.tolist())

# ticker = yf.Ticker("AAPL")

# # Income statement
# inc = ticker.income_stmt
# print("Income Statement shape:", inc.shape)
# print("Sample items:", inc.index.tolist()[:10])
# print()

# # Cash flow
# cf = ticker.cashflow
# print("Cash Flow shape:", cf.shape)
# print("Sample items:", cf.index.tolist()[:10])

# ticker = yf.Ticker("AAPL")

# # Dividends
# print("Dividends:")
# print(ticker.dividends.tail())
# print()

# # Analyst recommendations
# print("Recommendations:")
# print(ticker.recommendations.tail() if ticker.recommendations is not None else "None")
# print()

# # Major holders
# print("Major Holders:")
# print(ticker.major_holders)

import yfinance as yf

def get_ticker_summary(symbol: str) -> dict:
    """
    Returns a summary dict with key financial info.
    Returns {"error": "message"} if ticker is invalid.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info

    # Check for invalid ticker (near-empty response)
    if len(info) < 10:
        return {"error": f"Invalid ticker symbol: {symbol}"}

    # Helper for price lookup (handles ETFs and stocks)
    price = (
        info.get('currentPrice')
        or info.get('regularMarketPrice')
        or info.get('previousClose')
    )

    # Helper for formatting large numbers
    def format_large_number(n):
        if n is None:
            return None
        if n >= 1e12:
            return f"${n/1e12:.2f}T"
        if n >= 1e9:
            return f"${n/1e9:.2f}B"
        if n >= 1e6:
            return f"${n/1e6:.2f}M"
        return f"${n:,.0f}"

    return {
        "symbol": symbol.upper(),
        "name": info.get('longName') or info.get('shortName'),
        "sector": info.get('sector'),
        "current_price": f"${price:.2f}" if price else None,
        "market_cap": format_large_number(info.get('marketCap')),
        "pe_ratio": round(info.get('trailingPE'), 2) if info.get('trailingPE') else None
    }

    # Valid stock
print(get_ticker_summary("AAPL"))

# ETF (no sector)
print(get_ticker_summary("SPY"))

# Invalid
print(get_ticker_summary("FAKEZZZ"))