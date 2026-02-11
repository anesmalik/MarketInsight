import yfinance as yf
from langchain_core.tools import tool
import pandas as pd

@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info

    # Check for invalid ticker
    if len(info) < 10:
        return f"Error: Could not find ticker '{ticker}'. Please verify the symbol."

    # Get price with fallbacks
    price = (
        info.get('currentPrice')
        or info.get('regularMarketPrice')
        or info.get('previousClose')
    )

    if price is None:
        return f"Error: Could not retrieve price for {ticker}."

    name = info.get('longName') or info.get('shortName') or ticker.upper()

    return f"{name} ({ticker.upper()}) is currently trading at ${price:.2f}"

@tool
def get_company_info(ticker: str) -> str:
    """
    Get company profile information including sector, industry, description, and key details.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info

    # Check for invalid ticker
    if len(info) < 10:
        return f"Error: Could not find ticker '{ticker}'. Please verify the symbol."

    name = info.get('longName') or info.get('shortName') or ticker.upper()
    sector = info.get('sector') or "N/A"
    industry = info.get('industry') or "N/A"
    employees = info.get('fullTimeEmployees')
    website = info.get('website') or "N/A"
    summary = info.get('longBusinessSummary') or "No description available."

    # Truncate summary to 200 chars
    if len(summary) > 200:
        summary = summary[:200] + "..."

    # Format employee count
    emp_str = f"{employees:,}" if employees else "N/A"

    return f"""{name} ({ticker.upper()})
Sector: {sector}
Industry: {industry}
Employees: {emp_str}
Website: {website}

{summary}"""

@tool
def get_historical_data(ticker: str, period: str = "1mo") -> str:
    """
    Get historical price data for a stock.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
        period: Time period - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
    """
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period=period)

    if hist.empty:
        return f"Error: Could not retrieve historical data for '{ticker}'."

    # Return last 10 days to keep response manageable
    recent = hist.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']]

    result = f"Historical data for {ticker.upper()} (last {len(recent)} trading days):\n\n"
    for date, row in recent.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        result += f"{date_str}: Open ${row['Open']:.2f}, High ${row['High']:.2f}, Low ${row['Low']:.2f}, Close ${row['Close']:.2f}, Vol {row['Volume']:,.0f}\n"

    return result


@tool
def get_balance_sheet(ticker: str) -> str:
    """
    Get the balance sheet for a company showing assets, liabilities, and equity.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    bs = ticker_obj.balance_sheet

    if bs.empty:
        return f"Error: Could not retrieve balance sheet for '{ticker}'."

    # Get most recent year
    latest = bs.iloc[:, 0]
    date = bs.columns[0].strftime('%Y-%m-%d')

    def fmt(val):
        if pd.isna(val):
            return "N/A"
        return f"${val/1e9:.2f}B"

    key_items = [
        ('Total Assets', 'Total Assets'),
        ('Total Liabilities', 'Total Liabilities Net Minority Interest'),
        ('Stockholders Equity', 'Stockholders Equity'),
        ('Cash', 'Cash And Cash Equivalents'),
        ('Total Debt', 'Total Debt'),
        ('Working Capital', 'Working Capital'),
    ]

    result = f"Balance Sheet for {ticker.upper()} (as of {date}):\n\n"
    for label, key in key_items:
        val = latest.get(key)
        result += f"{label}: {fmt(val)}\n"

    return result


@tool
def get_income_statement(ticker: str) -> str:
    """
    Get the income statement showing revenue, expenses, and profit.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    inc = ticker_obj.income_stmt

    if inc.empty:
        return f"Error: Could not retrieve income statement for '{ticker}'."

    latest = inc.iloc[:, 0]
    date = inc.columns[0].strftime('%Y-%m-%d')

    def fmt(val):
        if pd.isna(val):
            return "N/A"
        return f"${val/1e9:.2f}B"

    key_items = [
        ('Total Revenue', 'Total Revenue'),
        ('Gross Profit', 'Gross Profit'),
        ('Operating Income', 'Operating Income'),
        ('Net Income', 'Net Income'),
        ('EBITDA', 'EBITDA'),
    ]

    result = f"Income Statement for {ticker.upper()} (fiscal year ending {date}):\n\n"
    for label, key in key_items:
        val = latest.get(key)
        result += f"{label}: {fmt(val)}\n"

    return result


@tool
def get_cash_flow(ticker: str) -> str:
    """
    Get cash flow statement showing operating, investing, and financing activities.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    cf = ticker_obj.cashflow

    if cf.empty:
        return f"Error: Could not retrieve cash flow statement for '{ticker}'."

    latest = cf.iloc[:, 0]
    date = cf.columns[0].strftime('%Y-%m-%d')

    def fmt(val):
        if pd.isna(val):
            return "N/A"
        return f"${val/1e9:.2f}B"

    key_items = [
        ('Operating Cash Flow', 'Operating Cash Flow'),
        ('Capital Expenditure', 'Capital Expenditure'),
        ('Free Cash Flow', 'Free Cash Flow'),
        ('Dividends Paid', 'Cash Dividends Paid'),
        ('Stock Repurchases', 'Repurchase Of Capital Stock'),
    ]

    result = f"Cash Flow Statement for {ticker.upper()} (fiscal year ending {date}):\n\n"
    for label, key in key_items:
        val = latest.get(key)
        result += f"{label}: {fmt(val)}\n"

    return result


@tool
def get_financial_ratios(ticker: str) -> str:
    """
    Get key financial ratios including P/E, P/B, debt-to-equity, and profitability metrics.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info

    if len(info) < 10:
        return f"Error: Could not find ticker '{ticker}'."

    def fmt_ratio(val):
        if val is None:
            return "N/A"
        return f"{val:.2f}"

    def fmt_pct(val):
        if val is None:
            return "N/A"
        return f"{val*100:.2f}%"

    name = info.get('longName') or ticker.upper()

    return f"""Financial Ratios for {name} ({ticker.upper()}):

Valuation:
- Trailing P/E: {fmt_ratio(info.get('trailingPE'))}
- Forward P/E: {fmt_ratio(info.get('forwardPE'))}
- Price/Book: {fmt_ratio(info.get('priceToBook'))}
- Price/Sales: {fmt_ratio(info.get('priceToSalesTrailing12Months'))}
- PEG Ratio: {fmt_ratio(info.get('pegRatio'))}

Profitability:
- Profit Margin: {fmt_pct(info.get('profitMargins'))}
- Operating Margin: {fmt_pct(info.get('operatingMargins'))}
- Return on Equity: {fmt_pct(info.get('returnOnEquity'))}
- Return on Assets: {fmt_pct(info.get('returnOnAssets'))}

Leverage:
- Debt/Equity: {fmt_ratio(info.get('debtToEquity'))}
- Current Ratio: {fmt_ratio(info.get('currentRatio'))}"""


@tool
def get_dividends(ticker: str) -> str:
    """
    Get dividend history and current dividend yield for a stock.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    divs = ticker_obj.dividends

    if len(info) < 10:
        return f"Error: Could not find ticker '{ticker}'."

    name = info.get('longName') or ticker.upper()
    div_yield = info.get('dividendYield')
    div_rate = info.get('dividendRate')

    yield_str = f"{div_yield*100:.2f}%" if div_yield else "N/A"
    rate_str = f"${div_rate:.2f}" if div_rate else "N/A"

    result = f"""Dividend Information for {name} ({ticker.upper()}):

Current Dividend Yield: {yield_str}
Annual Dividend Rate: {rate_str}

Recent Dividend History:
"""

    if divs.empty:
        result += "No dividend history available."
    else:
        for date, amount in divs.tail(5).items():
            date_str = date.strftime('%Y-%m-%d')
            result += f"  {date_str}: ${amount:.4f}\n"

    return result


@tool
def get_splits(ticker: str) -> str:
    """
    Get stock split history for a company.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    splits = ticker_obj.splits

    if splits.empty:
        return f"No stock split history found for {ticker.upper()}."

    result = f"Stock Split History for {ticker.upper()}:\n\n"
    for date, ratio in splits.tail(10).items():
        date_str = date.strftime('%Y-%m-%d')
        result += f"  {date_str}: {ratio:.0f}-for-1 split\n"

    return result


@tool
def get_major_holders(ticker: str) -> str:
    """
    Get major shareholders breakdown including institutional and insider ownership.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    holders = ticker_obj.major_holders

    if holders is None or holders.empty:
        return f"Error: Could not retrieve holder information for '{ticker}'."

    result = f"Major Holders for {ticker.upper()}:\n\n"
    for idx, row in holders.iterrows():
        result += f"  {row['Breakdown']}: {row['Value']}\n"

    return result


@tool
def get_institutional_holders(ticker: str) -> str:
    """
    Get top institutional holders (mutual funds, pension funds, etc.) for a stock.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    inst = ticker_obj.institutional_holders

    if inst is None or inst.empty:
        return f"Error: Could not retrieve institutional holders for '{ticker}'."

    result = f"Top Institutional Holders for {ticker.upper()}:\n\n"
    for _, row in inst.head(10).iterrows():
        holder = row.get('Holder', 'Unknown')
        shares = row.get('Shares', 0)
        pct = row.get('pctHeld', 0)
        pct_str = f"{pct*100:.2f}%" if pct else "N/A"
        result += f"  {holder}: {shares:,.0f} shares ({pct_str})\n"

    return result


@tool
def get_insider_transactions(ticker: str) -> str:
    """
    Get recent insider trading activity (buys and sells by executives and directors).

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    insider = ticker_obj.insider_transactions

    if insider is None or insider.empty:
        return f"No insider transactions found for {ticker.upper()}."

    result = f"Recent Insider Transactions for {ticker.upper()}:\n\n"
    for _, row in insider.head(10).iterrows():
        insider_name = row.get('Insider', 'Unknown')
        trans_type = row.get('Transaction', 'Unknown')
        shares = row.get('Shares', 0)
        date = row.get('Start Date', 'Unknown')
        if hasattr(date, 'strftime'):
            date = date.strftime('%Y-%m-%d')
        result += f"  {date}: {insider_name} - {trans_type} {shares:,.0f} shares\n"

    return result


@tool
def get_analyst_recommendations(ticker: str) -> str:
    """
    Get analyst recommendations and ratings summary (buy, hold, sell ratings).

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    recs = ticker_obj.recommendations

    if recs is None or recs.empty:
        return f"No analyst recommendations found for {ticker.upper()}."

    # Get most recent month
    latest = recs.iloc[0]

    return f"""Analyst Recommendations for {ticker.upper()}:

Strong Buy: {latest.get('strongBuy', 0)}
Buy: {latest.get('buy', 0)}
Hold: {latest.get('hold', 0)}
Sell: {latest.get('sell', 0)}
Strong Sell: {latest.get('strongSell', 0)}

Total Analysts: {latest.get('strongBuy', 0) + latest.get('buy', 0) + latest.get('hold', 0) + latest.get('sell', 0) + latest.get('strongSell', 0)}"""


@tool
def get_price_targets(ticker: str) -> str:
    """
    Get analyst price targets including current, low, high, and mean targets.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info

    if len(info) < 10:
        return f"Error: Could not find ticker '{ticker}'."

    current = info.get('currentPrice') or info.get('regularMarketPrice')
    target_low = info.get('targetLowPrice')
    target_high = info.get('targetHighPrice')
    target_mean = info.get('targetMeanPrice')
    target_median = info.get('targetMedianPrice')

    def fmt(val):
        return f"${val:.2f}" if val else "N/A"

    upside = ""
    if current and target_mean:
        upside_pct = ((target_mean - current) / current) * 100
        upside = f" ({upside_pct:+.1f}% from current)"

    return f"""Price Targets for {ticker.upper()}:

Current Price: {fmt(current)}
Mean Target: {fmt(target_mean)}{upside}
Median Target: {fmt(target_median)}
Low Target: {fmt(target_low)}
High Target: {fmt(target_high)}"""


@tool
def get_earnings(ticker: str) -> str:
    """
    Get earnings history and upcoming earnings date.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info

    if len(info) < 10:
        return f"Error: Could not find ticker '{ticker}'."

    name = info.get('longName') or ticker.upper()
    eps_trailing = info.get('trailingEps')
    eps_forward = info.get('forwardEps')

    def fmt(val):
        return f"${val:.2f}" if val else "N/A"

    return f"""Earnings Information for {name} ({ticker.upper()}):

Trailing EPS (TTM): {fmt(eps_trailing)}
Forward EPS (Est): {fmt(eps_forward)}"""


@tool
def ticker_lookup(company_name: str) -> str:
    """
    Look up a stock ticker symbol by company name. Use this when you know the company
    name but need to find its ticker symbol.

    Args:
        company_name: Company name to search for (e.g., "Apple", "Microsoft")
    """
    import requests

    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}&quotesCount=5"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()

        quotes = data.get('quotes', [])
        if not quotes:
            return f"No ticker found for '{company_name}'."

        result = f"Ticker symbols matching '{company_name}':\n\n"
        for q in quotes[:5]:
            symbol = q.get('symbol', 'N/A')
            name = q.get('longname') or q.get('shortname', 'N/A')
            exchange = q.get('exchange', 'N/A')
            result += f"  {symbol}: {name} ({exchange})\n"

        return result
    except Exception as e:
        return f"Error searching for ticker: {str(e)}"
