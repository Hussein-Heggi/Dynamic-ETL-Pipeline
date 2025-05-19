api_endpoint_map = {
        "alpha_vantage_api": {
            "endpoints": [
                "TIME_SERIES_INTRADAY",
                "TIME_SERIES_DAILY",
                "TIME_SERIES_DAILY_ADJUSTED",
                "TIME_SERIES_WEEKLY",
                "TIME_SERIES_WEEKLY_ADJUSTED",
                "TIME_SERIES_MONTHLY",
                "TIME_SERIES_MONTHLY_ADJUSTED",
                "GLOBAL_QUOTE",
                "SYMBOL_SEARCH",
                "CURRENCY_EXCHANGE_RATE",
                "FX_INTRADAY",
                "FX_DAILY",
                "FX_WEEKLY",
                "FX_MONTHLY",
                "SMA",
                "EMA",
                "MACD",
                "RSI",
                "BBANDS",
                "VWAP"
            ],
            "endpoint_descriptions": {
                "TIME_SERIES_INTRADAY": "Get intraday time series (1min, 5min, 15min, 30min, 60min intervals)",
                "TIME_SERIES_DAILY": "Daily historical prices (last 20+ years)",
                "TIME_SERIES_DAILY_ADJUSTED": "Daily prices adjusted for splits/dividends",
                "TIME_SERIES_WEEKLY": "Weekly historical prices",
                "TIME_SERIES_WEEKLY_ADJUSTED": "Weekly prices adjusted for splits/dividends",
                "TIME_SERIES_MONTHLY": "Monthly historical prices",
                "TIME_SERIES_MONTHLY_ADJUSTED": "Monthly prices adjusted for splits/dividends",
                "GLOBAL_QUOTE": "Latest price and volume data",
                "SYMBOL_SEARCH": "Search for ticker symbols and company names",
                "CURRENCY_EXCHANGE_RATE": "Real-time forex rates",
                "FX_INTRADAY": "Intraday foreign exchange rates",
                "FX_DAILY": "Daily foreign exchange rates",
                "FX_WEEKLY": "Weekly foreign exchange rates",
                "FX_MONTHLY": "Monthly foreign exchange rates",
                "SMA": "Simple Moving Average technical indicator",
                "EMA": "Exponential Moving Average technical indicator",
                "MACD": "Moving Average Convergence Divergence technical indicator",
                "RSI": "Relative Strength Index technical indicator",
                "BBANDS": "Bollinger Bands technical indicator",
                "VWAP": "Volume Weighted Average Price (intraday only)"
            },
            "endpoint_params": {
                "TIME_SERIES_INTRADAY": {
                    "required": ["symbol", "interval"],
                    "optional": ["outputsize", "datatype", "adjusted", "extended_hours", "month", "outputsize"]
                },
                "TIME_SERIES_DAILY": {
                    "required": ["symbol"],
                    "optional": ["outputsize", "datatype"]
                },
                "TIME_SERIES_DAILY_ADJUSTED": {
                    "required": ["symbol"],
                    "optional": ["outputsize", "datatype"]
                },
                "TIME_SERIES_WEEKLY": {
                    "required": ["symbol"],
                    "optional": ["datatype"]
                },
                "TIME_SERIES_WEEKLY_ADJUSTED": {
                    "required": ["symbol"],
                    "optional": ["datatype"]
                },
                "TIME_SERIES_MONTHLY": {
                    "required": ["symbol"],
                    "optional": ["datatype"]
                },
                "TIME_SERIES_MONTHLY_ADJUSTED": {
                    "required": ["symbol"],
                    "optional": ["datatype"]
                },
                "GLOBAL_QUOTE": {
                    "required": ["symbol"],
                    "optional": ["datatype"]
                },
                "SYMBOL_SEARCH": {
                    "required": ["keywords"],
                    "optional": ["datatype"]
                },
                "CURRENCY_EXCHANGE_RATE": {
                    "required": ["from_currency", "to_currency"],
                    "optional": ["datatype"]
                },
                "FX_INTRADAY": {
                    "required": ["from_symbol", "to_symbol", "interval"],
                    "optional": ["outputsize", "datatype"]
                },
                "FX_DAILY": {
                    "required": ["from_symbol", "to_symbol"],
                    "optional": ["outputsize", "datatype"]
                },
                "FX_WEEKLY": {
                    "required": ["from_symbol", "to_symbol"],
                    "optional": ["datatype"]
                },
                "FX_MONTHLY": {
                    "required": ["from_symbol", "to_symbol"],
                    "optional": ["datatype"]
                },
                "SMA": {
                    "required": ["symbol", "interval", "time_period", "series_type"],
                    "optional": ["datatype"]
                },
                "EMA": {
                    "required": ["symbol", "interval", "time_period", "series_type"],
                    "optional": ["datatype"]
                },
                "MACD": {
                    "required": ["symbol", "interval", "series_type"],
                    "optional": ["fastperiod", "slowperiod", "signalperiod", "datatype"]
                },
                "RSI": {
                    "required": ["symbol", "interval", "time_period", "series_type"],
                    "optional": ["datatype"]
                },
                "BBANDS": {
                    "required": ["symbol", "interval", "time_period", "series_type"],
                    "optional": ["nbdevup", "nbdevdn", "matype", "datatype"]
                },
                "VWAP": {
                    "required": ["symbol", "interval"],
                    "optional": ["datatype"]
                }
            }
        },
        "polygon_api": {
            "endpoints": [
                "V2_AGGS_TICKER_RANGE",
                "V1_OPEN_CLOSE",
                "V2_AGGS_GROUPED_LOCALE_MARKET_DATE",
                "V2_AGGS_TICKER_PREV",
                "V3_TRADES",
                "V3_QUOTES",
                "V1_LAST_STOCKS",
                "V1_LAST_QUOTE_STOCKS",
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKERS",
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKER",
                "V3_REFERENCE_TICKERS",
                "V3_REFERENCE_TICKER",
                "VX_SIMPLE_MOVINGAVERAGE",
                "VX_TECHNICAL_INDICATORS"
            ],
            "endpoint_descriptions": {
                "V2_AGGS_TICKER_RANGE": "Stock price history with open/high/low/close/volume (custom timeframes: minute, hour, day, week, month)",
                "V1_OPEN_CLOSE": "Daily opening and closing prices for specific date (includes after-hours data)",
                "V2_AGGS_GROUPED_LOCALE_MARKET_DATE": "Market-wide stock prices for specific day (all tickers in exchange)",
                "V2_AGGS_TICKER_PREV": "Previous day's stock market data (closing price and trading volume)",
                "V3_TRADES": "Historical trade-by-trade data (execution prices, sizes, timestamps)",
                "V3_QUOTES": "Historical bid/ask quotes (NBBO market data with timestamps)",
                "V1_LAST_STOCKS": "Latest trade price and details (real-time last transaction)",
                "V1_LAST_QUOTE_STOCKS": "Latest bid/ask prices (real-time market quote)",
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKERS": "Current market prices for all stocks (real-time snapshot)",
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKER": "Current stock market data (price, volume, bid/ask spread)",
                "V3_REFERENCE_TICKERS": "Stock symbol search (ticker lookup and company names)",
                "V3_REFERENCE_TICKER": "Company profile data (financial reference information)",
                "VX_SIMPLE_MOVINGAVERAGE": "Stock technical analysis: Simple Moving Average (SMA values)",
                "VX_TECHNICAL_INDICATORS": "Technical indicators for stocks (RSI, MACD, Bollinger Bands)"
            },
            "endpoint_params": {
                "V2_AGGS_TICKER_RANGE": {
                    "required": ["ticker"],
                    "optional": ["multiplier", "timespan", "from", "to", "adjusted", "sort", "limit"]
                },
                "V1_OPEN_CLOSE": {
                    "required": ["ticker", "date"],
                    "optional": ["adjusted"]
                },
                "V2_AGGS_GROUPED_LOCALE_MARKET_DATE": {
                    "required": ["locale", "market", "date"],
                    "optional": ["adjusted"]
                },
                "V2_AGGS_TICKER_PREV": {
                    "required": ["ticker"],
                    "optional": ["adjusted"]
                },
                "V3_TRADES": {
                    "required": ["ticker"],
                    "optional": ["timestamp", "timestamp.lt", "timestamp.lte", "timestamp.gt", "timestamp.gte", "limit", "order", "sort"]
                },
                "V3_QUOTES": {
                    "required": ["ticker"],
                    "optional": ["timestamp", "timestamp.lt", "timestamp.lte", "timestamp.gt", "timestamp.gte", "limit", "order", "sort"]
                },
                "V1_LAST_STOCKS": {
                    "required": ["ticker"],
                    "optional": []
                },
                "V1_LAST_QUOTE_STOCKS": {
                    "required": ["ticker"],
                    "optional": []
                },
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKERS": {
                    "required": ["locale", "market"],
                    "optional": []
                },
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKER": {
                    "required": ["locale", "market", "ticker"],
                    "optional": []
                },
                "V3_REFERENCE_TICKERS": {
                    "required": [],
                    "optional": ["ticker", "type", "market", "exchange", "cusip", "cik", "date", "search", "active", "sort", "order", "limit"]
                },
                "V3_REFERENCE_TICKER": {
                    "required": ["ticker"],
                    "optional": []
                },
                "VX_SIMPLE_MOVINGAVERAGE": {
                    "required": ["ticker", "window"],
                    "optional": ["timespan", "adjusted", "series_type", "expand_underlying"]
                },
                "VX_TECHNICAL_INDICATORS": {
                    "required": ["ticker", "indicator"],
                    "optional": ["timespan", "window", "series_type", "time_period", "sd", "ma_type"]
                }
            }
        }
    }