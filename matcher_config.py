# matcher_config.py

api_feature_map = {
    "polygon": {
        "features": [
            "get_aggs",
            "get_grouped_daily_aggs",
            "get_daily_open_close_agg",
            "volume",
            "vwap",
            "open_price",
            "close_price",
            "high_price",
            "low_price"
        ],
        "feature_descriptions": {
            "get_aggs": "Get aggregate bars for a stock over a given date range",
            "get_grouped_daily_aggs": "Get grouped daily bars for all stocks on a specific date",
            "get_daily_open_close_agg": "Get open and close prices for a specific date",
            "volume": "Trading volume for the period",
            "vwap": "Volume Weighted Average Price",
            "open_price": "Opening price",
            "close_price": "Closing price",
            "high_price": "Highest price",
            "low_price": "Lowest price"
        }
    },
    "alpha_vantage": {
        "features": [
            "TIME_SERIES_DAILY",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ],
        "feature_descriptions": {
            "TIME_SERIES_DAILY": "Daily time series data",
            "open": "Opening price",
            "high": "High price",
            "low": "Low price",
            "close": "Closing price",
            "volume": "Volume traded"
        }
    }
}

