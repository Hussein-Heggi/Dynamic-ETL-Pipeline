# polygon_client.py
import time
import pandas as pd
from typing import Dict, Any
from base_api_client import BaseAPIClient
from polygon import RESTClient  

class PolygonClient(BaseAPIClient):
    def __init__(self, api_key):
    
        self.client = RESTClient(api_key)

    def fetch_data(self, features):
        """
        Fetch data using the official client's methods.
        Uses 'endpoint_type' from features:
          - 0 -> get_aggs endpoint
          - 1 -> daily_market_summary endpoint
          - 2 -> get_daily_open_close_agg
          - 3 -> get_previous_close_agg
        Returns a dictionary with the raw response data and the features used.
        """
        
        endpoint_mapping = {
            0: lambda f: self.client.get_aggs(
                    ticker=f['ticker'],
                    multiplier=f['multiplier'],
                    timespan=f['timespan'],
                    from_=f['from'],  
                    to=f['to']
                ),
            1: lambda f: self.client.get_grouped_daily_aggs(
                    date=f['from']
                ),
            2: lambda f: self.client.get_daily_open_close_agg(
                    ticker=f['ticker'],
                    date=f['from']
                ),
            3: lambda f: self.client.get_previous_close_agg(
                    ticker=f['ticker']
                )
        }
        
        endpoint_type = features.get("endpoint_type", 0)
        if endpoint_type not in endpoint_mapping:
            raise ValueError(f"Invalid endpoint_type: {endpoint_type}")
        
        attempts = 0
        max_attempts = 3
        delay = 2  
        
        while attempts < max_attempts:
            try:
                response = endpoint_mapping[endpoint_type](features)
                return {"data": response, "features": features}
            except Exception as e:
                attempts += 1
                if attempts == max_attempts:
                    print(f"Request failed after {max_attempts} attempts: {e}")
                    raise e
                else:
                    print(f"Attempt {attempts} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    
    def parse_aggs(self):
       
        features_extracted = [
        "volume",               # 'v'
        "vwap",                 # 'vw'
        "open_price",          # 'o'
        "close_price",         # 'c'
        "high_price",          # 'h'
        "low_price",           # 'l'
        "timestamp",           # 't'
        "transactions"         # 'n'
    ]
        return features_extracted
    
    # Dedicated parser for get_grouped_daily_aggs endpoint
    def parse_grouped_daily(self):
       
        features_extracted = [
            "company_identifier",  # 'T'
            "volume",           # 'v'
            "vwap",             # 'vw'
            "open_price",       # 'o'
            "close_price",      # 'c'
            "high_price",       # 'h'
            "low_price",        # 'l'
            "timestamp",        # 't'
            "transactions"      # 'n'
        ]
        return features_extracted

    # Dedicated parser for get_daily_open_close_agg endpoint
    def parse_daily_open_close(self):
        """
        Extracts custom features such as open and close prices.
        """
        features_extracted = [
            "open_price",
            "close_price",
            "high_price",
             "company_identifier"
        ]
        return features_extracted
    
    # Dedicated parser for get_previous_close_agg endpoint


    def parse_response(self, response_package):
  
        response = response_package["data"]
        used_features = response_package["features"]
        endpoint_type = used_features.get("endpoint_type", 0)
        
        if not isinstance(response, (dict, list)):
                response = response.__dict__
           
        
        # Check the type of response and extract records accordingly.
        if isinstance(response, dict):
            if "results" in response:
                # Format 1: A dict with metadata and a 'results' key containing a list of records.
                results = response["results"]
            else:
                # Format 2: A flat dictionary representing a single record.
                results = [response]
        elif isinstance(response, list):
            results = response
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
        
        # Convert the list of records to a DataFrame.
        df = pd.DataFrame(results)
        
        # Map endpoint_type to the corresponding custom parser.
        parsers = {
            0: self.parse_aggs,
            1: self.parse_grouped_daily,
            2: self.parse_daily_open_close,
            3: self.parse_aggs
        }
        
        if endpoint_type not in parsers:
            raise ValueError(f"No parser available for endpoint_type: {endpoint_type}")
        
        # Use the appropriate parser to extract custom features.
        custom_features = parsers[endpoint_type]()
        
        print("features:",custom_features)
        
        return df, custom_features

    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute descriptive statistics on the DataFrame.
        Returns a dictionary with:
            - Descriptive statistics (mean, std, min, max, etc.)
            - Missing value counts
            - Skewness
            - Kurtosis
        """
        stats = {}
        stats["descriptive_stats"] = df.describe().to_dict()
        stats["missing_values"] = df.isnull().sum().to_dict()
        stats["skewness"] = df.skew().to_dict()
        stats["kurtosis"] = df.kurtosis().to_dict()
        return stats
