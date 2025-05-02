import time
import pandas as pd
from typing import Dict, Any, Tuple
from base_api_client import BaseAPIClient
from polygon import RESTClient

class PolygonClient(BaseAPIClient):
    def __init__(self, api_key: str):
        """
        Initializes the Polygon REST client and sets up endpoint descriptions.
        """
        self.client = RESTClient(api_key)
        # Descriptions for each endpoint; customize as needed
        self.endpoint_descriptions = {
            0: "Aggregated bars for a given ticker over a specified interval",
            1: "Grouped daily aggregates for the specified date",
            2: "Daily open/close aggregate for a given ticker and date",
            3: "Previous close aggregate for the given ticker"
        }

    def fetch_data(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch data using the official client's methods.
        Uses 'endpoint_type' from features to select:
          - 0 -> get_aggs
          - 1 -> get_grouped_daily_aggs
          - 2 -> get_daily_open_close_agg
          - 3 -> get_previous_close_agg
        Returns:
          Dict with raw response and input features.
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
        endpoint_type = features.get('endpoint_type', 0)
        if endpoint_type not in endpoint_mapping:
            raise ValueError(f"Invalid endpoint_type: {endpoint_type}")

        attempts = 0
        max_attempts = 3
        delay = 2
        while attempts < max_attempts:
            try:
                response = endpoint_mapping[endpoint_type](features)
                return {'data': response, 'features': features}
            except Exception as e:
                attempts += 1
                if attempts == max_attempts:
                    print(f"Request failed after {max_attempts} attempts: {e}")
                    raise
                print(f"Attempt {attempts} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)

    def parse_response(self, response_package: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parses raw response into DataFrame and metadata dict.
        Returns:
          - df: DataFrame with SDK-provided column names
          - info: dict of endpoint parameters and description
        """
        raw = response_package['data']
        params = response_package['features']
        et = params.get('endpoint_type', 0)

        # Normalize to list of records
        if not isinstance(raw, (dict, list)):
            raw = raw.__dict__
        records = raw.get('results', [raw]) if isinstance(raw, dict) else raw

        df = pd.DataFrame(records)

        info: Dict[str, Any] = {
            'company_identifier': params.get('ticker'),
            'start': params.get('from'),
            'end': params.get('to'),
            'timespan': params.get('timespan'),
            'multiplier': params.get('multiplier'),
            'description': self.endpoint_descriptions.get(et)
        }

        return df, info

    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute descriptive statistics on the DataFrame.
        Returns dict with:
          - descriptive_stats, missing_values, skewness, kurtosis
        """
        stats: Dict[str, Any] = {}
        stats['descriptive_stats'] = df.describe().to_dict()
        stats['missing_values'] = df.isnull().sum().to_dict()
        stats['skewness'] = df.skew().to_dict()
        stats['kurtosis'] = df.kurtosis().to_dict()
        return stats
    
    def export_to_csv(self, df: pd.DataFrame, stats: Dict[str, Any], base_path: str = "polygon_output") -> None:
        """
        Exports the DataFrame and its statistics to CSV files.
        Overwrites the files if they already exist.

        Args:
            df: The DataFrame to export.
            stats: The statistics dictionary to export.
            base_path: Prefix path for CSV files (without .csv extension).
        """
        df.to_csv(f"{base_path}_data.csv", index=False)
        pd.DataFrame(stats).to_csv(f"{base_path}_stats.csv")
