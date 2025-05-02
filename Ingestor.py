# ingestor.py
import pandas as pd
from typing import Dict, Any, List, Union
from Polygon_Client import PolygonClient
from alpha_vantage_client import AlphaVantageClient

class Ingestor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Ingestor with a registry of API clients.

        Args:
            config (Dict[str, Any]): Configuration dict containing API keys.
        """
        self.clients = {
            'polygon': PolygonClient(api_key=config.get('polygon_api_key')),
            'alpha_vantage': AlphaVantageClient(api_key=config.get('alpha_vantage_api_key')),
        }

    def process_features(
        self,
        features: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Process one or more feature requests across multiple API clients.

        Args:
            features: A single features dict or a list of them. Each must include an 'api' key to select the client.

        Returns:
            List of dicts, each containing:
              - 'client': identifier of the API used
              - 'df': Pandas DataFrame of results
              - 'gathered_features': list of features parsed
              - 'stats': computed statistics
        """
        # Normalize to list
        requests = features if isinstance(features, list) else [features]
        outputs: List[Dict[str, Any]] = []

        for feat in requests:
            feat_copy = feat.copy()
            api_choice = feat_copy.pop('api', 'polygon')
            client = self.clients.get(api_choice)
            if not client:
                raise ValueError(f"No client found for API: {api_choice}")

            raw_pkg = client.fetch_data(feat_copy)
            df, gathered = client.parse_response(raw_pkg)
            stats = client.compute_statistics(df)

            outputs.append({
                'client': api_choice,
                'df': df,
                'gathered_features': gathered,
                'stats': stats
            })

        return outputs


if __name__ == "__main__":
    # Example keys (replace with your own for real runs)
    config = {
        'polygon_api_key': 'amT2HDpKSqyIvpdbz5DY9qLwWwPDpaB0',
        'alpha_vantage_api_key': 'WXOG38FYIAUD05SZ'
    }

    ingestor = Ingestor(config)

    # Example requests for both APIs
    feature_requests = [
        {
            'api': 'polygon',
            #'ticker': 'AAPL',
            #'multiplier': 1,
            #'timespan': 'day',
            'from': '2025-01-03',
            #'to': '2025-02-25',
            'endpoint_type': 1
        }
    ]

    try:
        results = ingestor.process_features(feature_requests)
        for res in results:
            print(f"--- Results for {res['client']} ---")
            print("DataFrame head:")
            print(res['df'].head())
            print("Gathered Features:", res['gathered_features'])
            print("Stats Shape:", res['stats'])
            print()
    except Exception as e:
        print("Processing failed:", e)
