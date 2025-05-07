# ingestor.py
import pandas as pd
from typing import Dict, Any, List, Union
from Polygon_Client import PolygonClient
from alpha_vantage_client import AlphaVantageClient

class Ingestor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Ingestor with a registry of API clients.
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
        Process requests and always export the resulting DataFrame to CSV.
        """
        requests = features if isinstance(features, list) else [features]
        outputs: List[Dict[str, Any]] = []

        for feat in requests:
            feat_copy = feat.copy()
            api_choice = feat_copy.pop('api', 'polygon')
            client = self.clients.get(api_choice)
            if not client:
                raise ValueError(f"No client found for API: {api_choice}")

            raw_pkg = client.fetch_data(feat_copy)
            df = client.parse_response(raw_pkg)
            # stats = client.compute_statistics(df)

            # Export DataFrame to CSV file
           # identifier = gathered.get('company_identifier', api_choice)
            #start = gathered.get('start', '')
           # end = gathered.get('end', '')
            filename = f"{api_choice}_df.csv"
            df.to_csv(filename, index=False)

            outputs.append({
                'client': api_choice,
                'df': df
            })

        return outputs

if __name__ == "__main__":
    config = {
        'polygon_api_key': 'amT2HDpKSqyIvpdbz5DY9qLwWwPDpaB0',
        'alpha_vantage_api_key': 'WXOG38FYIAUD05SZ'
    }
    ingestor = Ingestor(config)
    # Example usage
    feature_requests = [
        {'api': 'polygon', 'ticker': 'AAPL', 'multiplier': 1, 'timespan': 'day', 'from': '2025-01-01', 'to': '2025-02-01', 'endpoint_type': 0}
    ]
    results = ingestor.process_features(feature_requests)
    for res in results:
        print(f"Exported CSV for {res['client']}")
