# ingestor.py
from typing import Dict, Any
from Polygon_Client import PolygonClient

class Ingestor:
    def __init__(self, config):
        # Initialize a registry of API clients. Additional API clients can be added as needed.
        self.clients = {
            'polygon': PolygonClient(api_key=config.get('polygon_api_key')),
            # e.g., 'alpha_vantage': AlphaVantageClient(api_key=config.get('alpha_vantage_api_key'))
        }
    
    def process_features(self, features: Dict[str, Any]):
        """
        Process pre-parsed features.
        Expected features dictionary includes keys such as:
            'api'           : API client identifier (e.g., 'polygon')
            'ticker'        : Stock ticker (e.g., 'AAPL')
            'multiplier'    : Aggregation multiplier
            'timespan'      : Aggregation timespan (e.g., 'day')
            'from'          : Start date (e.g., '2023-01-01')
            'to'            : End date (e.g., '2023-06-30')
            'endpoint_type' : 0 for get_aggs, 1 for daily_market_summary
        """
        original_features = features.copy()
        api_choice = features.pop('api', 'polygon')  # default to 'polygon'
        client = self.clients.get(api_choice)
        if not client:
            raise ValueError(f"No client found for API: {api_choice}")
        
        # Fetch the raw data package using the chosen client.
        raw_data_package = client.fetch_data(features)
        results, gathered_features = client.parse_response(raw_data_package)
        stats = client.compute_statistics(results)
        
        # Check that the gathered features match the requested features.
        #self.check_features(original_features, gathered_features)
        
        return results,gathered_features,stats

    # def check_features(self, requested_features, gathered_features):
    #     """
    #     Example check: verify that the requested ticker matches the ticker in the gathered features.
    #     Extend this function to perform more comprehensive validation as needed.
    #     """
    #     if requested_features.get("ticker") != gathered_features.get("ticker"):
    #         print("Warning: Mismatch in ticker between requested features and gathered features.")
    #     else:
    #         print("Feature check passed: Requested features match gathered data.")
    
if __name__ == "__main__":
    # Example usage:
    config = {'polygon_api_key': 'amT2HDpKSqyIvpdbz5DY9qLwWwPDpaB0'}
    ingestor = Ingestor(config)
    
    # Pre-parsed features provided directly to the ingestor.
    # Try changing 'endpoint_type' between 0 and 1.
    features = {
        'api': 'polygon',
        'ticker': 'AAPL',
        'multiplier': 1,
        'timespan': 'day',
        'from': '2025-01-03',
        'to': '2025-02-25',
        'endpoint_type': 0  # 0 for get_aggs, 1 for daily_market_summary etc etc 
    }
    
    try:
        dataset,gathered_features,stats = ingestor.process_features(features)
        print("Dataset:", dataset)
        print("Gathered Features:",gathered_features)
        print("statistics",stats)
    except Exception as e:
        print("Processing failed:", e)
