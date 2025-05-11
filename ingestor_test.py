# ingestor.py

import pandas as pd
from typing import Dict, Any, List
from Polygon_Client import PolygonClient
from alpha_vantage_client import AlphaVantageClient
from prompt_eng import FeatureMatcher

class Ingestor:
    def __init__(
        self,
        config: Dict[str, Any],
        api_feature_map: Dict[str, Any],
        prompt: str,
        embedding_model: str = "ProsusAI/finbert"
    ):
        """
        1) Call LLM to get recommended APIs from prompt.
        2) Build static feature requests with dynamic api_choice.
        3) Fetch and parse in __init__, storing DataFrames.
        """
        # set up HTTP clients
        self.clients = {
            'polygon': PolygonClient(api_key=config.get('polygon_api_key')),
            'alpha_vantage': AlphaVantageClient(api_key=config.get('alpha_vantage_api_key')),
        }
        # set up LLM matcher
        self.matcher = FeatureMatcher(api_feature_map, embedding_model)
        # call LLM for API recommendations
        _, recommended_apis, _ = self.matcher.match_prompt(prompt)
        if not recommended_apis:
            raise RuntimeError(f"No APIs recommended for prompt: {prompt!r}")

        # build feature requests with static params for now
        self.requests: List[Dict[str, Any]] = []
        for api in recommended_apis:
            if api == 'polygon':
                params = {
                    'ticker': 'AAPL',
                    'multiplier': 1,
                    'timespan': 'day',
                    'from': '2025-01-01',
                    'to': '2025-02-01',
                    'endpoint_type': 0
                }
            elif api == 'alpha_vantage':
                params = {
                    'ticker': 'AAPL',
                    'function': 'TIME_SERIES_DAILY_ADJUSTED'
                }
            else:
                params = {}
            # include api choice
            feature = {'api': api, **params}
            self.requests.append(feature)

        # fetch, parse, and store DataFrames
        self.dfs: Dict[str, pd.DataFrame] = {}
        for req in self.requests:
            api_name = req.pop('api')
            client = self.clients.get(api_name)
            if not client:
                raise ValueError(f"No client found for API: {api_name}")
            raw = client.fetch_data(req)
            df = client.parse_response(raw)
            self.dfs[api_name] = df

    def print_dfs(self) -> None:
        """Print each DataFrame for each API"""
        for api, df in self.dfs.items():
            print(f"\n=== DataFrame for {api} ===")
            print(df)

    def save_dfs(self, directory: str = '.') -> None:
        """Save each DataFrame to CSV in given directory"""
        for api, df in self.dfs.items():
            path = f"{directory}/{api}_df.csv"
            print(type(df))
            # print(df)
            df.to_csv(path, index=False)
            print(f"Saved {api} to {path}")



from matcher_config import api_feature_map

config = {
    'polygon_api_key': 'amT2HDpKSqyIvpdbz5DY9qLwWwPDpaB0',
    'alpha_vantage_api_key': 'WXOG38FYIAUD05SZ'
}
prompt = "Get daily open & close prices plus volume for AAPL over Jan 2025"

ing = Ingestor(config, api_feature_map, prompt)
ing.print_dfs()
ing.save_dfs("./output")

