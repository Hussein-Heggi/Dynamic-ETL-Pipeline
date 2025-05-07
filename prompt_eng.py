import os
import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from openai import OpenAI
import torch

class FeatureMatcher:
    """
    A class that matches user prompts to API features using:
    - GPT for keyword extraction
    - FAISS + FinBERT for semantic similarity
    - Supports API-feature mapping structure
    """
    
    def __init__(self, api_feature_map, embedding_model="ProsusAI/finbert"):
        """
        Initialize with the API-feature mapping structure
        
        Args:
            api_feature_map: Dictionary mapping APIs to their features and descriptions
            embedding_model: Transformer model to use
        """
        self.api_feature_map = api_feature_map
        
        # Build unified feature index across all APIs
        self.feature_info = self.build_feature_index()
        self.feature_names = list(self.feature_info.keys())
        self.feature_texts = [info['description'] for info in self.feature_info.values()]
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        
        # Build FAISS index
        self.build_faiss_index()

    def build_feature_index(self):
        """
        Create a unified index of all features with their metadata
        Returns dictionary with structure:
        {
            "feature_name": {
                "description": "...",
                "apis": ["api1", "api2", ...]
            }
        }
        """
        feature_index = {}
        
        for api_name, api_data in self.api_feature_map.items():
            for feature in api_data['features']:
                if feature not in feature_index:
                    feature_index[feature] = {
                        'description': api_data['feature_descriptions'][feature],
                        'apis': []
                    }
                feature_index[feature]['apis'].append(api_name)
                
        return feature_index

    def generate_embeddings(self, texts):
        """Generate embeddings using transformers model"""
        # Tokenize input
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Perform mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Create attention mask with the same dimensions as token_embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings along axis 1, but ignore padding tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum the mask along axis 1
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask
        
        # Convert to numpy and normalize
        embeddings = mean_embeddings.numpy()
        embeddings = normalize(embeddings)
        
        return embeddings

    def build_faiss_index(self, index_type="flat", nlist=10):
        """Build FAISS index for feature embeddings"""
        self.feature_embeddings = self.generate_embeddings(self.feature_texts)
        d = self.feature_embeddings.shape[1]

        if index_type == "flat":
            self.index = faiss.IndexFlatIP(d)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.feature_embeddings)
            self.index.nprobe = min(nlist, 10)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        self.index.add(self.feature_embeddings)

    def extract_keywords_with_gpt(self, prompt):
        """Extract relevant keywords from prompt using GPT"""
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not set.")

        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

        system_prompt = (
            "you are assisting me with extracting keywords from prompts related to finance."
            "Extract relevant data fields or keywords from user requests."
            "Return only a Python list of keywords, e.g., ['keyword1', 'keyword2']."
            "Extract as many key words as possible."
            "if the timeline of such data is implied withing the prompt is implied then output keywords related to it."
        )

        response = client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=128,
            temperature=0.2
        )
        
        content = response.choices[0].message.content.strip()
        try:
            keywords = eval(content, {"__builtins__": {}})
            if not isinstance(keywords, list):
                raise ValueError("Model response is not a list.")
        except Exception:
            keywords = [word.strip() for word in content.strip("[]").replace('"', '').replace("'", '').split(',')]
        return keywords

    def semantic_match_faiss(self, phrases):
        """Find matching features using semantic similarity
        Returns dictionary where each feature appears only once with its highest score
        """
        phrase_embeddings = self.generate_embeddings(phrases)
        D, I = self.index.search(phrase_embeddings, k=1)
        
        # First pass - collect all matches
        all_matches = {}
        for i, phrase in enumerate(phrases):
            score = D[i][0]
            if score > 0.7:  # Similarity threshold
                matched_feature = self.feature_names[I[i][0]]
                if matched_feature not in all_matches or score > all_matches[matched_feature]['score']:
                    all_matches[matched_feature] = {
                        'phrase': phrase,
                        'score': round(score, 2),
                        'apis': self.feature_info[matched_feature]['apis']
                    }
        
        # Convert to phrase-keyed dictionary (optional)
        matches = {
            match['phrase']: {
                'feature': feature,
                'score': match['score'],
                'apis': match['apis']
            }
            for feature, match in all_matches.items()
        }
        
        return matches

    def match_prompt(self, prompt, target_api=None):
        """
        Match prompt to features, optionally filtering by target API
        
        Args:
            prompt: User input text
            target_api: Optional API name to filter results
        Returns:
            Dictionary of matches with feature and API information
            List of recommended APIs
            Dictionary mapping APIs to their matching features
        """
        print(f"\n Prompt: {prompt}")
        
        # Extract keywords
        keywords = self.extract_keywords_with_gpt(prompt)
        print(f" GPT Keywords: {keywords}")
        
        # Get semantic matches
        sem_matches = self.semantic_match_faiss(keywords)
        
        # Filter by target API if specified
        if target_api:
            sem_matches = {
                k: v for k, v in sem_matches.items() 
                if target_api in v['apis']
            }
        
        print(f"\n Feature Matches: {sem_matches}")
        
        # Create list of recommended APIs
        recommended_apis = set()
        api_feature_map = {}
        
        for match in sem_matches.values():
            for api in match['apis']:
                recommended_apis.add(api)
                if api not in api_feature_map:
                    api_feature_map[api] = []
                api_feature_map[api].append({
                    'feature': match['feature'],
                    'score': match['score']
                })
        
        return sem_matches, list(recommended_apis), api_feature_map

if __name__ == "__main__":
    api_feature_map = {
        "polygon_api": {
            "features": [
                "get_aggs",
                "get_grouped_daily_aggs",
                "get_daily_open_close_agg",
                "get_previous_close_agg",
                "volume",
                "vwap",
                "open_price",
                "close_price",
                "high_price",
                "low_price",
                "timestamp",
                "transactions",
                "company_identifier"
            ],
            "feature_descriptions": {
                "get_aggs": "Get aggregate bars for a stock over a given date range",
                "get_grouped_daily_aggs": "Get grouped daily bars for all stocks on a specific date",
                "get_daily_open_close_agg": "Get open, close prices and other details for a specific date",
                "get_previous_close_agg": "Get previous day's closing price and other details",
                "volume": "Trading volume for the period",
                "vwap": "Volume Weighted Average Price",
                "open_price": "Opening price for the period",
                "close_price": "Closing price for the period",
                "high_price": "Highest price during the period",
                "low_price": "Lowest price during the period",
                "timestamp": "Timestamp of the data point",
                "transactions": "Number of transactions during the period",
                "company_identifier": "Stock ticker symbol"
            }
        },
        "alpha_vantage_api": {
            "features": [
                "TIME_SERIES_DAILY",
                "TIME_SERIES_WEEKLY",
                "TIME_SERIES_MONTHLY",
                "TIME_SERIES_INTRADAY",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjusted_close",
                "dividend_amount",
                "split_coefficient"
            ],
            "feature_descriptions": {
                "TIME_SERIES_DAILY": "Daily historical time series (adjusted or unadjusted)",
                "TIME_SERIES_WEEKLY": "Weekly historical time series",
                "TIME_SERIES_MONTHLY": "Monthly historical time series",
                "TIME_SERIES_INTRADAY": "Intraday time series with configurable intervals",
                "open": "Opening price for the period",
                "high": "Highest price during the period",
                "low": "Lowest price during the period",
                "close": "Closing price for the period",
                "volume": "Trading volume for the period",
                "adjusted_close": "Close price adjusted for splits and dividends",
                "dividend_amount": "Dividend amount for the period",
                "split_coefficient": "Split coefficient for the period"
            }
        }
    }

    # Initialize with api_feature_map
    matcher = FeatureMatcher(api_feature_map, embedding_model="ProsusAI/finbert")

    prompt = "I need stock market data including opening and closing prices with trading volume"
    feature_matches, recommended_apis, api_feature_mapping = matcher.match_prompt(prompt)
    
    print("\n Recommended APIs:")
    for api in recommended_apis:
        print(f"- {api}")
    
    print("\n API-Feature Mapping:")
    for api, features in api_feature_mapping.items():
        print(f"\n{api}:")
        for feature in features:
            print(f"  - {feature['feature']} (score: {feature['score']})")